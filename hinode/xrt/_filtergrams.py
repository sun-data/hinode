from typing import Sequence
import typing
import os
import pathlib
import dataclasses
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import named_arrays as na
import hinode
from ._data import _filter

__all__ = [
    "Filtergram",
]

_value_missing = -999
"""The value ``xrt_prep`` gives to the pixels with no data."""

_level_saturation = 2500 * u.DN
"""
The signal above which ``xrt_prep`` considers a pixel saturated,
and to which it sets every saturated pixel.
"""

_center_vignetting = 1024
"""
The column and the row of the CCD, in unsummed pixels,
about which ``xrt_prep`` takes the vignetting to be symmetric.
"""

_plate_scale_vignetting = 1.0286 * u.arcsec
"""
The angle subtended by an unsummed pixel,
as ``xrt_prep`` takes it when correcting for vignetting.
"""

_angle_graze = 0.91 * u.deg
"""
The average grazing angle of the mirror,
as ``xrt_prep`` takes it when correcting for vignetting.
"""


def _vignetting(
    num_x: int,
    num_y: int,
    chip_sum: int,
    pos_col: int,
    pos_row: int,
    axis_detector_x: str,
    axis_detector_y: str,
) -> na.AbstractScalar:
    """
    The fraction of the light which reaches each pixel of an image,
    which ``xrt_prep`` divides the image by, unless it is a dark frame.

    This is a port of ``nono_vignette.pro`` in SolarSoft,
    in which the fraction falls off linearly with the angle from the center
    of the CCD.

    Parameters
    ----------
    num_x
        The number of columns of the image, ``NAXIS1``.
    num_y
        The number of rows of the image, ``NAXIS2``.
    chip_sum
        The number of pixels of the CCD summed along each axis into each
        pixel of the image, ``CHIP_SUM``.
    pos_col
        The first column of the CCD in the image, in unsummed pixels,
        ``POS_COL``.
    pos_row
        The first row of the CCD in the image, in unsummed pixels,
        ``POS_ROW``.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    """
    # The position of the image on the CCD is divided by the summing
    # in integer arithmetic, as ``nono_vignette.pro`` does.
    offset_x = pos_col // chip_sum
    offset_y = pos_row // chip_sum

    x = na.arange(0, num_x, axis=axis_detector_x) + offset_x
    y = na.arange(0, num_y, axis=axis_detector_y) + offset_y

    center = _center_vignetting / chip_sum

    radius = np.sqrt(np.square(x - center) + np.square(y - center))
    angle = radius * chip_sum * _plate_scale_vignetting

    return 1 - (2 / 3) * (angle / _angle_graze).to(u.dimensionless_unscaled)


@dataclasses.dataclass(eq=False, repr=False)
class Filtergram(
    na.FunctionArray[
        na.ExplicitTemporalWcsPositionalVectorArray,
        na.ScalarArray,
    ],
):
    """
    A sequence of Level 1 images captured by the X-Ray Telescope (XRT)
    through one filter.

    The images have been prepared by ``xrt_prep`` in SolarSoft,
    which subtracts the dark current, removes the read-out noise,
    corrects for vignetting, and divides by the exposure time,
    so the outputs are in DN per second.

    Each image is one frame along :attr:`axis_time`,
    and its coordinates come from its own header,
    since the pointing changes from one image to the next.

    Examples
    --------

    Load the Al_poly images captured while the EUV Snapshot Imaging
    Spectrograph (ESIS) was observing the Sun on 2019 September 30,
    and display the one with the most saturated pixels,
    which the jet called event E saturates.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import hinode

        # Load the images captured during the ESIS flight
        images = hinode.xrt.open(
            time_start="2019-09-30T18:06:11",
            time_stop="2019-09-30T18:11:01",
            filter="Al_poly",
        )

        # Select the image with the most saturated pixels
        num_saturated = images.saturated.sum(
            axis=(images.axis_detector_x, images.axis_detector_y),
        )
        index = {images.axis_time: int(np.argmax(num_saturated.ndarray))}
        image = images[index]

        # Display the image
        unit = image.inputs.position.x.unit
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
            na.plt.pcolormesh(
                image.inputs.position.x,
                image.inputs.position.y,
                C=image.outputs,
                ax=ax,
                cmap="gray",
                norm="asinh",
                vmin=0 * u.DN / u.s,
                vmax=100 * u.DN / u.s,
            )
            ax.set_aspect("equal")
            ax.set_title(image.inputs.time.ndarray)
            ax.set_xlabel(f"helioprojective $x$ ({unit:latex_inline})")
            ax.set_ylabel(f"helioprojective $y$ ({unit:latex_inline})")

        image.saturated.sum()
    """

    timedelta: u.Quantity | na.AbstractScalar = dataclasses.field(
        default_factory=lambda: 0 * u.s,
    )
    """The exposure time of each image."""

    saturated: None | na.ScalarArray = None
    """
    Whether each pixel is saturated, or charge has bled into it from a
    saturated pixel.

    ``xrt_prep`` sets every pixel above the saturation level, 2500 DN,
    to exactly 2500 DN before correcting for vignetting and dividing by the
    exposure time, so a pixel is saturated if the product of its value,
    the exposure time, and the vignetting function of ``xrt_prep`` is at
    least 2500 DN.
    The level is lowered by 0.01% to allow for the rounding of the exposure
    time in the header and of the values stored in the file.

    The CCD bleeds the charge of a saturated pixel along its columns,
    so the pixels just above and just below each saturated pixel,
    which ``xrt_prep`` grades as possibly affected by "bloom/bleed",
    are marked as well.
    """

    filter: None | str = None
    """
    The filters the images were taken through,
    named as in :func:`hinode.xrt.urls`.
    """

    axis_time: str = "time"
    """The logical axis corresponding to changes in time."""

    axis_detector_x: str = "detector_x"
    """The logical axis corresponding to changes in detector :math:`x`-coordinate."""

    axis_detector_y: str = "detector_y"
    """The logical axis corresponding to changes in detector :math:`y`-coordinate."""

    @classmethod
    def from_time_range(
        cls,
        time_start: str | astropy.time.Time,
        time_stop: str | astropy.time.Time,
        filter: str = "Al_poly",
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
        directory: None | pathlib.Path = None,
        overwrite: bool = False,
        num_retry: int = 5,
    ) -> "Filtergram":
        """
        Download the Level 1 images which began during a given time range
        and were taken through a given filter,
        and construct an instance of :class:`Filtergram`.

        Parameters
        ----------
        time_start
            The earliest start time of the images.
        time_stop
            The time before which an image must begin.
        filter
            The filters the images were taken through,
            named as in :func:`hinode.xrt.urls`.
        axis_time
            The logical axis corresponding to changes in time.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.
        directory
            The directory to place the downloaded files in.
            If :obj:`None` (the default), :data:`hinode.directory_default` is used.
        overwrite
            Boolean flag controlling whether to download files which are already
            in `directory`.
        num_retry
            The number of times to try to connect to the server.
        """
        urls = hinode.xrt.urls(
            time_start=time_start,
            time_stop=time_stop,
            filter=filter,
            axis_time=axis_time,
            num_retry=num_retry,
        )

        if urls.size == 0:
            raise ValueError(
                f"No {filter} images began between {time_start} and {time_stop}."
            )

        paths = hinode.xrt.download(
            urls=urls,
            directory=directory,
            overwrite=overwrite,
            num_retry=num_retry,
        )

        return cls.from_fits(
            path=paths,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    @classmethod
    def from_fits(
        cls,
        path: str | os.PathLike | Sequence[str | os.PathLike] | na.AbstractScalarArray,
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ) -> "Filtergram":
        """
        Load one or more Level 1 XRT files taken through the same filter.

        The images are placed one after another along `axis_time`,
        in the order the files are given,
        and images smaller than the largest one are padded with NaN.

        Parameters
        ----------
        path
            A Level 1 XRT file, or a sequence of them,
            or an array of them such as the one :func:`hinode.xrt.download`
            returns, which is flattened.
        axis_time
            The logical axis corresponding to changes in time.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.

        Notes
        -----
        The pixels with no data, which ``xrt_prep`` sets to -999,
        are set to NaN.

        The headers give the roll of each image as ``CROTA2`` rather than as
        a ``PC`` matrix, and it is converted to one the way
        :cite:t:`Calabretta2002` describe for that older keyword.
        """
        if isinstance(path, na.AbstractScalarArray):
            path = [str(p) for p in np.ravel(np.asarray(path.ndarray))]
        elif isinstance(path, (str, os.PathLike)):
            path = [path]

        if len(path) == 0:
            raise ValueError("No files were given.")

        headers = [astropy.io.fits.getheader(p) for p in path]

        filters = list(dict.fromkeys(_filter(h) for h in headers))
        if len(filters) > 1:
            raise ValueError(f"The files are of different filters, {filters}.")

        num_t = len(headers)
        shape_y = max(int(h["NAXIS2"]) for h in headers)
        shape_x = max(int(h["NAXIS1"]) for h in headers)

        axes = (axis_time, axis_detector_y, axis_detector_x)

        # Filled in place, so that the images are held in memory only once
        images = np.full((num_t, shape_y, shape_x), np.nan, dtype=np.float32)
        saturated = na.ScalarArray(np.zeros(images.shape, dtype=bool), axes=axes)
        for i, (p, header) in enumerate(zip(path, headers)):
            data = np.asarray(astropy.io.fits.getdata(p, memmap=False))
            num_y, num_x = data.shape
            images[i, :num_y, :num_x] = data

            # The signal before ``xrt_prep`` divided it by the vignetting
            # function and by the exposure time.
            # It is found one image at a time,
            # so that it needs no more memory than one image.
            image = na.ScalarArray(
                ndarray=data << (u.DN / u.s),
                axes=(axis_detector_y, axis_detector_x),
            )
            signal = image * (header["EXPTIME"] * u.s)
            if header["EC_IMTY_"] != "dark":
                signal = signal * _vignetting(
                    num_x=num_x,
                    num_y=num_y,
                    chip_sum=int(header["CHIP_SUM"]),
                    pos_col=int(header["POS_COL"]),
                    pos_row=int(header["POS_ROW"]),
                    axis_detector_x=axis_detector_x,
                    axis_detector_y=axis_detector_y,
                )

            index = {
                axis_time: i,
                axis_detector_y: slice(None, num_y),
                axis_detector_x: slice(None, num_x),
            }
            saturated[index] = signal >= 0.9999 * _level_saturation

        images[images == _value_missing] = np.nan

        def scalar(key: str, unit: u.UnitBase | None = None) -> na.ScalarArray:
            a = np.array([h[key] for h in headers], dtype=float)
            if unit is not None:
                a = a << unit
            return na.ScalarArray(a, axes=axis_time)

        def angle(key: str, key_unit: str) -> na.ScalarArray:
            a = [h[key] * u.Unit(h[key_unit]) for h in headers]
            return na.ScalarArray(u.Quantity(a).to(u.arcsec), axes=axis_time)

        timedelta = scalar("EXPTIME", u.s)

        # Several of the headers say ``TIMESYS = 'UTC (TBR)'``
        time = astropy.time.Time([h["DATE_OBS"] for h in headers], scale="utc")
        time.format = "isot"

        # A named array holds a time the way it holds an ndarray,
        # which its annotations do not say.
        time = typing.cast(np.ndarray, time)

        cdelt_x = angle("CDELT1", "CUNIT1")
        cdelt_y = angle("CDELT2", "CUNIT2")

        roll = scalar("CROTA2", u.deg)
        cos = np.cos(roll)
        sin = np.sin(roll)

        inputs = na.ExplicitTemporalWcsPositionalVectorArray(
            time=na.ScalarArray(time, axes=axis_time),
            crval=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(
                    x=angle("CRVAL1", "CUNIT1"),
                    y=angle("CRVAL2", "CUNIT2"),
                ),
            ),
            # One less than the FITS keyword, which counts pixels from one
            # where :class:`named_arrays.AbstractWcsVector` counts them from
            # zero.
            crpix=na.CartesianNdVectorArray(
                components={
                    axis_detector_x: scalar("CRPIX1") - 1,
                    axis_detector_y: scalar("CRPIX2") - 1,
                },
            ),
            cdelt=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(
                    x=cdelt_x,
                    y=cdelt_y,
                ),
            ),
            pc=na.PositionalMatrixArray(
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(
                        components={
                            axis_detector_x: cos,
                            axis_detector_y: -sin * cdelt_y / cdelt_x,
                        },
                    ),
                    y=na.CartesianNdVectorArray(
                        components={
                            axis_detector_x: sin * cdelt_x / cdelt_y,
                            axis_detector_y: cos,
                        },
                    ),
                ),
            ),
            shape_wcs={
                axis_detector_x: shape_x + 1,
                axis_detector_y: shape_y + 1,
            },
        )

        outputs = na.ScalarArray(images << (u.DN / u.s), axes=axes)

        # Charge bleeds along the columns of the CCD
        above = {axis_detector_y: slice(1, None)}
        below = {axis_detector_y: slice(None, ~0)}
        bleed = saturated.copy()
        bleed[above] = typing.cast(na.ScalarArray, bleed[above] | saturated[below])
        bleed[below] = typing.cast(na.ScalarArray, bleed[below] | saturated[above])
        bleed = typing.cast(na.ScalarArray, bleed & np.isfinite(outputs))

        return cls(
            inputs=inputs,
            outputs=outputs,
            timedelta=timedelta,
            saturated=bleed,
            filter=filters[0],
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )
