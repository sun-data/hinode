from typing import Sequence, Literal
import typing
import os
import re
import pathlib
import dataclasses
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import named_arrays as na
import hinode
from ._data import _filter, _history
from ._leak import _leak, _history_leak
from ._uncertainty import _quality, _error_jpeg, _error_dark
from ._vignetting import _vignetting, _error_vignetting
from ._coalign import _coalign

__all__ = [
    "Filtergram",
]

_value_missing = -999
"""The value ``xrt_prep`` gives to the pixels with no data."""

_pattern_normalized = re.compile(
    r"\(XRT_RENORMALIZE\) Normalized from [\d.]+ sec --> ([\d.]+) sec"
)
"""
The entry in the history of a file which ``xrt_prep`` writes when it divides
the image by its exposure time,
and the exposure time it normalized the image to.
"""

_pattern_saturated = re.compile(
    r"\(XRT_SATURATED_PIXELS\) Replaced \d+ saturated pixels with value = ([\d.]+)"
)
"""
The entry in the history of a file which ``xrt_prep`` writes when it sets
every pixel above the saturation level to that level,
and the level.
"""


def _check_normalized(
    header: astropy.io.fits.Header,
    path: str | os.PathLike,
) -> None:
    """
    Check that ``xrt_prep`` divided an image by its exposure time,
    which it does only if asked to.

    Parameters
    ----------
    header
        The primary header of an XRT file.
    path
        The file, for the error message.

    Raises
    ------
    ValueError
        If the image is not in DN per second.
    """
    exposures = _pattern_normalized.findall(_history(header))
    if not exposures or float(exposures[-1]) != 1:
        raise ValueError(
            f"{path} is not in DN per second, "
            "since xrt_prep did not normalize it to an exposure time of 1 s, "
            "which its /normalize keyword does."
        )


def _level_saturation(
    header: astropy.io.fits.Header,
    path: str | os.PathLike,
) -> u.Quantity:
    """
    The signal above which ``xrt_prep`` considered a pixel of an image
    saturated, and which it set every saturated pixel to,
    as the history of the file records it.

    Parameters
    ----------
    header
        The primary header of an XRT file.
    path
        The file, for the error message.

    Raises
    ------
    ValueError
        If the history of the file does not record the level.
    """
    levels = _pattern_saturated.findall(_history(header))
    if not levels:
        raise ValueError(
            f"The history of {path} does not say "
            "which value xrt_prep set its saturated pixels to."
        )
    return float(levels[-1]) * u.DN


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
    By default the pointing is corrected with the co-alignment database of
    SolarSoft from the cross-correlation of XRT with AIA
    :cite:p:`Yoshimura2015`,
    so that the images line up with those of AIA,
    as :attr:`coalignment` records.

    The time of each image, ``inputs.time``, is the start of its exposure,
    ``DATE_OBS``, as in the other sun-data packages,
    so the middle of each exposure is ``inputs.time + timedelta / 2``.

    Since 2012, visible light has leaked into XRT, which ``xrt_prep`` does
    not subtract.
    Load it with ``leak=True`` and subtract it with :meth:`remove_leak`.
    The uncertainty of each pixel is found with :meth:`uncertainty`.

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
        image = images[np.argmax(num_saturated, axis=images.axis_time)]

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
    to exactly that level before correcting for vignetting and dividing by
    the exposure time, and records the level in the history of the file,
    so a pixel is saturated if the product of its value,
    the exposure time, and the vignetting function of ``xrt_prep`` is at
    least the recorded level.
    The level is lowered by 0.01% to allow for the rounding of the exposure
    time in the header and of the values stored in the file.

    The CCD bleeds the charge of a saturated pixel along its columns,
    so the pixels just above and just below each saturated pixel,
    which ``xrt_prep`` grades as possibly affected by "bloom/bleed",
    are marked as well.
    """

    vignetting: None | na.ScalarArray = None
    """
    The fraction of the light which reaches each pixel,
    which ``xrt_prep`` divided each image by,
    from the model of ``nono_vignette.pro`` in SolarSoft,
    or 1 for a dark frame, which ``xrt_prep`` does not correct for vignetting,
    or :obj:`None` if the uncertainty was not loaded.
    """

    uncertainty_map: None | na.ScalarArray = None
    """
    The uncertainty of each pixel due to the camera and to the processing of
    the image, in DN per second,
    as the uncertainty map of ``xrt_prep``, ``uncert_map``, gives it
    :cite:p:`Kobelski2014`:
    the error of the JPEG compression, of the dark subtraction,
    and of the vignetting correction,
    or :obj:`None` if the uncertainty was not loaded.

    It does not include the photon noise, which depends on the spectrum of
    the plasma, so the whole uncertainty is found with :meth:`uncertainty`.

    It differs from the map of ``xrt_prep`` in three ways.
    ``xrt_prep`` finds the error of the JPEG compression from the range of
    the signal of the Level 0 image in each block of 8 by 8 pixels,
    and here the signal is found from the Level 1 image instead,
    by undoing the vignetting correction and the division by the exposure
    time.
    ``xrt_prep`` gives an image which was not compressed with a loss an
    error of 1 DN for its compression,
    from the value of -1 by which ``get_jpeg_unc.pro`` signals that it has no
    model for it,
    and here such an image has no error for its compression.
    ``xrt_prep`` also includes an empirical model of the error of its Fourier
    filtering of the read-out noise, which is left out here.
    """

    leak: None | na.ScalarArray = None
    """
    The visible light which leaks into each pixel and has not been
    subtracted, in DN per second,
    or :obj:`None` if the leak was not loaded.

    Since the entrance filters of XRT began to tear in 2012,
    visible light has leaked into the X-ray filters :cite:p:`Takeda2016`.
    The leak is found from the image of the leak SolarSoft has for the
    filter and the stray-light phase of each image,
    which ``xrt_synleaksub.pro`` selects,
    cut out and resampled to the part of the CCD each image covers.
    It is zero before the first tear, on 2012 May 9,
    for dark frames, which no light reaches,
    for files whose history says that SolarSoft has already subtracted it,
    and once :meth:`remove_leak` has subtracted it.

    The images of the leak were taken with Hinode pointed near the center
    of the Sun, and the pattern of the leak changes with the pointing,
    so it is most reliable for images near the center of the Sun.
    It also varies by about 10% with the contamination of the CCD.
    """

    filter: None | str = None
    """
    The filters the images were taken through,
    named as in :func:`hinode.xrt.urls`.
    """

    coalignment: None | na.ScalarArray = None
    """
    How the pointing of each image was found,
    numbered as ``CALIBRATION_TYPE`` of ``xrt_read_coaldb.pro`` in SolarSoft:
    1 for the cross-correlation with AIA,
    2 for a fit to the limb of the Sun in G-band,
    3 for the Ultra Fine Sun Sensors (UFSS),
    4 for a fit to the limb in X-rays,
    6 for another method,
    and -1 for an image no database had an entry for,
    which keeps the pointing of its header,
    or :obj:`None` if the pointing was not corrected.
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
        leak: bool = False,
        uncertainty: bool = False,
        coalign: None | Literal["aia", "ufss"] = "aia",
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
        leak
            Whether to load the visible light leaking into each pixel,
            :attr:`leak`.
        uncertainty
            Whether to load :attr:`vignetting` and :attr:`uncertainty_map`,
            which :meth:`uncertainty` needs.
        coalign
            Which co-alignment database of SolarSoft to correct the pointing
            of the images with, as ``xrt_read_coaldb.pro`` does,
            or :obj:`None` to keep the pointing of the headers.
            ``"aia"`` (the default) uses the cross-correlation of each image
            with AIA 335 Å :cite:p:`Yoshimura2015`,
            and the Ultra Fine Sun Sensors (UFSS) of Hinode for an image it
            has no entry for,
            as ``xrt_read_coaldb.pro`` does with ``/aia_cc``.
            ``"ufss"`` uses the UFSS alone, as it does by default.
            See :attr:`coalignment`.
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
            leak=leak,
            uncertainty=uncertainty,
            coalign=coalign,
            directory=directory,
            num_retry=num_retry,
        )

    @classmethod
    def from_fits(
        cls,
        path: str | os.PathLike | Sequence[str | os.PathLike] | na.AbstractScalarArray,
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
        leak: bool = False,
        uncertainty: bool = False,
        coalign: None | Literal["aia", "ufss"] = "aia",
        directory: None | pathlib.Path = None,
        num_retry: int = 5,
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
        leak
            Whether to load the visible light leaking into each pixel,
            :attr:`leak`, from SolarSoft.
        uncertainty
            Whether to load :attr:`vignetting` and :attr:`uncertainty_map`,
            which :meth:`uncertainty` needs.
            They are not loaded by default,
            since each takes as much memory as the images.
        coalign
            Which co-alignment database of SolarSoft to correct the pointing
            of the images with, as ``xrt_read_coaldb.pro`` does,
            or :obj:`None` to keep the pointing of the headers.
            ``"aia"`` (the default) uses the cross-correlation of each image
            with AIA 335 Å :cite:p:`Yoshimura2015`,
            and the Ultra Fine Sun Sensors (UFSS) of Hinode for an image it
            has no entry for,
            as ``xrt_read_coaldb.pro`` does with ``/aia_cc``.
            ``"ufss"`` uses the UFSS alone, as it does by default.
            See :attr:`coalignment`.
        directory
            The directory to place the downloaded images of the leak and the
            co-alignment databases in.
            If :obj:`None` (the default), :data:`hinode.directory_default` is used.
        num_retry
            The number of times to try to connect to the server
            for the co-alignment databases.

        Raises
        ------
        ValueError
            If ``xrt_prep`` did not divide an image by its exposure time,
            which it does only if asked to with its ``/normalize`` keyword,
            if the history of a file does not record the value
            ``xrt_prep`` set the saturated pixels to,
            if `leak` is :obj:`True` and SolarSoft has no image of the leak
            for the filter and the time of an image,
            if `uncertainty` is :obj:`True` and the compression of an
            image is not known,
            or if `coalign` is not ``"aia"``, ``"ufss"``, or :obj:`None`.

        Notes
        -----
        The pixels with no data, which ``xrt_prep`` sets to -999,
        are set to NaN.

        The headers give the roll of each image as ``CROTA2`` rather than as
        a ``PC`` matrix, and it is converted to one the way
        :cite:t:`Calabretta2002` describe for that older keyword.

        ``xrt_prep`` has already corrected the pointing of the Level 1 files
        with the UFSS, so the center of an image moves only with ``"aia"``,
        the default.
        Either database also gives each image the roll ``xrt_rollangle.pro``
        gives it now.
        Since the attitude anomaly of 2021 December 27,
        the roll is interpolated in a database which grows as the roll is
        measured,
        and some of the files ``xrt_prep`` made before it existed have a roll
        in their header which is more than a degree from it,
        and up to 23 degrees during the anomaly.
        Like ``xrt_read_coaldb.pro``,
        the correction places the center the database gives at pixel
        ``NAXIS // 2 + 0.5``, counted from one,
        with the plate scale of the X-ray or the G-band images.
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

        # Checked before any of the images are read
        for p, header in zip(path, headers):
            _check_normalized(header, p)
        levels = [_level_saturation(header, p) for p, header in zip(path, headers)]

        coalignment = None
        if coalign is not None:
            keywords, calibration = _coalign(
                time=[h["DATE_OBS"] for h in headers],
                filter_2=[int(h["EC_FW2"]) for h in headers],
                chip_sum=[int(h["CHIP_SUM"]) for h in headers],
                num_x=[int(h["NAXIS1"]) for h in headers],
                num_y=[int(h["NAXIS2"]) for h in headers],
                coalign=coalign,
                directory=directory,
                num_retry=num_retry,
            )
            for header, keywords_header in zip(headers, keywords):
                header.update(keywords_header)
            coalignment = na.ScalarArray(calibration, axes=axis_time)

        shape = {
            axis_time: len(headers),
            axis_detector_y: max(int(h["NAXIS2"]) for h in headers),
            axis_detector_x: max(int(h["NAXIS1"]) for h in headers),
        }

        # Filled in place, one image at a time,
        # so that the images are held in memory only once,
        # and the arrays made along the way are no larger than one image.
        unit_outputs = u.DN / u.s
        outputs = na.ScalarArray.full(shape, np.nan, dtype=np.float32) << unit_outputs
        outputs = typing.cast(na.ScalarArray, outputs)
        saturated = na.ScalarArray.zeros(shape, dtype=bool)
        vignetting = None
        uncertainty_map = None
        if uncertainty:
            vignetting = na.ScalarArray.full(shape, np.nan, dtype=np.float32)
            uncertainty_map = na.ScalarArray.full(shape, np.nan, dtype=np.float32)
            uncertainty_map = uncertainty_map << unit_outputs
            uncertainty_map = typing.cast(na.ScalarArray, uncertainty_map)
        leaks = None
        if leak:
            leaks = na.ScalarArray.full(shape, np.nan, dtype=np.float32)
            leaks = typing.cast(na.ScalarArray, leaks << unit_outputs)

        # Charge bleeds along the columns of the CCD
        above = {axis_detector_y: slice(1, None)}
        below = {axis_detector_y: slice(None, ~0)}

        for i, (p, header, level) in enumerate(zip(path, headers, levels)):
            data = astropy.io.fits.getdata(p, memmap=False)
            image = na.ScalarArray(
                ndarray=np.asarray(data, dtype=np.float32),
                axes=(axis_detector_y, axis_detector_x),
            )
            image[image == _value_missing] = np.nan
            image = typing.cast(na.ScalarArray, image << unit_outputs)

            num_x = image.shape[axis_detector_x]
            num_y = image.shape[axis_detector_y]

            chip_sum = int(header["CHIP_SUM"])

            dark = header["EC_IMTY_"] == "dark"

            vignetting_image: float | na.ScalarArray = 1.0
            if not dark:
                vignetting_image = typing.cast(
                    na.ScalarArray,
                    _vignetting(
                        num_x=num_x,
                        num_y=num_y,
                        chip_sum=chip_sum,
                        pos_col=int(header["POS_COL"]),
                        pos_row=int(header["POS_ROW"]),
                        axis_detector_x=axis_detector_x,
                        axis_detector_y=axis_detector_y,
                    ),
                )

            # The signal before ``xrt_prep`` divided it by the vignetting
            # function and by the exposure time.
            exposure = header["EXPTIME"] * u.s
            signal = typing.cast(na.ScalarArray, image * exposure * vignetting_image)

            saturated_image = typing.cast(na.ScalarArray, signal >= 0.9999 * level)
            bleed = saturated_image.copy()
            bleed[above] = typing.cast(
                na.ScalarArray, bleed[above] | saturated_image[below]
            )
            bleed[below] = typing.cast(
                na.ScalarArray, bleed[below] | saturated_image[above]
            )

            index = {
                axis_time: i,
                axis_detector_y: slice(None, num_y),
                axis_detector_x: slice(None, num_x),
            }
            outputs[index] = image
            saturated[index] = bleed & np.isfinite(image)

            if vignetting is not None and uncertainty_map is not None:
                vignetting[index] = vignetting_image

                error_vignetting: float | na.ScalarArray = 0.0
                if not dark:
                    error_vignetting = _error_vignetting(
                        num_x=num_x,
                        num_y=num_y,
                        chip_sum=chip_sum,
                        p1_col=int(header["P1COL"]),
                        p1_row=int(header["P1ROW"]),
                        axis_detector_x=axis_detector_x,
                        axis_detector_y=axis_detector_y,
                    )

                # The errors of the camera are in DN of the signal,
                # before the vignetting correction and the normalization.
                quality = _quality(
                    compression=int(header["IMGCOMP1"]),
                    table=int(header["QTABLE1"]),
                )
                error_jpeg = _error_jpeg(
                    signal=signal,
                    quality=quality,
                    axis_detector_x=axis_detector_x,
                    axis_detector_y=axis_detector_y,
                )
                error_dark = _error_dark(
                    history=_history(header),
                    chip_sum=chip_sum,
                    num_x=num_x,
                    num_y=num_y,
                    quality=quality,
                )
                error_camera = np.sqrt(np.square(error_jpeg) + np.square(error_dark))
                uncertainty_map[index] = np.sqrt(
                    np.square(error_camera / (vignetting_image * exposure))
                    + np.square(error_vignetting * image)
                )

            if leaks is not None:
                # No light reaches the CCD of a dark frame,
                # and SolarSoft records its subtraction of the leak in the
                # history, as ``xrt_synleaksub.pro`` checks.
                if dark or _history_leak in _history(header):
                    leaks[index] = 0 * unit_outputs
                else:
                    leaks[index] = _leak(
                        filter=filters[0],
                        time=astropy.time.Time(header["DATE_OBS"], scale="utc"),
                        num_x=num_x,
                        num_y=num_y,
                        chip_sum=chip_sum,
                        pos_col=int(header["POS_COL"]),
                        pos_row=int(header["POS_ROW"]),
                        axis_detector_x=axis_detector_x,
                        axis_detector_y=axis_detector_y,
                        directory=directory,
                    )

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
                axis_detector_x: shape[axis_detector_x] + 1,
                axis_detector_y: shape[axis_detector_y] + 1,
            },
        )

        return cls(
            inputs=inputs,
            outputs=outputs,
            timedelta=timedelta,
            saturated=saturated,
            vignetting=vignetting,
            uncertainty_map=uncertainty_map,
            leak=leaks,
            filter=filters[0],
            coalignment=coalignment,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    def remove_leak(
        self,
        scale: float = 1,
    ) -> "Filtergram":
        """
        Subtract the visible light leaking into each pixel, :attr:`leak`,
        from the images, as ``xrt_synleaksub.pro`` in SolarSoft does.

        The result has no leak left to subtract, so its :attr:`leak` is zero,
        and subtracting it again changes nothing.

        Parameters
        ----------
        scale
            The factor to multiply the leak by before subtracting it,
            ``kfact`` in ``xrt_synleaksub.pro``.

        Examples
        --------

        The median signal of the Al_poly images captured while ESIS was
        observing the Sun, before and after the leak is subtracted.

        .. jupyter-execute::

            import hinode

            images = hinode.xrt.open(
                time_start="2019-09-30T18:06:11",
                time_stop="2019-09-30T18:11:01",
                leak=True,
            )

            axes = (images.axis_detector_x, images.axis_detector_y)

            print(images.outputs.median(axes))
            print(images.remove_leak().outputs.median(axes))
        """
        if self.leak is None:
            raise ValueError(
                "The leak was not loaded, so it cannot be removed. "
                "Load the images with `leak=True`."
            )

        outputs = typing.cast(na.ScalarArray, self.outputs - scale * self.leak)
        leak = typing.cast(na.ScalarArray, 0 * self.leak)

        return dataclasses.replace(self, outputs=outputs, leak=leak)

    def uncertainty(
        self,
        noise_photon: u.Quantity | na.AbstractScalar,
    ) -> na.ScalarArray:
        r"""
        The one-sigma uncertainty of each pixel, in DN per second.

        This is the photon noise of the signal combined with the uncertainty
        due to the camera and the processing of the image,
        :attr:`uncertainty_map`,

        .. math::

            \sigma^2 = \frac{F \max(I, 0)}{V t} + \sigma_\text{map}^2,

        where :math:`I` is the signal of the pixel, :attr:`outputs`,
        :math:`V` is the vignetting, :attr:`vignetting`,
        :math:`t` is the exposure time, :attr:`timedelta`,
        and :math:`F` is the factor which sets the photon noise
        :cite:p:`Kobelski2014`.

        The photons of each wavelength give a different signal :math:`s`,
        so the variance of the signal due to the counting of the photons is
        :math:`F = \langle s^2 \rangle / \langle s \rangle` times the
        signal before the vignetting correction and the normalization.
        :math:`F` depends on the spectrum, so it is given.
        For plasma at one temperature it is the ratio of the response of
        :func:`hinode.xrt.temperature_response` with ``variance=True``
        to the response without it,
        and for a differential emission measure (DEM) it is the ratio of the
        two responses each weighted by the DEM.
        Subtract the leak first, with :meth:`remove_leak`,
        since visible photons give much less signal each than X-rays.

        Parameters
        ----------
        noise_photon
            The factor which sets the photon noise,
            :math:`\langle s^2 \rangle / \langle s \rangle`,
            in DN per photon, which may vary from pixel to pixel.

        Examples
        --------

        The median relative uncertainty of the Al_poly images captured while
        ESIS was observing the Sun,
        with the photon noise of plasma at 2 MK.

        .. jupyter-execute::

            import numpy as np
            import astropy.units as u
            import hinode

            images = hinode.xrt.open(
                time_start="2019-09-30T18:06:11",
                time_stop="2019-09-30T18:11:01",
                leak=True,
                uncertainty=True,
            ).remove_leak()

            time = images.inputs.time.ndarray[0]
            response = hinode.xrt.temperature_response("Al_poly", time)
            variance = hinode.xrt.temperature_response("Al_poly", time, variance=True)

            # The temperature closest to 2 MK
            index = dict(
                filter=0,
                temperature=int(np.argmin(np.abs(response.inputs.ndarray - 2 * u.MK))),
            )
            noise_photon = (variance.outputs / response.outputs)[index]

            uncertainty = images.uncertainty(noise_photon)

            axes = (images.axis_detector_x, images.axis_detector_y)
            (uncertainty / images.outputs).median(axes)
        """
        if self.vignetting is None or self.uncertainty_map is None:
            raise ValueError(
                "The vignetting and the uncertainty map were not loaded. "
                "Load the images with `uncertainty=True`."
            )

        # A negative signal, which the read-out noise and the subtraction of
        # the leak leave in faint pixels, has no photon noise.
        signal = (self.outputs.value > 0) * self.outputs
        factor = na.as_named_array(noise_photon) * u.ph

        variance = signal * factor / (self.vignetting * self.timedelta)
        variance = variance + np.square(self.uncertainty_map)

        return typing.cast(na.ScalarArray, np.sqrt(variance))
