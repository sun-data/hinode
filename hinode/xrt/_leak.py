import typing
import pathlib
import functools
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import named_arrays as na
from ._data import download

__all__ = []

_url_leak = "https://sohoftp.nascom.nasa.gov/solarsoft/hinode/xrt/idl/util/leak_fits/"
"""
The directory of SolarSoft which holds the images of the visible light
leaking into XRT.
"""

_start_phase = astropy.time.Time(
    [
        "2012-05-09T12:00",
        "2015-06-14T12:30",
        "2017-05-27T11:00",
        "2018-05-29T00:00",
        "2022-06-08T12:40",
        "2023-05-05T04:30",
        "2024-05-11T06:15",
    ],
    scale="utc",
)
"""
The start of each stray-light phase after the first,
as ``check_sl_phase.pro`` in SolarSoft gives them.

Visible light has leaked into XRT since the entrance filters began to tear
in 2012, and each further tear began a new phase,
in which the leak has a pattern of its own.
"""

_files_leak = {
    "Al_mesh": {
        2: "term_p2am_20150718_160913.fits",
        3: "term_p3am_20170808_180126.fits",
        4: "term_p4am_20180712_171919.fits",
        5: "term_p5am_20220709_180901.fits",
        6: "term_p6am_20230506_193649.fits",
        7: "term_p7am_20240523_181532.fits",
    },
    "Al_poly": {
        2: "term_p2ap_20150620_172818.fits",
        3: "term_p3ap_20170809_183821.fits",
        4: "term_p4ap_20180712_171928.fits",
        5: "term_p5ap_20220709_180910.fits",
        6: "term_p6ap_20230506_193658.fits",
        7: "term_p7ap_20240523_181541.fits",
    },
    "C_poly": {
        2: "term_p2cp_20150620_190645.fits",
    },
}
"""
The image of the leak through each filter in each stray-light phase,
as ``xrt_synleaksub.pro`` in SolarSoft selects them.
"""

_history_leak = "Light leak subtraction: DONE"
"""
The entry in the history of a file which ``xrt_synleaksub.pro`` writes when
it subtracts the leak.
"""

_size_leak = 1024
"""
The number of pixels along each axis of the images of the leak,
which are summed 2 by 2 on the CCD.
"""


def _phase(time: astropy.time.Time) -> int:
    """
    The stray-light phase at a given time,
    as ``check_sl_phase.pro`` in SolarSoft finds it.

    Parameters
    ----------
    time
        The time.
    """
    return int(np.sum(np.asarray(time >= _start_phase)))


@functools.cache
def _image_leak(
    file: str,
    directory: None | pathlib.Path,
) -> np.ndarray:
    """
    An image of the leak from SolarSoft, downloaded once,
    in DN per second in each pixel of 2 by 2 CCD pixels,
    indexed by row and column.

    Parameters
    ----------
    file
        The name of the image.
    directory
        The directory to place the downloaded image in.
    """
    urls = na.ScalarArray(np.array([_url_leak + file]), axes="file")
    path = np.asarray(download(urls, directory=directory).ndarray).item()
    data = astropy.io.fits.getdata(str(path), memmap=False)
    return np.asarray(data, dtype=float)


def _leak(
    filter: str,
    time: astropy.time.Time,
    num_x: int,
    num_y: int,
    chip_sum: int,
    pos_col: int,
    pos_row: int,
    axis_detector_x: str,
    axis_detector_y: str,
    directory: None | pathlib.Path = None,
) -> na.ScalarArray:
    """
    The visible light leaking into each pixel of an image, in DN per second,
    from the image of the leak SolarSoft has for its filter and its
    stray-light phase.

    The images of the leak cover the whole CCD, summed 2 by 2,
    so the part the image covers is cut out of it and resampled to the
    summing of the image:
    each pixel of an image which was not summed gets a quarter of the
    bilinear interpolation of the leak,
    as ``rebin`` in ``xrt_synleaksub.pro`` does for a full-resolution image,
    and each pixel of an image summed 4 by 4 or more gets the sum of the
    pixels of the leak it covers.

    Parameters
    ----------
    filter
        The filters the image was taken through,
        named as in :func:`hinode.xrt.urls`.
    time
        The start of the exposure.
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
    directory
        The directory to place the downloaded image of the leak in.
        If :obj:`None`, :data:`hinode.directory_default` is used.

    Raises
    ------
    ValueError
        If SolarSoft has no image of the leak for the filter and the phase.
    """
    phase = _phase(time)
    shape = {axis_detector_y: num_y, axis_detector_x: num_x}
    unit = u.DN / u.s

    # The entrance filters had not yet torn
    if phase == 0:
        return typing.cast(na.ScalarArray, na.ScalarArray.zeros(shape) << unit)

    file = _files_leak.get(filter, {}).get(phase)
    if file is None:
        raise ValueError(
            f"SolarSoft has no image of the light leak through {filter} "
            f"in stray-light phase {phase}, which began {_start_phase[phase - 1]}."
        )

    image = na.ScalarArray(
        ndarray=_image_leak(file, directory) << unit,
        axes=(axis_detector_y, axis_detector_x),
    )
    last = _size_leak - 1

    x = na.arange(0, num_x, axis=axis_detector_x)
    y = na.arange(0, num_y, axis=axis_detector_y)

    if chip_sum == 1:
        # The position of each pixel on the leak image,
        # which `rebin` interpolates between,
        # repeating the last pixel of the leak image beyond it.
        x = (x + pos_col) / 2
        y = (y + pos_row) / 2
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, last)
        y1 = np.minimum(y0 + 1, last)
        fx = x - x0
        fy = y - y0

        def value(i: na.AbstractScalar, j: na.AbstractScalar) -> na.ScalarArray:
            index = {axis_detector_x: i, axis_detector_y: j}
            return typing.cast(na.ScalarArray, image[index])

        result = (1 - fy) * ((1 - fx) * value(x0, y0) + fx * value(x1, y0))
        result = result + fy * ((1 - fx) * value(x0, y1) + fx * value(x1, y1))
        result = result / 4

    elif chip_sum % 2 == 0:
        # Each pixel of the image covers `factor` by `factor` pixels of the
        # leak image, starting from the pixel of the leak image which holds
        # the first pixel of the CCD in the image.
        factor = chip_sum // 2
        result = na.ScalarArray.zeros(shape) << unit
        for i in range(factor):
            for j in range(factor):
                index = {
                    axis_detector_x: pos_col // 2 + factor * x + i,
                    axis_detector_y: pos_row // 2 + factor * y + j,
                }
                result = result + image[index]

    else:
        raise ValueError(f"The summing of the CCD, {chip_sum}, is not supported.")

    return typing.cast(na.ScalarArray, result)
