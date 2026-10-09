import re
import typing
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = []

_quality_table = (98, 90, 75, 50, 95, 92, 85, 65)
"""
The quality of the JPEG compression of each quantization table, ``QTABLE1``,
as ``get_xrt_complevel.pro`` in SolarSoft gives them.
"""

_jpeg = {
    50: ([12.0], 0, 12.0),
    65: (
        [-0.481573, 0.271544, -0.00648255, 0.000101617, -9.49282e-7]
        + [5.33988e-9, -1.82363e-11, 3.69312e-14, -4.07136e-17, 1.87929e-20],
        378,
        9.48318,
    ),
    75: (
        [-0.125886, 0.185941, -0.00294657, 3.14739e-5, -2.20472e-7]
        + [9.98487e-10, -2.87709e-12, 5.07344e-15, -4.97955e-18, 2.08052e-21],
        475,
        7.4321,
    ),
    85: (
        [-0.0403521, 0.170855, -0.00334726, 4.18432e-5, -3.45394e-7]
        + [1.87510e-9, -6.56483e-12, 1.41939e-14, -1.71855e-17, 8.89557e-21],
        375,
        4.55106,
    ),
    90: (
        [-0.298208, 0.195721, -0.00628192, 0.000148311, -2.56535e-6]
        + [3.01413e-8, -2.27592e-10, 1.04751e-12, -2.66459e-15, 2.86624e-18],
        190,
        3.07025,
    ),
    92: (
        [-0.295559, 0.185933, -0.00583421, 0.000113539, -1.52283e-6]
        + [1.43983e-8, -9.33111e-11, 3.89080e-13, -9.30292e-16, 9.62735e-19],
        200,
        2.45152,
    ),
    95: (
        [-0.341410, 0.204188, -0.00903790, 0.000157361, 2.20624e-6]
        + [-1.67321e-7, 3.60387e-9, -3.96033e-11, 2.24208e-13, -5.18546e-16],
        100,
        1.56126,
    ),
    98: ([0.0], 1, 0.7),
}
"""
The error of the JPEG compression of a block of 8 by 8 pixels, in DN,
as a function of the range of the values in the block,
for each quality of the compression,
from ``get_jpeg_unc.pro`` in SolarSoft :cite:p:`Kobelski2014`.

Each quality has the coefficients of a polynomial in the range,
the range from which the error is a constant instead,
and that constant.
"""

_size_block = 8
"""The number of pixels along each axis of a block of the JPEG compression."""

_dark = {
    False: {
        1: ({90: 0.9602, 98: 1.3239}, 1.1030, 0.177),
        2: ({90: 0.9800, 98: 1.4538}, 1.1624, 0.557),
        4: ({98: 1.6640}, 1.1859, 1.000),
        8: ({50: 0.543, 98: 1.6903}, 1.3348, 1.566),
    },
    True: {
        1: ({90: 1.1386, 98: 1.5700}, 1.308, 0.189),
        2: ({90: 1.1978, 98: 1.6516}, 1.376, 0.530),
        4: ({98: 2.503}, 1.784, 1.088),
        8: ({50: 0.535, 98: 1.666}, 1.316, 2.522),
    },
}
"""
The parameters of the error of the dark subtraction of ``xrt_prep``,
with its default model of the dark, from ``under_table.pro`` in SolarSoft
:cite:p:`Kobelski2014`.

The parameters are indexed by whether strip darks were used and by the
summing of the CCD.
Each has the average error for some qualities of the JPEG compression,
the average error for the other qualities,
and the scatter of the error.
"""

_filter_fourier = {1: 0.763, 2: 0.875, 4: 0.895}
"""
The factor by which the Fourier filtering of ``xrt_prep`` reduces the error
of the dark subtraction for each summing of the CCD,
from ``under_table.pro`` in SolarSoft, and 0.914 for the other summings.
"""


def _quality(
    compression: int,
    table: int,
) -> None | int:
    """
    The quality of the JPEG compression of an image,
    or :obj:`None` if it was not compressed with a loss,
    as ``get_xrt_complevel.pro`` in SolarSoft finds it.

    Parameters
    ----------
    compression
        The kind of compression, ``IMGCOMP1``:
        0 for none, 3 for lossless, and 7 for JPEG.
    table
        The quantization table of the JPEG compression, ``QTABLE1``.
    """
    if compression in (0, 3):
        return None
    if compression != 7:
        raise ValueError(f"The compression of the image, {compression}, is not known.")
    if not 0 <= table < len(_quality_table):
        raise ValueError(f"The quantization table of the image, {table}, is not known.")
    return _quality_table[table]


def _error_jpeg(
    signal: na.ScalarArray,
    quality: None | int,
    axis_detector_x: str,
    axis_detector_y: str,
) -> na.ScalarArray:
    """
    The error of the JPEG compression of each pixel of an image, in DN,
    as ``jpeg_unc.pro`` in SolarSoft estimates it from the range of the
    signal in each block of 8 by 8 pixels.

    The pixels beyond the last whole block along either axis are given no
    error, as ``jpeg_unc.pro`` gives them.

    Parameters
    ----------
    signal
        The signal of each pixel, in DN, before the dark was subtracted and
        the image was corrected for vignetting.
    quality
        The quality of the compression,
        or :obj:`None` if the image was not compressed with a loss.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    """
    shape = signal.shape
    result = na.ScalarArray.zeros(shape) << u.DN

    if quality is None:
        return result

    if quality not in _jpeg:
        raise ValueError(
            f"The quality of the JPEG compression, {quality}, is not known."
        )

    coefficients, cutoff, constant = _jpeg[quality]

    num_x = shape[axis_detector_x] // _size_block
    num_y = shape[axis_detector_y] // _size_block

    # The pixels of each block along axes of their own
    pixel_x = na.arange(0, _size_block, axis="_pixel_x")
    pixel_y = na.arange(0, _size_block, axis="_pixel_y")
    block_x = na.arange(0, num_x, axis=axis_detector_x)
    block_y = na.arange(0, num_y, axis=axis_detector_y)
    blocks = signal[
        {
            axis_detector_x: _size_block * block_x + pixel_x,
            axis_detector_y: _size_block * block_y + pixel_y,
        }
    ]

    # The range of the pixels of each block which have a value
    axis_pixel = ("_pixel_x", "_pixel_y")
    blocks = typing.cast(na.ScalarArray, blocks / u.DN).value
    finite = np.isfinite(blocks)
    maximum = blocks.max(axis=axis_pixel, initial=-np.inf, where=finite)
    minimum = blocks.min(axis=axis_pixel, initial=np.inf, where=finite)
    span = typing.cast(na.ScalarArray, maximum - minimum)

    # A block with no values is given any error, since its pixels have none
    span[~np.isfinite(span)] = 0

    polynomial = na.ScalarArray.zeros(span.shape)
    for c in reversed(coefficients):
        polynomial = polynomial * span + c
    error = (span < cutoff) * polynomial + (span >= cutoff) * constant
    error = (error > 0) * error

    # Each pixel is given the error of its block
    index_block = {
        axis_detector_x: na.arange(0, _size_block * num_x, axis=axis_detector_x)
        // _size_block,
        axis_detector_y: na.arange(0, _size_block * num_y, axis=axis_detector_y)
        // _size_block,
    }
    index = {
        axis_detector_x: slice(None, _size_block * num_x),
        axis_detector_y: slice(None, _size_block * num_y),
    }
    result[index] = typing.cast(na.ScalarArray, error[index_block]) << u.DN

    return result


def _error_dark(
    history: str,
    chip_sum: int,
    num_x: int,
    num_y: int,
    quality: None | int,
) -> u.Quantity:
    """
    The error of the dark subtraction of an image, in DN,
    as ``under_table.pro`` in SolarSoft gives it for the default model of the
    dark of ``xrt_prep``.

    Whether strip darks were used, and which Fourier filtering was applied,
    are read from the history of the file.

    Parameters
    ----------
    history
        The history of the file, as :func:`hinode.xrt._data._history` joins it.
    chip_sum
        The number of pixels of the CCD summed along each axis into each
        pixel of the image, ``CHIP_SUM``.
    num_x
        The number of columns of the image, ``NAXIS1``.
    num_y
        The number of rows of the image, ``NAXIS2``.
    quality
        The quality of the JPEG compression of the image,
        or :obj:`None` if it was not compressed with a loss.
    """
    strip = "strip darks used" in history
    match = re.search(r"clean_type = (\d+)", history)
    type_clean = 0 if match is None else int(match.group(1))

    if chip_sum not in _dark[strip]:
        raise ValueError(f"The summing of the CCD, {chip_sum}, is not supported.")
    average, average_other, scatter = _dark[strip][chip_sum]
    average = average.get(-1 if quality is None else quality, average_other)

    factor_fourier = 1.0
    if type_clean != 3:
        factor_fourier = _filter_fourier.get(chip_sum, 0.914)

    # The error is a little different for images smaller than the CCD
    factor_size = 1.0
    if num_x != 2048 or num_y != 2048:
        factor_x = 1.003076 - 0.00098327976 * np.log10(num_x)
        factor_y = 0.96479402 + 0.010715234 * np.log10(num_y)
        factor_size = factor_x * factor_y

    error = np.sqrt(scatter**2 + (average * factor_fourier * factor_size) ** 2)
    return error * u.DN
