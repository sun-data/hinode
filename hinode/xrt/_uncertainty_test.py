import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from hinode.xrt._uncertainty import (
    _quality,
    _error_jpeg,
    _error_dark,
    _error_vignetting,
)

_history_strip = (
    "XRT_CLEAN_RO: clean_type = 0 Full Fourier preclean/clean [default] "
    "512x2048 strip darks used -- zero-point less well determined."
)
"""Part of the history of the Level 1 files of 2019 September 30."""


@pytest.mark.parametrize(
    argnames="compression,table,expected",
    argvalues=[
        (0, 1, None),
        (3, 1, None),
        (7, 1, 90),
        (7, 0, 98),
        (7, 7, 65),
    ],
)
def test_quality(compression: int, table: int, expected: None | int) -> None:
    assert _quality(compression, table) == expected


def test_quality_unknown() -> None:
    with pytest.raises(ValueError, match="not known"):
        _quality(5, 0)


def test_error_jpeg() -> None:
    """
    The error of each pixel is that of the range of its block of 8 by 8
    pixels, and the pixels beyond the last whole block have no error.
    """
    signal = np.zeros((12, 24))
    signal[:8, 8:16] = np.linspace(0, 50, 64).reshape(8, 8)
    signal[:8, 16:24] = np.linspace(0, 300, 64).reshape(8, 8)
    signal[0, 16] = np.nan

    result = _error_jpeg(
        signal=na.ScalarArray(signal << u.DN, axes=("y", "x")),
        quality=90,
        axis_detector_x="x",
        axis_detector_y="y",
    )
    result = result.ndarray_aligned(("y", "x")).to_value(u.DN)

    # A flat block, whose polynomial is negative there
    assert np.all(result[:8, :8] == 0)

    # A block with a range of 50 DN
    assert np.allclose(result[:8, 8:16], 2.871390515625)

    # A block with a range beyond the cutoff, despite a pixel with no value
    assert np.allclose(result[:8, 16:24], 3.07025)

    # The rows beyond the last whole block
    assert np.all(result[8:] == 0)


def test_error_jpeg_quality_unknown() -> None:
    signal = na.ScalarArray(np.zeros((8, 8)) << u.DN, axes=("y", "x"))
    with pytest.raises(ValueError, match="not known"):
        _error_jpeg(signal, 77, "x", "y")


def test_error_jpeg_lossless() -> None:
    signal = na.ScalarArray(np.arange(64.0).reshape(8, 8) << u.DN, axes=("y", "x"))
    result = _error_jpeg(signal, None, "x", "y")
    assert np.all(result == 0 * u.DN)


@pytest.mark.parametrize(
    argnames="history,chip_sum,num_x,num_y,quality,expected",
    argvalues=[
        (_history_strip, 1, 384, 384, 90, 0.88314570084691),
        ("clean_type = 0", 2, 2048, 2048, 98, 1.3886769983063016),
        (_history_strip.replace("= 0", "= 3"), 1, 384, 384, 90, 1.1463365426683774),
        ("", 1, 2048, 2048, None, 0.8600006075119947),
    ],
)
def test_error_dark(
    history: str,
    chip_sum: int,
    num_x: int,
    num_y: int,
    quality: None | int,
    expected: float,
) -> None:
    result = _error_dark(history, chip_sum, num_x, num_y, quality)
    assert np.isclose(result, expected * u.DN, rtol=1e-12)


def test_error_dark_chip_sum() -> None:
    with pytest.raises(ValueError, match="not supported"):
        _error_dark("", 3, 2048, 2048, 90)


def test_error_vignetting() -> None:
    """
    The relative error of the vignetting correction is 0.45% within about
    10 arcminutes of the center of the CCD, and grows beyond.
    """
    result = _error_vignetting(
        num_x=2048,
        num_y=2048,
        chip_sum=1,
        p1_col=0,
        p1_row=0,
        axis_detector_x="x",
        axis_detector_y="y",
    )
    assert result[dict(x=1024, y=1024)] == 0.0045
    assert np.isclose(result[dict(x=0, y=0)].ndarray, 0.1390335796672054)
