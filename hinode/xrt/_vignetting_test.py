import pytest
import numpy as np
import astropy.units as u
import astropy.io.fits
from hinode.xrt._vignetting import _angle_field, _vignetting, _error_vignetting


def _vignetting_numpy(header: astropy.io.fits.Header) -> np.ndarray:
    """
    The vignetting function of ``xrt_prep``, indexed by row and column,
    following ``nono_vignette.pro`` line by line.
    """
    num_x = header["NAXIS1"]
    num_y = header["NAXIS2"]
    chip_sum = header["CHIP_SUM"]

    x = np.arange(num_x, dtype=float)[np.newaxis, :] + header["POS_COL"] // chip_sum
    y = np.arange(num_y, dtype=float)[:, np.newaxis] + header["POS_ROW"] // chip_sum

    x0 = 1024 / chip_sum
    y0 = 1024 / chip_sum
    arcsec_per_pix = 1.0286 * chip_sum
    graze_angle = 0.91 * 60

    angle = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) * arcsec_per_pix / 60

    return 1 - (2 / 3) * (angle / graze_angle)


@pytest.mark.parametrize(
    argnames="chip_sum,pos_col,pos_row",
    argvalues=[
        (1, 872, 856),
        (2, 3, 1),
        (4, 0, 2047),
    ],
)
def test_vignetting(
    chip_sum: int,
    pos_col: int,
    pos_row: int,
) -> None:
    header = astropy.io.fits.Header()
    header["NAXIS1"] = 7
    header["NAXIS2"] = 5
    header["CHIP_SUM"] = chip_sum
    header["POS_COL"] = pos_col
    header["POS_ROW"] = pos_row

    result = _vignetting(
        num_x=7,
        num_y=5,
        chip_sum=chip_sum,
        pos_col=pos_col,
        pos_row=pos_row,
        axis_detector_x="x",
        axis_detector_y="y",
    )

    expected = _vignetting_numpy(header)

    assert result.shape == {"x": 7, "y": 5}
    assert np.allclose(result.ndarray_aligned(("y", "x")), expected, rtol=1e-12)


def test_vignetting_center() -> None:
    """The light is not vignetted at the center of the CCD."""
    result = _vignetting(
        num_x=3,
        num_y=3,
        chip_sum=1,
        pos_col=1023,
        pos_row=1023,
        axis_detector_x="x",
        axis_detector_y="y",
    )

    assert result[dict(x=1, y=1)] == 1
    assert result[dict(x=0, y=1)] < 1


def test_angle_field() -> None:
    """
    The angle from the center of the CCD,
    for an image summed 4 by 4 whose first column is not a multiple of 4,
    which ``xrt_prep`` divides by the summing in integer arithmetic.
    """
    result = _angle_field(
        num_x=2,
        num_y=1,
        chip_sum=4,
        col=1026,
        row=1024,
        axis_detector_x="x",
        axis_detector_y="y",
    )
    expected = [0, 4 * 1.0286] * u.arcsec
    assert np.allclose(result.ndarray_aligned(("y", "x"))[0], expected)


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
