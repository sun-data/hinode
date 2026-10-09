import typing
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = []

_center_vignetting = 1024
"""
The column and the row of the CCD, in unsummed pixels,
about which ``xrt_prep`` takes the vignetting to be symmetric.
"""

_plate_scale_vignetting = u.Quantity(1.0286, u.arcsec)
"""
The angle subtended by an unsummed pixel,
as ``xrt_prep`` takes it when correcting for vignetting.
"""

_angle_graze = u.Quantity(0.91, u.deg)
"""
The average grazing angle of the mirror,
as ``xrt_prep`` takes it when correcting for vignetting.
"""

_angle_error_vignetting = u.Quantity(9.916, u.arcmin)
"""
The field angle within which ``xrt_vign_unc.pro`` gives the vignetting
correction a constant relative error.
"""


def _angle_field(
    num_x: int,
    num_y: int,
    chip_sum: int,
    col: int,
    row: int,
    axis_detector_x: str,
    axis_detector_y: str,
) -> na.ScalarArray:
    """
    The angle of each pixel of an image from the center of the CCD,
    as ``nono_vignette.pro`` and ``xrt_vign_unc.pro`` in SolarSoft find it.

    Parameters
    ----------
    num_x
        The number of columns of the image, ``NAXIS1``.
    num_y
        The number of rows of the image, ``NAXIS2``.
    chip_sum
        The number of pixels of the CCD summed along each axis into each
        pixel of the image, ``CHIP_SUM``.
    col
        The first column of the CCD in the image, in unsummed pixels.
    row
        The first row of the CCD in the image, in unsummed pixels.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    """
    # The position of the image on the CCD is divided by the summing
    # in integer arithmetic, as the IDL does.
    x = na.arange(0, num_x, axis=axis_detector_x) + col // chip_sum
    y = na.arange(0, num_y, axis=axis_detector_y) + row // chip_sum

    center = _center_vignetting / chip_sum

    radius = np.sqrt(np.square(x - center) + np.square(y - center))

    return typing.cast(na.ScalarArray, radius * chip_sum * _plate_scale_vignetting)


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
    angle = _angle_field(
        num_x=num_x,
        num_y=num_y,
        chip_sum=chip_sum,
        col=pos_col,
        row=pos_row,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
    )

    ratio = angle.to(u.arcsec).value / _angle_graze.to_value(u.arcsec)

    return 1 - (2 / 3) * ratio


def _error_vignetting(
    num_x: int,
    num_y: int,
    chip_sum: int,
    p1_col: int,
    p1_row: int,
    axis_detector_x: str,
    axis_detector_y: str,
) -> na.ScalarArray:
    """
    The relative error of the vignetting correction of each pixel of an image,
    which is not a dark frame,
    as ``xrt_vign_unc.pro`` in SolarSoft gives it.

    Parameters
    ----------
    num_x
        The number of columns of the image, ``NAXIS1``.
    num_y
        The number of rows of the image, ``NAXIS2``.
    chip_sum
        The number of pixels of the CCD summed along each axis into each
        pixel of the image, ``CHIP_SUM``.
    p1_col
        The first column of the CCD in the image, in unsummed pixels,
        ``P1COL``, which ``xrt_vign_unc.pro`` uses in place of ``POS_COL``.
    p1_row
        The first row of the CCD in the image, in unsummed pixels,
        ``P1ROW``, which ``xrt_vign_unc.pro`` uses in place of ``POS_ROW``.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    """
    angle = _angle_field(
        num_x=num_x,
        num_y=num_y,
        chip_sum=chip_sum,
        col=p1_col,
        row=p1_row,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
    )
    angle = angle.to(u.arcmin).value
    limit = _angle_error_vignetting.to_value(u.arcmin)

    # Both cases apply at the limit itself, as in ``xrt_vign_unc.pro``
    outer = (21.4681 - 6.11904 * angle + 0.444524 * np.square(angle)) / 1000 - 0.0045
    result = 0.0045 * (angle <= limit) + (angle >= limit) * outer

    return typing.cast(na.ScalarArray, result)
