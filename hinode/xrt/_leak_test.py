import pathlib
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import hinode.xrt._leak
from hinode.xrt._leak import _leak, _phase, _size_leak


def _rebin_numpy(image: np.ndarray, factor: int) -> np.ndarray:
    """
    An image magnified by an integer factor along both axes,
    as ``rebin`` in IDL magnifies it,
    interpolating linearly between the pixels
    and repeating the last pixel beyond it.
    """
    for axis in range(image.ndim):
        num = image.shape[axis]
        position = np.arange(num * factor) / factor
        i0 = np.floor(position).astype(int)
        i1 = np.minimum(i0 + 1, num - 1)
        fraction = position - i0
        shape = [1] * image.ndim
        shape[axis] = -1
        fraction = fraction.reshape(shape)
        image = (1 - fraction) * np.take(image, i0, axis) + fraction * np.take(
            image, i1, axis
        )
    return image


def _image_synthetic(file: str, directory: None | pathlib.Path) -> np.ndarray:
    """A random image of the leak, made without connecting to SolarSoft."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 2, size=(_size_leak, _size_leak))


@pytest.mark.parametrize(
    argnames="time,expected",
    argvalues=[
        ("2007-01-01", 0),
        ("2012-05-09T11:59", 0),
        ("2012-05-09T12:00", 1),
        ("2019-09-30T18:08:37", 4),
        ("2023-05-05T04:29", 5),
        ("2023-05-05T04:30", 6),
        ("2025-01-01", 7),
    ],
)
def test_phase(time: str, expected: int) -> None:
    assert _phase(astropy.time.Time(time, scale="utc")) == expected


def test_leak_before_tears() -> None:
    """There was no leak before the entrance filters began to tear."""
    result = _leak(
        filter="Al_poly",
        time=astropy.time.Time("2010-01-01"),
        num_x=7,
        num_y=5,
        chip_sum=1,
        pos_col=0,
        pos_row=0,
        axis_detector_x="x",
        axis_detector_y="y",
    )
    assert result.shape == {"y": 5, "x": 7}
    assert np.all(result == 0 * u.DN / u.s)


@pytest.mark.parametrize(
    argnames="filter,time",
    argvalues=[
        ("Ti_poly", "2019-09-30"),
        ("Al_poly", "2013-01-01"),
        ("Gband", "2019-09-30"),
    ],
)
def test_leak_missing(filter: str, time: str) -> None:
    with pytest.raises(ValueError, match="no image of the light leak"):
        _leak(
            filter=filter,
            time=astropy.time.Time(time),
            num_x=7,
            num_y=5,
            chip_sum=1,
            pos_col=0,
            pos_row=0,
            axis_detector_x="x",
            axis_detector_y="y",
        )


@pytest.mark.parametrize(
    argnames="chip_sum,num_x,num_y,pos_col,pos_row",
    argvalues=[
        (1, 2048, 2048, 0, 0),
        (1, 7, 5, 872, 856),
        (1, 6, 4, 2041, 2043),
        (2, 1024, 1024, 0, 0),
        (2, 7, 5, 872, 857),
        (4, 512, 512, 0, 0),
        (4, 7, 5, 872, 856),
        (8, 3, 2, 1016, 8),
    ],
)
def test_leak_resample(
    monkeypatch: pytest.MonkeyPatch,
    chip_sum: int,
    num_x: int,
    num_y: int,
    pos_col: int,
    pos_row: int,
) -> None:
    """
    The leak image is cut out and resampled to the summing of the image,
    as ``xrt_synleaksub.pro`` does for a full-resolution or 2 by 2 image.
    """
    monkeypatch.setattr(hinode.xrt._leak, "_image_leak", _image_synthetic)
    image = _image_synthetic("", None)

    result = _leak(
        filter="Al_poly",
        time=astropy.time.Time("2019-09-30"),
        num_x=num_x,
        num_y=num_y,
        chip_sum=chip_sum,
        pos_col=pos_col,
        pos_row=pos_row,
        axis_detector_x="x",
        axis_detector_y="y",
    )

    if chip_sum == 1:
        expected = _rebin_numpy(image, 2) / 4
        expected = expected[pos_row : pos_row + num_y, pos_col : pos_col + num_x]
    else:
        factor = chip_sum // 2
        binned = image.reshape(
            _size_leak // factor, factor, _size_leak // factor, factor
        ).sum(axis=(1, 3))
        x = pos_col // chip_sum
        y = pos_row // chip_sum
        expected = binned[y : y + num_y, x : x + num_x]

    assert result.shape == {"y": num_y, "x": num_x}
    assert result.unit == u.DN / u.s
    assert np.allclose(result.ndarray_aligned(("y", "x")).value, expected)


def test_leak_chip_sum_odd() -> None:
    with pytest.raises(ValueError, match="not supported"):
        _leak(
            filter="Al_poly",
            time=astropy.time.Time("2019-09-30"),
            num_x=7,
            num_y=5,
            chip_sum=3,
            pos_col=0,
            pos_row=0,
            axis_detector_x="x",
            axis_detector_y="y",
        )


def test_leak_event_e() -> None:
    """
    The leak in the field of view of XRT while ESIS was observing,
    from the leak image of SolarSoft for phase 4.
    """
    result = _leak(
        filter="Al_poly",
        time=astropy.time.Time("2019-09-30T18:08:37"),
        num_x=384,
        num_y=384,
        chip_sum=1,
        pos_col=872,
        pos_row=856,
        axis_detector_x="x",
        axis_detector_y="y",
    )

    value = result.ndarray.to_value(u.DN / u.s)
    assert 0.7 < value.min() < value.max() < 1.1
    assert np.isclose(np.median(value), 0.92, atol=0.01)
    assert isinstance(result, na.ScalarArray)
