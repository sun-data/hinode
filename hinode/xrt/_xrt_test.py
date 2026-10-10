import astropy.units as u
import hinode


def test_open() -> None:
    result = hinode.xrt.open(
        time_start="2019-09-30T18:08:30",
        time_stop="2019-09-30T18:09:00",
        filter="Al_poly",
    )
    assert isinstance(result, hinode.xrt.Filtergram)
    assert result.shape[result.axis_time] == 2
    assert result.filter == "Al_poly"
    assert result.outputs.unit == u.DN / u.s
    assert result.leak is None
    assert result.coalignment is not None
    assert result.coalignment.ndarray.tolist() == [1, 1]


def test_open_without_coalign() -> None:
    result = hinode.xrt.open(
        time_start="2019-09-30T18:08:30",
        time_stop="2019-09-30T18:09:00",
        coalign=None,
    )
    assert result.coalignment is None


def test_open_leak() -> None:
    result = hinode.xrt.open(
        time_start="2019-09-30T18:08:30",
        time_stop="2019-09-30T18:09:00",
        leak=True,
        uncertainty=True,
    )
    assert result.leak is not None
    assert result.leak.shape == result.outputs.shape
    assert result.uncertainty_map is not None
