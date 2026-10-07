import astropy.units as u
import hinode


def test_open():
    result = hinode.xrt.open(
        time_start="2019-09-30T18:08:30",
        time_stop="2019-09-30T18:09:00",
        filter="Al_poly",
    )
    assert isinstance(result, hinode.xrt.Filtergram)
    assert result.shape[result.axis_time] == 2
    assert result.filter == "Al_poly"
    assert result.outputs.unit == u.DN / u.s
