import pytest
import numpy as np
import astropy.units as u
import astropy.constants
import astropy.time
import named_arrays as na
import xrtpy.response
import hinode
from hinode.xrt._response import _name_xrtpy

_time = astropy.time.Time("2019-09-30T18:08:00")


@pytest.mark.parametrize(
    argnames="filter,expected",
    argvalues=[
        ("Al_poly", "Al-poly"),
        ("Gband", "G-band"),
        ("Al_poly/Ti_poly", "Al-poly/Ti-poly"),
        ("Be_thick", "Be-thick"),
    ],
)
def test_name_xrtpy(filter: str, expected: str) -> None:
    assert _name_xrtpy(filter) == expected


@pytest.mark.parametrize(
    argnames="filter,axis_filter,num_filter",
    argvalues=[
        ("Al_poly", "filter", 1),
        (["Al_poly", "Ti_poly", "Al_poly/Ti_poly"], "filter", 3),
        (na.ScalarArray(np.array("Al_poly")), "filter", 1),
        (
            na.ScalarArray(np.array(["Al_poly", "Be_thin"]), axes="channel"),
            "channel",
            2,
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="photons",
    argvalues=[False, True],
)
def test_temperature_response(
    filter: str | list[str] | na.ScalarArray,
    axis_filter: str,
    num_filter: int,
    photons: bool,
) -> None:
    result = hinode.xrt.temperature_response(filter, _time, photons=photons)

    assert isinstance(result, na.FunctionArray)
    assert result.outputs.shape == {axis_filter: num_filter, "temperature": 61}
    assert result.inputs.shape == {"temperature": 61}

    unit = u.ph if photons else u.DN
    assert result.outputs.unit.is_equivalent(unit * u.cm**5 / u.s / u.pix)

    log_temperature = np.log10(result.inputs.ndarray.to_value(u.K))
    assert np.allclose(log_temperature[[0, -1]], [5, 8], atol=1e-6)
    assert np.all(result.outputs.ndarray > 0)


def test_temperature_response_xrtpy() -> None:
    """The response is the one :mod:`xrtpy` gives."""
    result = hinode.xrt.temperature_response("Al_poly", _time)

    channel = xrtpy.response.TemperatureResponseFundamental("Al-poly", _time)
    expected = channel.temperature_response()

    assert np.array_equal(result.outputs[dict(filter=0)].ndarray, expected)
    assert np.array_equal(result.inputs.ndarray, channel.CHIANTI_temperature)


def test_temperature_response_photons() -> None:
    """
    The signal of one photon from plasma at any temperature lies between
    that of the longest and of the shortest wavelengths the filter passes.
    """
    response = hinode.xrt.temperature_response("Al_poly", _time)
    photons = hinode.xrt.temperature_response("Al_poly", _time, photons=True)

    ratio = (response.outputs / photons.outputs).to(u.DN / u.ph)

    channel = xrtpy.response.TemperatureResponseFundamental("Al-poly", _time)
    energy = astropy.constants.h * astropy.constants.c / channel.wavelength
    signal = energy / channel.ev_per_electron / channel.ccd_gain_right
    signal = signal.to(u.DN) / u.ph

    assert np.all(ratio.ndarray >= signal.min())
    assert np.all(ratio.ndarray <= signal.max())

    # Hotter plasma emits shorter wavelengths, from log T 6 to 7
    hot = ratio[dict(filter=0, temperature=slice(20, 41))]
    assert np.all(np.diff(hot, axis="temperature") > 0)


def test_temperature_response_abundance() -> None:
    """Photospheric abundances have less of the elements of the X-ray lines."""
    coronal = hinode.xrt.temperature_response("Al_poly", _time)
    photospheric = hinode.xrt.temperature_response(
        filter="Al_poly",
        time=_time,
        abundance="photospheric",
    )
    assert np.sum(photospheric.outputs.ndarray) < np.sum(coronal.outputs.ndarray)


@pytest.mark.parametrize(
    argnames="filter,time",
    argvalues=[
        (
            na.ScalarArray(np.array([["Al_poly"]]), axes=("a", "b")),
            _time,
        ),
        (
            "Al_poly",
            astropy.time.Time(["2019-09-30", "2019-10-01"]),
        ),
    ],
)
def test_temperature_response_shape(
    filter: str | na.ScalarArray,
    time: astropy.time.Time,
) -> None:
    with pytest.raises(ValueError, match="must be"):
        hinode.xrt.temperature_response(filter, time)
