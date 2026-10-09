import pytest
import numpy as np
import astropy.units as u
import astropy.constants
import astropy.time
import named_arrays as na
import xrtpy.response
from xrtpy.response.effective_area import parse_filter_input
import hinode
from hinode.xrt._response import _name_xrtpy

_time = astropy.time.Time("2019-09-30T18:08:00")


@pytest.mark.parametrize(
    argnames="filter",
    argvalues=[
        "Al_poly",
        "C_poly",
        "Be_thin",
        "Be_med",
        "Al_med",
        "Al_mesh",
        "Ti_poly",
        "Gband",
        "Al_thick",
        "Be_thick",
        "Al_poly/Ti_poly",
        "C_poly/Ti_poly",
    ],
)
def test_name_xrtpy(filter: str) -> None:
    """Every filter, named as in the headers, is one :mod:`xrtpy` knows."""
    parsed = parse_filter_input(_name_xrtpy(filter))
    names = [parsed.filter1, parsed.filter2] if parsed.is_combo else [parsed.filter1]
    assert [n.replace("-", "").lower() for n in names] == [
        n.replace("_", "").lower() for n in filter.split("/")
    ]


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
    argnames="variance",
    argvalues=[False, True],
)
def test_temperature_response(
    filter: str | list[str] | na.ScalarArray,
    axis_filter: str,
    num_filter: int,
    variance: bool,
) -> None:
    result = hinode.xrt.temperature_response(filter, _time, variance=variance)

    assert isinstance(result, na.FunctionArray)
    assert result.outputs.shape == {axis_filter: num_filter, "temperature": 61}
    assert result.inputs.shape == {"temperature": 61}

    unit = u.DN * u.cm**5 / u.s / u.pix
    if variance:
        unit = unit * u.DN / u.ph
    assert result.outputs.unit == unit

    log_temperature = np.log10(result.inputs.ndarray.to_value(u.K))
    assert np.allclose(log_temperature[[0, -1]], [5, 8], atol=1e-6)
    assert np.all(result.outputs.ndarray > 0)


def test_temperature_response_xrtpy() -> None:
    """The response is the one :mod:`xrtpy` gives."""
    result = hinode.xrt.temperature_response("Al_poly", _time)

    channel = xrtpy.response.TemperatureResponseFundamental("Al-poly", _time)
    expected = channel.temperature_response()

    assert np.allclose(result.outputs[dict(filter=0)].ndarray, expected, rtol=1e-12)
    assert np.array_equal(result.inputs.ndarray, channel.CHIANTI_temperature)


def _signal_numpy(filter: str) -> tuple[np.ndarray, np.ndarray]:
    """
    The rate at which the CCD absorbs photons of each wavelength from plasma
    at each temperature, indexed by temperature and wavelength,
    and the signal of one photon of each wavelength,
    from the parts :mod:`xrtpy` computes.
    """
    channel = xrtpy.response.TemperatureResponseFundamental(filter, _time)
    wavelength = channel.wavelength
    rate = (
        channel.spectra()
        * channel.effective_area()
        * channel.solid_angle_per_pixel
        * np.gradient(wavelength)
    )
    energy = astropy.constants.h * astropy.constants.c / wavelength / u.ph
    signal = energy / channel.ev_per_electron / channel.ccd_gain_right
    return rate, signal


@pytest.mark.parametrize(
    argnames="filter",
    argvalues=["Al_poly", "Ti_poly"],
)
def test_temperature_response_variance(filter: str) -> None:
    """
    The variance of the signal grows with the sum over wavelength of the
    rate of the photons times the square of their signal,
    so the factor which sets the photon noise is larger than the average
    signal of one photon.
    """
    response = hinode.xrt.temperature_response(filter, _time)
    variance = hinode.xrt.temperature_response(filter, _time, variance=True)

    rate, signal = _signal_numpy(_name_xrtpy(filter))
    expected = (rate * signal**2).sum(axis=1)
    result = variance.outputs[dict(filter=0)].ndarray
    assert np.allclose(result, expected.to(result.unit), rtol=1e-12)

    factor = (variance.outputs / response.outputs)[dict(filter=0)].ndarray
    average = (rate * signal).sum(axis=1) / rate.sum(axis=1)
    assert np.all(factor.to(u.DN / u.ph) >= average.to(u.DN / u.ph))
    assert np.all(factor <= signal.max())


def test_temperature_response_variance_value() -> None:
    """
    The factor which sets the photon noise of plasma at log T 6.0 through
    Al_poly, about 1.7 times the average signal of one photon, 0.94 DN.
    """
    response = hinode.xrt.temperature_response("Al_poly", _time)
    variance = hinode.xrt.temperature_response("Al_poly", _time, variance=True)
    index = dict(filter=0, temperature=20)
    assert np.isclose(response.inputs[index].ndarray, 10**6 * u.K)
    factor = (variance.outputs / response.outputs)[index].ndarray
    assert np.isclose(factor, 1.6094 * u.DN / u.ph, rtol=1e-4)


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
