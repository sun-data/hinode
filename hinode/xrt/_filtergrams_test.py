import re
import pathlib
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import astropy.wcs
import astropy.io.fits
import named_arrays as na
import hinode

_time_start = astropy.time.Time("2019-09-30T18:08:00")
_time_stop = astropy.time.Time("2019-09-30T18:09:00")


def _path(time_start: str, time_stop: str) -> pathlib.Path:
    """The one Al_poly file which began during the given time range."""
    urls = hinode.xrt.urls(time_start, time_stop)
    (path,) = hinode.xrt.download(urls).ndarray
    return pathlib.Path(path)


def _write(
    file: pathlib.Path,
    num_x: None | int = None,
    num_y: None | int = None,
    missing: None | tuple[int, int] = None,
    filter_2: None | str = None,
) -> pathlib.Path:
    """
    Write a copy of the 18:08:37 file, the one in which event E saturates,
    changed as given.

    Parameters
    ----------
    file
        Where to write the copy.
    num_x
        If not :obj:`None`, the number of columns to keep.
    num_y
        If not :obj:`None`, the number of rows to keep.
    missing
        If not :obj:`None`, the column and row of a pixel to mark as missing.
    filter_2
        If not :obj:`None`, the filter the copy says is in the second wheel.
    """
    path = _path("2019-09-30T18:08:30", "2019-09-30T18:08:40")
    data = astropy.io.fits.getdata(path)
    header = astropy.io.fits.getheader(path)

    if missing is not None:
        data[missing[1], missing[0]] = -999

    data = data[:num_y, :num_x]

    if filter_2 is not None:
        header["EC_FW2_"] = filter_2

    astropy.io.fits.writeto(file, data, header)

    return file


def _num_saturated(header: astropy.io.fits.Header) -> int:
    """The number of saturated pixels ``xrt_prep`` says it found."""
    history = " ".join(str(card) for card in header["HISTORY"])
    match = re.search(r"Replaced (\d+) saturated pixels", history)
    assert match is not None
    return int(match.group(1))


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        hinode.xrt.Filtergram.from_time_range(
            time_start=_time_start,
            time_stop=_time_stop,
        ),
        hinode.xrt.Filtergram.from_time_range(
            time_start=_time_start,
            time_stop=_time_stop,
            filter="Al_poly",
            axis_time="t",
            axis_detector_x="x",
            axis_detector_y="y",
        ),
    ],
)
class TestFiltergram:

    def test_axes(self, array: hinode.xrt.Filtergram):
        axes = {array.axis_time, array.axis_detector_x, array.axis_detector_y}
        assert set(array.outputs.shape) == axes
        assert set(array.inputs.crpix.components) == axes - {array.axis_time}
        assert set(array.inputs.pc.position.x.components) == axes - {array.axis_time}
        assert set(array.inputs.pc.position.y.components) == axes - {array.axis_time}

    def test_shape(self, array: hinode.xrt.Filtergram):
        assert array.shape[array.axis_time] == 4
        assert array.shape[array.axis_detector_x] == 384
        assert array.shape[array.axis_detector_y] == 384

    def test_time(self, array: hinode.xrt.Filtergram):
        time = array.inputs.time
        assert time.shape == {array.axis_time: array.shape[array.axis_time]}
        assert np.all(time.ndarray >= _time_start)
        assert np.all(time.ndarray < _time_stop)
        assert np.all(np.diff(time.ndarray.jd) > 0)

    def test_timedelta(self, array: hinode.xrt.Filtergram):
        timedelta = array.timedelta
        assert timedelta.shape == {array.axis_time: array.shape[array.axis_time]}
        assert np.all(timedelta > 10 * u.s)

    def test_outputs(self, array: hinode.xrt.Filtergram):
        outputs = array.outputs
        assert outputs.unit == u.DN / u.s
        assert not np.any(outputs == -999 * u.DN / u.s)
        assert np.nanmedian(outputs) > 0 * u.DN / u.s

    def test_saturated(self, array: hinode.xrt.Filtergram):
        saturated = array.saturated
        assert isinstance(saturated, na.ScalarArray)
        assert saturated.ndarray.dtype == bool
        assert saturated.shape == array.outputs.shape

        # Event E saturates the frames at 18:08:19 and 18:08:37
        num = saturated.sum(axis=(array.axis_detector_x, array.axis_detector_y))
        assert np.all(num.ndarray == [0, 6, 3, 0])

    def test_filter(self, array: hinode.xrt.Filtergram):
        assert array.filter == "Al_poly"

    def test_getitem(self, array: hinode.xrt.Filtergram):
        index = {array.axis_time: 1}
        result = array[index]
        assert isinstance(result, hinode.xrt.Filtergram)
        assert result.timedelta.shape == {}
        assert result.timedelta == array.timedelta[index]
        assert result.saturated is not None
        assert array.saturated is not None
        assert np.all(result.saturated == array.saturated[index])


@pytest.mark.parametrize(
    argnames="time_start,time_stop",
    argvalues=[
        ("2019-09-30T18:08:10", "2019-09-30T18:08:20"),
        ("2019-09-30T18:08:30", "2019-09-30T18:08:40"),
        ("2019-09-30T18:10:10", "2019-09-30T18:10:20"),
    ],
)
def test_saturated_against_history(time_start: str, time_stop: str):
    """
    The saturated pixels are the ones ``xrt_prep`` says it found,
    and the pixels just above and below them.
    """
    path = _path(time_start, time_stop)

    result = hinode.xrt.Filtergram.from_fits(path)

    header = astropy.io.fits.getheader(path)
    data = astropy.io.fits.getdata(path)

    saturated = data * header["EXPTIME"] >= 0.9999 * 2500
    assert np.sum(saturated) == _num_saturated(header) > 0

    expected = saturated.copy()
    expected[1:] |= saturated[:-1]
    expected[:-1] |= saturated[1:]

    assert result.saturated is not None
    assert np.all(result.saturated.ndarray[0] == expected)


def test_inputs_against_astropy_wcs():
    """
    The coordinates of each pixel must be the ones :mod:`astropy.wcs`
    makes of the header, which gives the roll as ``CROTA2``.
    """
    path = _path("2019-09-30T18:08:30", "2019-09-30T18:08:40")

    result = hinode.xrt.Filtergram.from_fits(path)

    header = astropy.io.fits.getheader(path)
    assert header["CROTA2"] != 0

    # Without the projection,
    # which :class:`named_arrays.AbstractWcsVector` does not apply.
    header["CTYPE1"] = "HPLN"
    header["CTYPE2"] = "HPLT"
    wcs = astropy.wcs.WCS(header)

    position = result.inputs[{result.axis_time: 0}].position
    shape = result.inputs.shape_wcs
    axis_x = result.axis_detector_x
    axis_y = result.axis_detector_y

    for corner in ((0, 0), (1, 2), (0, -1), (-1, -1), (100, 250)):
        index = {
            axis_x: corner[0] % shape[axis_x],
            axis_y: corner[1] % shape[axis_y],
        }

        # Astropy counts pixels from zero here, and the vertex of index `j`
        # lies half a pixel below the center of pixel `j`.
        pixel = [[index[axis_x] - 0.5, index[axis_y] - 0.5]]
        expected = wcs.wcs_pix2world(pixel, 0)[0] * u.deg

        assert np.isclose(position.x[index].ndarray, expected[0], rtol=1e-10)
        assert np.isclose(position.y[index].ndarray, expected[1], rtol=1e-10)


def test_from_fits_missing(tmp_path: pathlib.Path):
    path = _write(tmp_path / "a.fits", missing=(10, 20))

    result = hinode.xrt.Filtergram.from_fits(path)

    index = {
        result.axis_time: 0,
        result.axis_detector_x: 10,
        result.axis_detector_y: 20,
    }
    assert np.isnan(result.outputs[index])
    assert result.saturated is not None
    assert not result.saturated[index]
    assert np.sum(np.isnan(result.outputs)) == 1


def test_from_fits_concatenate(tmp_path: pathlib.Path):
    a = _write(tmp_path / "a.fits", num_x=300, num_y=200)
    b = _write(tmp_path / "b.fits", num_x=100, num_y=384)

    result = hinode.xrt.Filtergram.from_fits([a, b])

    axis_t = result.axis_time
    axis_x = result.axis_detector_x
    axis_y = result.axis_detector_y

    assert result.shape == {axis_t: 2, axis_y: 384, axis_x: 300}
    assert result.inputs.shape_wcs == {axis_x: 301, axis_y: 385}

    pad_a = {axis_t: 0, axis_y: slice(200, None)}
    pad_b = {axis_t: 1, axis_x: slice(100, None)}
    assert result.saturated is not None
    for pad in [pad_a, pad_b]:
        assert np.all(np.isnan(result.outputs[pad]))
        assert not np.any(result.saturated[pad])

    image_a = hinode.xrt.Filtergram.from_fits(a).outputs
    image_b = hinode.xrt.Filtergram.from_fits(b).outputs
    assert np.all(result.outputs[{axis_t: 0, axis_y: slice(None, 200)}] == image_a)
    assert np.all(result.outputs[{axis_t: 1, axis_x: slice(None, 100)}] == image_b)


def test_from_fits_different_filters(tmp_path: pathlib.Path):
    a = _write(tmp_path / "a.fits")
    b = _write(tmp_path / "b.fits", filter_2="Ti_poly")

    with pytest.raises(ValueError, match="different filters"):
        hinode.xrt.Filtergram.from_fits([a, b])


def test_from_fits_no_files():
    with pytest.raises(ValueError, match="No files"):
        hinode.xrt.Filtergram.from_fits([])


def test_from_time_range_no_images():
    with pytest.raises(ValueError, match="No Al_poly images"):
        hinode.xrt.Filtergram.from_time_range(
            time_start="2019-09-30T18:08:01",
            time_stop="2019-09-30T18:08:02",
        )
