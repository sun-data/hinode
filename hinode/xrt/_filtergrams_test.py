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
from hinode.xrt._filtergrams import _vignetting

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
    values: None | dict[tuple[int, int], float] = None,
    image_type: None | str = None,
    history: None | tuple[str, str] = None,
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
    values
        If not :obj:`None`, the column and row of pixels,
        and the values to give them.
    image_type
        If not :obj:`None`, the type of image the copy says it is.
    history
        If not :obj:`None`, a piece of text to replace in each line of the
        history of the copy, and the text to replace it with.
    """
    path = _path("2019-09-30T18:08:30", "2019-09-30T18:08:40")
    data = astropy.io.fits.getdata(path)
    header = astropy.io.fits.getheader(path)

    if missing is not None:
        data[missing[1], missing[0]] = -999

    if values is not None:
        for (column, row), value in values.items():
            data[row, column] = value

    data = data[:num_y, :num_x]

    if filter_2 is not None:
        header["EC_FW2_"] = filter_2

    if image_type is not None:
        header["EC_IMTY_"] = image_type

    if history is not None:
        old, new = history
        lines = [str(line).replace(old, new) for line in header["HISTORY"]]
        del header["HISTORY"]
        for line in lines:
            header.add_history(line)

    astropy.io.fits.writeto(file, data, header)

    return file


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

    def test_axes(self, array: hinode.xrt.Filtergram) -> None:
        axes = {array.axis_time, array.axis_detector_x, array.axis_detector_y}
        assert set(array.outputs.shape) == axes
        assert set(array.inputs.crpix.components) == axes - {array.axis_time}
        assert set(array.inputs.pc.position.x.components) == axes - {array.axis_time}
        assert set(array.inputs.pc.position.y.components) == axes - {array.axis_time}

    def test_shape(self, array: hinode.xrt.Filtergram) -> None:
        assert array.shape[array.axis_time] == 4
        assert array.shape[array.axis_detector_x] == 384
        assert array.shape[array.axis_detector_y] == 384

    def test_time(self, array: hinode.xrt.Filtergram) -> None:
        time = array.inputs.time
        assert time.shape == {array.axis_time: array.shape[array.axis_time]}
        assert np.all(time.ndarray >= _time_start)
        assert np.all(time.ndarray < _time_stop)
        assert np.all(np.diff(time.ndarray.jd) > 0)

    def test_timedelta(self, array: hinode.xrt.Filtergram) -> None:
        timedelta = array.timedelta
        assert timedelta.shape == {array.axis_time: array.shape[array.axis_time]}
        assert np.all(timedelta > 10 * u.s)

    def test_outputs(self, array: hinode.xrt.Filtergram) -> None:
        outputs = array.outputs
        assert outputs.unit == u.DN / u.s
        assert not np.any(outputs == -999 * u.DN / u.s)
        assert np.nanmedian(outputs) > 0 * u.DN / u.s

    def test_saturated(self, array: hinode.xrt.Filtergram) -> None:
        saturated = array.saturated
        assert isinstance(saturated, na.ScalarArray)
        assert saturated.ndarray.dtype == bool
        assert saturated.shape == array.outputs.shape

        # Event E saturates the frames at 18:08:19 and 18:08:37
        num = saturated.sum(axis=(array.axis_detector_x, array.axis_detector_y))
        assert np.all(num.ndarray == [0, 6, 3, 0])

    def test_filter(self, array: hinode.xrt.Filtergram) -> None:
        assert array.filter == "Al_poly"

    def test_getitem(self, array: hinode.xrt.Filtergram) -> None:
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
def test_saturated_against_history(time_start: str, time_stop: str) -> None:
    """
    The saturated pixels are the ones ``xrt_prep`` says it found,
    and the pixels just above and below them.
    """
    path = _path(time_start, time_stop)

    result = hinode.xrt.Filtergram.from_fits(path)

    header = astropy.io.fits.getheader(path)
    data = astropy.io.fits.getdata(path)

    signal = data * header["EXPTIME"] * _vignetting_numpy(header)

    # ``xrt_prep`` set the saturated pixels to exactly the saturation level
    saturated = signal >= 0.9999 * 2500
    assert np.sum(saturated) == _num_saturated(header) > 0
    assert np.allclose(signal[saturated], 2500, rtol=1e-5)

    expected = saturated.copy()
    expected[1:] |= saturated[:-1]
    expected[:-1] |= saturated[1:]

    assert result.saturated is not None
    assert np.all(result.saturated.ndarray[0] == expected)


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


def test_saturated_vignetting(tmp_path: pathlib.Path) -> None:
    """
    A pixel is saturated if its value is the saturation level once the
    vignetting correction and the exposure time are taken out of it,
    not if its value alone, times the exposure time, is above the level.
    """
    path = _path("2019-09-30T18:08:30", "2019-09-30T18:08:40")
    header = astropy.io.fits.getheader(path)
    exptime = header["EXPTIME"]
    vignetting = _vignetting_numpy(header)

    # Two corners, far from the center of the CCD and from event E
    saturated = (0, 0)
    bright = (383, 383)

    level_saturated = 2500 / (exptime * vignetting[saturated[1], saturated[0]])
    level_bright = 0.999 * 2500 / (exptime * vignetting[bright[1], bright[0]])

    # Without the vignetting, the bright pixel would seem saturated
    assert level_bright * exptime > 2500

    file = _write(
        file=tmp_path / "a.fits",
        values={saturated: level_saturated, bright: level_bright},
    )

    result = hinode.xrt.Filtergram.from_fits(file)

    assert result.saturated is not None
    axis_x = result.axis_detector_x
    axis_y = result.axis_detector_y
    index = {result.axis_time: 0}

    assert result.saturated[index | {axis_x: saturated[0], axis_y: saturated[1]}]
    assert result.saturated[index | {axis_x: saturated[0], axis_y: saturated[1] + 1}]
    assert not result.saturated[index | {axis_x: bright[0], axis_y: bright[1]}]


def test_saturated_dark(tmp_path: pathlib.Path) -> None:
    """``xrt_prep`` does not correct a dark frame for vignetting."""
    path = _path("2019-09-30T18:08:30", "2019-09-30T18:08:40")
    exptime = astropy.io.fits.getheader(path)["EXPTIME"]

    file = _write(
        file=tmp_path / "a.fits",
        values={(0, 0): 2500 / exptime},
        image_type="dark",
    )

    result = hinode.xrt.Filtergram.from_fits(file)

    assert result.saturated is not None
    index = {
        result.axis_time: 0,
        result.axis_detector_x: 0,
        result.axis_detector_y: 0,
    }
    assert result.saturated[index]


def test_saturated_level_from_history(tmp_path: pathlib.Path) -> None:
    """The saturation level is the one the history of the file records."""
    path = _path("2019-09-30T18:08:30", "2019-09-30T18:08:40")
    header = astropy.io.fits.getheader(path)
    pixel = (0, 0)
    value = 2200 / (header["EXPTIME"] * _vignetting_numpy(header)[pixel[::-1]])

    def saturated(file: pathlib.Path, history: None | tuple[str, str]) -> bool:
        _write(file, values={pixel: value}, history=history)
        result = hinode.xrt.Filtergram.from_fits(file)
        assert result.saturated is not None
        index = {
            result.axis_time: 0,
            result.axis_detector_x: pixel[0],
            result.axis_detector_y: pixel[1],
        }
        return bool(result.saturated[index])

    history = ("with value = 2500", "with value = 2000")
    assert not saturated(tmp_path / "a.fits", history=None)
    assert saturated(tmp_path / "b.fits", history=history)


@pytest.mark.parametrize(
    argnames="history,match",
    argvalues=[
        (("--> 1.00 sec", "--> 2.00 sec"), "not in DN per second"),
        (("(XRT_RENORMALIZE)", "(XRT_OTHER)"), "not in DN per second"),
        (("(XRT_SATURATED_PIXELS)", "(XRT_OTHER)"), "saturated pixels"),
    ],
)
def test_from_fits_history(
    tmp_path: pathlib.Path,
    history: tuple[str, str],
    match: str,
) -> None:
    """
    A file which ``xrt_prep`` did not normalize to DN per second,
    or whose saturation level is not recorded, is not loaded.
    """
    file = _write(tmp_path / "a.fits", history=history)
    with pytest.raises(ValueError, match=re.escape(match)):
        hinode.xrt.Filtergram.from_fits(file)


def test_inputs_against_astropy_wcs() -> None:
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


def test_from_fits_missing(tmp_path: pathlib.Path) -> None:
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


def test_from_fits_concatenate(tmp_path: pathlib.Path) -> None:
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


def test_from_fits_different_filters(tmp_path: pathlib.Path) -> None:
    a = _write(tmp_path / "a.fits")
    b = _write(tmp_path / "b.fits", filter_2="Ti_poly")

    with pytest.raises(ValueError, match="different filters"):
        hinode.xrt.Filtergram.from_fits([a, b])


def test_from_fits_no_files() -> None:
    with pytest.raises(ValueError, match="No files"):
        hinode.xrt.Filtergram.from_fits([])


def test_from_fits_array() -> None:
    """The array of paths that :func:`hinode.xrt.download` returns is accepted."""
    urls = hinode.xrt.urls(_time_start, _time_stop)
    paths = hinode.xrt.download(urls)

    result = hinode.xrt.Filtergram.from_fits(paths)
    expected = hinode.xrt.Filtergram.from_fits(list(paths.ndarray))

    assert result.shape == expected.shape
    assert np.all(result.outputs == expected.outputs)


def test_from_time_range_directory(tmp_path: pathlib.Path) -> None:
    """The images are downloaded into the given directory."""
    kwargs = dict(
        time_start="2019-09-30T18:08:30",
        time_stop="2019-09-30T18:08:40",
        directory=tmp_path,
    )

    result = hinode.xrt.Filtergram.from_time_range(**kwargs)

    (file,) = tmp_path.rglob("*.fits")
    mtime = file.stat().st_mtime_ns

    assert result.shape[result.axis_time] == 1

    # The file is downloaded again only if asked to be
    hinode.xrt.Filtergram.from_time_range(**kwargs)
    assert file.stat().st_mtime_ns == mtime

    hinode.xrt.Filtergram.from_time_range(**kwargs, overwrite=True)
    assert file.stat().st_mtime_ns != mtime


def test_from_time_range_no_images() -> None:
    with pytest.raises(ValueError, match="No Al_poly images"):
        hinode.xrt.Filtergram.from_time_range(
            time_start="2019-09-30T18:08:01",
            time_stop="2019-09-30T18:08:02",
        )
