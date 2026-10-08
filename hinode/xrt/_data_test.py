import time
import pathlib
import datetime
import pytest
import requests
import numpy as np
import astropy.time
import astropy.io.fits
import named_arrays as na
import hinode
import hinode.xrt._data
from hinode.xrt._data import _get, _hours, _files, _header, _filter


def _names(urls: na.ScalarArray) -> list[str]:
    """The hour directory and the name of each file."""
    return ["/".join(str(url).split("/")[-2:]) for url in urls.ndarray]


def _response(url: str, status: int, content: bytes = b"") -> requests.Response:
    """A response from a server, made without connecting to one."""
    response = requests.Response()
    response.url = url
    response.status_code = status
    response.reason = "test"
    response._content = content
    return response


@pytest.mark.parametrize(
    argnames="time_start,time_stop,filter,expected",
    argvalues=[
        (
            "2019-09-30T18:08:00",
            "2019-09-30T18:09:00",
            "Al_poly",
            [
                "H1800/L1_XRT20190930_180800.5.fits",
                "H1800/L1_XRT20190930_180819.0.fits",
                "H1800/L1_XRT20190930_180837.5.fits",
                "H1800/L1_XRT20190930_180856.0.fits",
            ],
        ),
        (
            "2019-09-30T18:09:00",
            "2019-09-30T18:10:00",
            "Gband",
            [
                "H1800/L1_XRT20190930_180932.5.fits",
                "H1800/L1_XRT20190930_180935.0.fits",
                "H1800/L1_XRT20190930_180938.0.fits",
            ],
        ),
        # The range spans two of the hourly directories of the archive
        (
            "2019-09-30T17:59:30",
            "2019-09-30T18:00:30",
            "Al_poly",
            [
                "H1700/L1_XRT20190930_175935.9.fits",
                "H1700/L1_XRT20190930_175954.4.fits",
                "H1800/L1_XRT20190930_180012.9.fits",
                "H1800/L1_XRT20190930_180026.9.fits",
            ],
        ),
        # A start time in TT, which is 18:00:00 TT but 17:58:50.8 UTC,
        # so the range begins in the directory of the hour before
        (
            astropy.time.Time("2019-09-30T18:00:00", scale="tt"),
            "2019-09-30T18:00:30",
            "Al_poly",
            [
                "H1700/L1_XRT20190930_175903.3.fits",
                "H1700/L1_XRT20190930_175917.4.fits",
                "H1700/L1_XRT20190930_175935.9.fits",
                "H1700/L1_XRT20190930_175954.4.fits",
                "H1800/L1_XRT20190930_180012.9.fits",
                "H1800/L1_XRT20190930_180026.9.fits",
            ],
        ),
        # No image began during the range
        (
            "2019-09-30T18:08:01",
            "2019-09-30T18:08:02",
            "Al_poly",
            [],
        ),
    ],
)
def test_urls(
    time_start: str | astropy.time.Time,
    time_stop: str,
    filter: str,
    expected: list[str],
) -> None:
    result = hinode.xrt.urls(
        time_start=time_start,
        time_stop=time_stop,
        filter=filter,
        axis_time="t",
    )
    assert isinstance(result, na.ScalarArray)
    assert result.shape == {"t": len(expected)}
    assert _names(result) == expected

    for url in result.ndarray:
        header = _header(str(url))
        assert _filter(header) == filter
        time = astropy.time.Time(header["DATE_OBS"])
        assert astropy.time.Time(time_start) <= time < astropy.time.Time(time_stop)


def test_urls_image_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the images of type ``normal`` are found, not the dark frames."""

    expected = hinode.xrt.urls("2019-09-30T18:08:00", "2019-09-30T18:09:00")
    dark = str(expected.ndarray[1])

    header_original = hinode.xrt._data._header

    def header(url: str, num_retry: int = 5) -> astropy.io.fits.Header:
        result = header_original(url, num_retry)
        if url == dark:
            result["EC_IMTY_"] = "dark"
        return result

    monkeypatch.setattr(hinode.xrt._data, "_header", header)

    result = hinode.xrt.urls("2019-09-30T18:08:00", "2019-09-30T18:09:00")

    assert list(result.ndarray) == [u for u in expected.ndarray if u != dark]


@pytest.mark.parametrize(
    argnames="time_start,time_stop,expected",
    argvalues=[
        (
            astropy.time.Time("2019-09-30T17:30:00"),
            astropy.time.Time("2019-09-30T19:00:00"),
            [17, 18, 19],
        ),
        (
            astropy.time.Time("2019-09-30T18:00:00", scale="tt"),
            astropy.time.Time("2019-09-30T18:30:00"),
            [17, 18],
        ),
        # The hour before a leap second
        (
            astropy.time.Time("2016-12-31T22:30:00"),
            astropy.time.Time("2017-01-01T00:30:00"),
            [22, 23, 0],
        ),
        (
            astropy.time.Time("2016-12-31T23:59:60.5"),
            astropy.time.Time("2017-01-01T00:00:10"),
            [23, 0],
        ),
    ],
)
def test_hours(
    time_start: astropy.time.Time,
    time_stop: astropy.time.Time,
    expected: list[int],
) -> None:
    result = _hours(time_start, time_stop)
    assert all(isinstance(hour, datetime.datetime) for hour in result)
    assert [hour.hour for hour in result] == expected
    assert len(set(result)) == len(result)


def test_files_missing_hour() -> None:
    """An hour with no directory in the archive has no files."""
    assert _files(datetime.datetime(2000, 1, 1, 0)) == []


@pytest.mark.parametrize(
    argnames="statuses,expected",
    argvalues=[
        ([200], None),
        ([503, 200], None),
        ([429, 500, 200], None),
        ([404], FileNotFoundError),
        ([403, 200], requests.exceptions.HTTPError),
        ([503, 503, 503], ConnectionError),
    ],
)
def test_get(
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[int],
    expected: None | type[Exception],
) -> None:
    """
    A request is tried again after waiting longer each time,
    unless the error cannot change.
    """
    url = "https://example.com/a.fits"

    calls: list[str] = []
    sleeps: list[float] = []

    def get(url: str, **kwargs: object) -> requests.Response:
        status = statuses[len(calls)]
        calls.append(url)
        return _response(url, status, content=b"SIMPLE  =")

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    num_retry = 3

    if expected is None:
        result = _get(url, num_retry=num_retry)
        assert result.status_code == 200
        num_calls = len(statuses)
    else:
        with pytest.raises(expected):
            _get(url, num_retry=num_retry)
        num_calls = 1 if expected is not ConnectionError else num_retry

    assert len(calls) == num_calls
    assert sleeps == [2**i for i in range(num_calls - 1)]


def test_header(tmp_path: pathlib.Path) -> None:
    urls = hinode.xrt.urls("2019-09-30T18:08:30", "2019-09-30T18:08:40")
    (url,) = urls.ndarray

    result = _header(str(url))

    (path,) = hinode.xrt.download(urls, directory=tmp_path).ndarray
    expected = astropy.io.fits.getheader(path)

    assert list(result.keys()) == list(expected.keys())
    for key in ["DATE_OBS", "EXPTIME", "EC_FW1_", "EC_FW2_", "NAXIS1", "CROTA2"]:
        assert result[key] == expected[key]


@pytest.mark.parametrize(
    argnames="filter_1,filter_2,expected",
    argvalues=[
        ("Al_poly", "Open", "Al_poly"),
        ("Open", "Gband", "Gband"),
        ("Al_poly", "Ti_poly", "Al_poly/Ti_poly"),
        ("Open", "Open", "Open"),
        ("Al_poly ", "Open    ", "Al_poly"),
    ],
)
def test_filter(filter_1: str, filter_2: str, expected: str) -> None:
    header = astropy.io.fits.Header()
    header["EC_FW1_"] = filter_1
    header["EC_FW2_"] = filter_2
    assert _filter(header) == expected


def test_download(tmp_path: pathlib.Path) -> None:
    urls = hinode.xrt.urls("2019-09-30T18:08:00", "2019-09-30T18:08:40")

    result = hinode.xrt.download(urls, directory=tmp_path)

    assert isinstance(result, na.ScalarArray)
    assert result.shape == urls.shape

    for url, path in zip(urls.ndarray, result.ndarray):
        path = pathlib.Path(path)
        assert path.is_relative_to(tmp_path)
        assert path.name == str(url).split("/")[~0]
        assert astropy.io.fits.getheader(path)["DATA_LEV"] == 1

    assert not list(tmp_path.rglob("*.part"))

    # The files are not downloaded again
    mtime = [pathlib.Path(path).stat().st_mtime_ns for path in result.ndarray]
    again = hinode.xrt.download(urls, directory=tmp_path)
    assert np.all(again.ndarray == result.ndarray)
    assert mtime == [pathlib.Path(path).stat().st_mtime_ns for path in again.ndarray]


def test_download_repeated(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A URL given more than once is downloaded once."""
    urls = hinode.xrt.urls("2019-09-30T18:08:00", "2019-09-30T18:08:30")
    url_a, url_b = urls.ndarray

    urls = na.ScalarArray(np.array([url_a, url_b, url_a]), axes="t")

    calls: list[str] = []
    get_original = hinode.xrt._data._get

    def get(url: str, num_retry: int = 5) -> requests.Response:
        calls.append(url)
        return get_original(url, num_retry)

    monkeypatch.setattr(hinode.xrt._data, "_get", get)

    result = hinode.xrt.download(urls, directory=tmp_path)

    assert sorted(calls) == sorted([url_a, url_b])
    assert result.shape == {"t": 3}
    assert result.ndarray[0] == result.ndarray[2] != result.ndarray[1]
    assert len(list(tmp_path.rglob("*.fits"))) == 2
    assert not list(tmp_path.rglob("*.part"))


def test_download_not_fits(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response which is not a FITS file, such as an error page, is not kept."""
    url = (
        hinode.xrt._data._url_level_1 + "2019/09/30/H1800/L1_XRT20190930_180837.5.fits"
    )

    def get(url: str, num_retry: int = 5) -> requests.Response:
        return _response(url, 200, content=b"<html>Down for maintenance</html>")

    monkeypatch.setattr(hinode.xrt._data, "_get", get)

    urls = na.ScalarArray(np.array([url]), axes="t")

    with pytest.raises(ValueError, match="is not a FITS file"):
        hinode.xrt.download(urls, directory=tmp_path)

    assert not [p for p in tmp_path.rglob("*") if p.is_file()]
