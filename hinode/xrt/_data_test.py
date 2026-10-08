import os
import re
import time
import pathlib
import datetime
import threading
import http.server
import pytest
import requests
import joblib
import numpy as np
import astropy.time
import astropy.io.fits
import named_arrays as na
import hinode
import hinode.xrt._data
from hinode.xrt._data import (
    _get,
    _hours,
    _files,
    _header,
    _filter,
    _replace,
    _signature_fits,
    _url_level_1,
)

_url_test = _url_level_1 + "2019/09/30/H1800/L1_XRT20190930_180837.5.fits"
"""The URL of the file in which event E saturates."""


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


def _get_fits(url: str, num_retry: int = 5) -> requests.Response:
    """A response holding a FITS file, made without connecting to a server."""
    return _response(url, 200, content=_signature_fits + b" new")


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
    argnames="error",
    argvalues=[
        FileNotFoundError("removed from the archive"),
        requests.exceptions.HTTPError("416 Range Not Satisfiable"),
        ValueError("no END card"),
    ],
)
def test_urls_unreadable(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """A file whose header cannot be read is left out with a warning."""

    expected = hinode.xrt.urls("2019-09-30T18:08:00", "2019-09-30T18:09:00")
    bad = str(expected.ndarray[1])

    header_original = hinode.xrt._data._header

    def header(url: str, num_retry: int = 5) -> astropy.io.fits.Header:
        if url == bad:
            raise error
        return header_original(url, num_retry)

    monkeypatch.setattr(hinode.xrt._data, "_header", header)

    with pytest.warns(UserWarning, match=re.escape(bad)):
        result = hinode.xrt.urls("2019-09-30T18:08:00", "2019-09-30T18:09:00")

    assert list(result.ndarray) == [u for u in expected.ndarray if u != bad]


def test_urls_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A header which cannot be read because the archive cannot be reached
    is an error rather than a warning.
    """

    def header(url: str, num_retry: int = 5) -> astropy.io.fits.Header:
        raise ConnectionError(f"Could not get {url}.")

    monkeypatch.setattr(hinode.xrt._data, "_header", header)

    with pytest.raises(ConnectionError):
        hinode.xrt.urls("2019-09-30T18:08:00", "2019-09-30T18:09:00")


@pytest.mark.parametrize(
    argnames="time_start,time_stop,expected",
    argvalues=[
        # A range which stops on the hour does not include that hour
        (
            astropy.time.Time("2019-09-30T17:30:00"),
            astropy.time.Time("2019-09-30T19:00:00"),
            [17, 18],
        ),
        (
            astropy.time.Time("2019-09-30T17:30:00"),
            astropy.time.Time("2019-09-30T19:00:00.001"),
            [17, 18, 19],
        ),
        (
            astropy.time.Time("2019-09-30T19:00:00"),
            astropy.time.Time("2019-09-30T19:00:00"),
            [],
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
        (
            astropy.time.Time("2016-12-31T23:00:00"),
            astropy.time.Time("2017-01-01T00:00:00"),
            [23],
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
    argnames="responses,expected",
    argvalues=[
        ([200], None),
        ([503, 200], None),
        ([429, 500, 200], None),
        ([408, 200], None),
        ([requests.exceptions.ConnectionError, 200], None),
        ([requests.exceptions.ReadTimeout, 200], None),
        ([requests.exceptions.ChunkedEncodingError, 200], None),
        ([404], FileNotFoundError),
        ([403, 200], requests.exceptions.HTTPError),
        ([requests.exceptions.InvalidURL, 200], requests.exceptions.InvalidURL),
        ([503, 503, 503], ConnectionError),
    ],
)
def test_get(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[int | type[Exception]],
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
        response = responses[len(calls)]
        calls.append(url)
        if isinstance(response, type):
            raise response(url)
        return _response(url, response, content=_signature_fits)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    num_retry = 3

    if expected is None:
        result = _get(url, num_retry=num_retry)
        assert result.status_code == 200
        num_calls = len(responses)
    else:
        with pytest.raises(expected):
            _get(url, num_retry=num_retry)
        num_calls = 1 if expected is not ConnectionError else num_retry

    assert len(calls) == num_calls
    assert sleeps == [2**i for i in range(num_calls - 1)]


def test_get_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A response shorter than the length the server promised is tried again,
    since the archive sometimes ends a response early.
    """
    paths: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):

        def do_GET(self) -> None:
            paths.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(_signature_fits)
            self.close_connection = True

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    try:
        with pytest.raises(ConnectionError):
            _get(f"http://127.0.0.1:{server.server_address[1]}/a.fits", num_retry=2)
    finally:
        server.shutdown()
        server.server_close()

    assert paths == ["/a.fits", "/a.fits"]


def test_header(tmp_path: pathlib.Path) -> None:
    urls = hinode.xrt.urls("2019-09-30T18:08:30", "2019-09-30T18:08:40")
    (url,) = urls.ndarray

    result = _header(str(url))

    (path,) = hinode.xrt.download(urls, directory=tmp_path).ndarray
    expected = astropy.io.fits.getheader(path)

    assert list(result.keys()) == list(expected.keys())
    for key in ["DATE_OBS", "EXPTIME", "EC_FW1_", "EC_FW2_", "NAXIS1", "CROTA2"]:
        assert result[key] == expected[key]


def test_header_memory(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The headers are cached in whichever cache :data:`hinode.memory` is."""
    text = astropy.io.fits.Header(dict(SIMPLE=True, BITPIX=8, NAXIS=0)).tostring()

    calls: list[str] = []

    def get(
        url: str,
        num_retry: int = 5,
        headers: None | dict[str, str] = None,
    ) -> requests.Response:
        calls.append(url)
        return _response(url, 206, content=text.encode("ascii"))

    monkeypatch.setattr(hinode.xrt._data, "_get", get)
    monkeypatch.setattr(hinode, "memory", joblib.Memory(tmp_path, verbose=0))

    url = "https://example.com/a.fits"
    for _ in range(2):
        assert _header(url)["NAXIS"] == 0

    assert calls == [url]
    assert list(tmp_path.rglob("output.pkl"))


def test_memory_location() -> None:
    """
    The downloaded files are not in the cache,
    so that clearing the cache does not delete them.
    """
    location = pathlib.Path(hinode.memory.store_backend.location)
    path = hinode.directory_default / "/".join(_url_test.split("/")[3:])
    assert not path.is_relative_to(location)


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

    def get(url: str, num_retry: int = 5) -> requests.Response:
        return _response(url, 200, content=b"<html>Down for maintenance</html>")

    monkeypatch.setattr(hinode.xrt._data, "_get", get)

    urls = na.ScalarArray(np.array([_url_test]), axes="t")

    with pytest.raises(ValueError, match="is not a FITS file"):
        hinode.xrt.download(urls, directory=tmp_path)

    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


@pytest.mark.parametrize(
    argnames="url",
    argvalues=[
        "https://example.com",
        _url_level_1 + "2019/09/30/H1800/",
    ],
)
def test_download_not_a_file(tmp_path: pathlib.Path, url: str) -> None:
    urls = na.ScalarArray(np.array([url]), axes="t")
    with pytest.raises(ValueError, match="not the URL of a file"):
        hinode.xrt.download(urls, directory=tmp_path)


def test_download_overwrite_in_use(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A file which cannot be replaced, such as one which is open on Windows,
    is an error if a new copy was asked for.
    """
    monkeypatch.setattr(hinode.xrt._data, "_get", _get_fits)

    urls = na.ScalarArray(np.array([_url_test]), axes="t")
    (path,) = hinode.xrt.download(urls, directory=tmp_path).ndarray
    path = pathlib.Path(path)
    path.write_bytes(b"old")

    def replace(
        source: pathlib.Path,
        destination: pathlib.Path,
        num_retry: int = 5,
    ) -> None:
        raise PermissionError(destination)

    monkeypatch.setattr(hinode.xrt._data, "_replace", replace)

    with pytest.raises(PermissionError):
        hinode.xrt.download(urls, directory=tmp_path, overwrite=True)

    assert path.read_bytes() == b"old"
    assert not list(tmp_path.rglob("*.part"))


def test_download_in_use(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file which another process downloaded first and still has open is kept."""
    monkeypatch.setattr(hinode.xrt._data, "_get", _get_fits)

    def replace(
        source: pathlib.Path,
        destination: pathlib.Path,
        num_retry: int = 5,
    ) -> None:
        destination.write_bytes(b"other")
        raise PermissionError(destination)

    monkeypatch.setattr(hinode.xrt._data, "_replace", replace)

    urls = na.ScalarArray(np.array([_url_test]), axes="t")
    (path,) = hinode.xrt.download(urls, directory=tmp_path).ndarray

    assert pathlib.Path(path).read_bytes() == b"other"
    assert not list(tmp_path.rglob("*.part"))


def test_download_failed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A download which fails part of the way through leaves no file behind."""
    monkeypatch.setattr(hinode.xrt._data, "_get", _get_fits)

    def replace(
        source: pathlib.Path,
        destination: pathlib.Path,
        num_retry: int = 5,
    ) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(hinode.xrt._data, "_replace", replace)

    urls = na.ScalarArray(np.array([_url_test]), axes="t")
    with pytest.raises(OSError, match="No space"):
        hinode.xrt.download(urls, directory=tmp_path)

    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


@pytest.mark.skipif(os.name == "nt", reason="Windows does not use permission bits")
def test_download_permissions(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A downloaded file has the permissions the umask gives a new file,
    so that others can read the files in a shared directory.
    """
    monkeypatch.setattr(hinode.xrt._data, "_get", _get_fits)

    urls = na.ScalarArray(np.array([_url_test]), axes="t")

    umask = os.umask(0o022)
    try:
        (path,) = hinode.xrt.download(urls, directory=tmp_path).ndarray
    finally:
        os.umask(umask)

    assert pathlib.Path(path).stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    argnames="num_fail",
    argvalues=[0, 2, 5],
)
def test_replace(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    num_fail: int,
) -> None:
    """A file is moved into place once the move is permitted, if it is in time."""
    source = tmp_path / "a.part"
    source.write_bytes(b"new")
    destination = tmp_path / "a.fits"
    destination.write_bytes(b"old")

    replace_original = os.replace
    calls: list[pathlib.Path] = []
    sleeps: list[float] = []

    def replace(src: pathlib.Path, dst: pathlib.Path) -> None:
        calls.append(src)
        if len(calls) <= num_fail:
            raise PermissionError(dst)
        replace_original(src, dst)

    num_retry = 5

    with monkeypatch.context() as m:
        m.setattr(os, "replace", replace)
        m.setattr(time, "sleep", sleeps.append)
        if num_fail < num_retry:
            _replace(source, destination, num_retry=num_retry)
        else:
            with pytest.raises(PermissionError):
                _replace(source, destination, num_retry=num_retry)

    expected = b"new" if num_fail < num_retry else b"old"
    assert destination.read_bytes() == expected
    assert len(calls) == min(num_fail + 1, num_retry)
    delay = hinode.xrt._data._delay_replace
    assert sleeps == [delay * 2**i for i in range(min(num_fail, num_retry - 1))]
