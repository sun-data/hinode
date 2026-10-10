import os
import time
import pathlib
import datetime
import warnings
import pytest
import requests
import numpy as np
import astropy.io.fits
import hinode
import hinode.xrt._data
import hinode.xrt._coalign
from hinode.xrt._coalign import (
    _seconds,
    _isot,
    _start,
    _interpol,
    _roll,
    _database_roll,
    _listing,
    _lookup,
    _nearest,
    _calibration,
    _check_coalign,
    _coalign,
    _download_database,
    _age_refresh,
    _age_check,
    _name_listing,
    _url_coalign,
)
from hinode.xrt._data import _path_local

_date_obs = "2019-09-30T18:08:00.577"
"""The start of the Al_poly image which the tutorial inverts for a DEM."""


def _header(time_start: str, time_stop: str, filter: str) -> astropy.io.fits.Header:
    """The header of the one file through a filter which began during a time range."""
    urls = hinode.xrt.urls(time_start, time_stop, filter=filter)
    (path,) = hinode.xrt.download(urls).ndarray
    return astropy.io.fits.getheader(path)


def _corrected(
    header: astropy.io.fits.Header,
    coalign: str,
) -> tuple[astropy.io.fits.Header, int]:
    """
    A copy of a header with the keywords :func:`_coalign` gives it,
    and the type of calibration of its pointing.
    """
    (keywords,), calibration = _coalign(
        time=[header["DATE_OBS"]],
        filter_2=[int(header["EC_FW2"])],
        chip_sum=[int(header["CHIP_SUM"])],
        num_x=[int(header["NAXIS1"])],
        num_y=[int(header["NAXIS2"])],
        coalign=coalign,  # type: ignore[arg-type]
    )
    result = header.copy()
    result.update(keywords)
    return result, int(calibration[0])


def _response(text: str) -> requests.Response:
    """A response with the given text, made without connecting to a server."""
    response = requests.Response()
    response.status_code = 200
    response._content = text.encode()
    return response


@pytest.mark.parametrize(
    argnames="isot,expected",
    argvalues=[
        ("1979-01-01T00:00:00", 0),
        ("1979-01-02T00:00:01.5", 86401.5),
        (
            _date_obs,
            (
                datetime.datetime(2019, 9, 30, 18, 8, 0, 577000)
                - datetime.datetime(1979, 1, 1)
            ).total_seconds(),
        ),
    ],
)
def test_seconds(isot: str, expected: float) -> None:
    """Seconds since 1979 are counted without leap seconds, as ``anytim`` counts them."""
    assert _seconds([isot])[0] == expected


def test_start() -> None:
    result = _start(["20190930_1503", "20190930_2100"])
    assert np.all(result == _seconds(["2019-09-30T15:03", "2019-09-30T21:00"]))


def test_isot() -> None:
    assert _isot(_seconds(["2026-10-04T05:01:43"])[0]) == "2026-10-04T05:01:43"


def test_interpol() -> None:
    """Values beyond the samples are extrapolated from the first and the last interval."""
    xp = np.array([0.0, 1.0, 3.0])
    fp = np.array([0.0, 2.0, 3.0])
    x = np.array([-1.0, 0.5, 2.0, 5.0])
    assert np.allclose(_interpol(x, xp, fp), [-2.0, 1.0, 2.5, 4.0])


@pytest.mark.parametrize(
    argnames="isot,expected",
    argvalues=[
        # Before 2012 March 10
        ("2007-06-01T12:43:00.331", -0.353642642498),
        # From 2012 March 10 to June 8
        ("2012-06-01T12:28:28.295", -0.336584538221),
        # From 2012 June 8 to 2013 January 21
        ("2012-09-01T12:00:02.749", -0.347388714552),
        # From 2013 January 21 to the attitude anomaly of 2021 December 27
        ("2015-07-01T12:03:06.961", -0.368229240179),
        (_date_obs, -0.35857245326),
        # After the anomaly, from the database
        ("2025-06-01T12:35:26.993", 1.22919750214),
        ("2026-09-01T12:30:03.762", -0.392326414585),
    ],
)
def test_roll(isot: str, expected: float) -> None:
    """
    The roll is the one ``xrt_prep`` wrote to the headers of these files,
    which it took from ``xrt_rollangle.pro``.

    The headers are rounded to single precision,
    and between 2012 June and 2013 January differ by up to 5e-5 degrees.
    """
    result = _roll(_seconds([isot]), hinode.directory_default)
    assert np.isclose(result[0], expected, rtol=0, atol=1e-4)


def test_roll_anomaly() -> None:
    """
    During the attitude anomaly the roll follows the model of
    ``xrt_rollangle.pro``, which ends at -23 degrees,
    rather than the roll of about -0.26 degrees in the headers of the files
    ``xrt_prep`` made before the model was written.
    """
    isot = ["2021-12-27T13:00:13.873", "2021-12-27T15:01:41.946"]
    result = _roll(_seconds(isot), hinode.directory_default)
    assert np.allclose(result, [-0.31375, -23.2853], rtol=0, atol=1e-4)


@pytest.mark.parametrize(
    argnames="ref_type,expected",
    argvalues=[
        (-1, -1),
        (1, 2),
        (3, 3),
        (5, 4),
        (6, 6),
        (10, 1),
        (12, 1),
    ],
)
def test_calibration(ref_type: int, expected: int) -> None:
    assert _calibration(np.array([ref_type]))[0] == expected


def _row(name: str, changed: str) -> str:
    """A row of a listing of a directory of a database, as the server writes it."""
    return (
        f'<tr><td><a href="{name}.geny">{name}.geny</a></td>'
        f'<td align="right">{changed}  </td></tr>\n'
    )


def _age(path: pathlib.Path, seconds: float) -> None:
    """Make a file look as if it was written some seconds ago."""
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_listing(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The directory is listed again only if an image is later than the start of
    the last file and the listing is older than a day,
    or if the listing is older than a month.
    """
    calls = []

    def get(url: str, num_retry: int = 5) -> requests.Response:
        calls.append(url)
        return _response('<a href="20190930_1503.geny"> <a href="20190930_0631.geny">')

    monkeypatch.setattr(hinode.xrt._coalign, "_get", get)

    early = _seconds(["2019-09-30T12:00"])[0]
    late = _seconds(["2019-10-01T00:00"])[0]

    result = _listing("aia", early, tmp_path)
    assert list(result) == ["20190930_0631", "20190930_1503"]
    assert calls == [f"{_url_coalign}coaldb_c/"]

    # Kept, since the image is in a file before the last one
    assert _listing("aia", early, tmp_path) == result
    # Kept, since the listing is new
    _listing("aia", late, tmp_path)
    assert len(calls) == 1

    path = tmp_path / "sdb/hinode/xrt/xrt_msu_coalign/coaldb_c" / _name_listing
    _age(path, 2 * _age_refresh)
    _listing("aia", early, tmp_path)
    assert len(calls) == 1
    _listing("aia", late, tmp_path)
    assert len(calls) == 2

    # Listed again for any image once the listing is a month old
    _age(path, 2 * _age_check)
    _listing("aia", early, tmp_path)
    assert len(calls) == 3


def test_listing_changed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A file is given the time it was listed again only if the listing shows
    that it has changed, or if it is new.
    """
    text = (
        _row("20190930_0631", "2019-10-01 10:00")
        + _row("20190930_1503", "2019-10-08 10:00")
        + _row("20191001_0000", "2019-10-08 10:00")
    )
    monkeypatch.setattr(hinode.xrt._coalign, "_get", lambda url, n=5: _response(text))

    path = tmp_path / "sdb/hinode/xrt/xrt_msu_coalign/coaldb_c" / _name_listing
    path.parent.mkdir(parents=True)
    path.write_text(
        "20190930_0631 2019-10-01T10:00 1.0\n20190930_1503 2019-10-01T10:00 1.0"
    )
    _age(path, 2 * _age_check)

    start = time.time()
    result = _listing("aia", 0, tmp_path)

    assert result["20190930_0631"] == 1.0
    assert result["20190930_1503"] >= start
    assert result["20191001_0000"] >= start

    # The times are kept for the next listing
    assert _listing("aia", 0, tmp_path) == result
    assert "20190930_1503 2019-10-08T10:00 " in path.read_text()


def test_listing_unreachable(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If the directory cannot be listed again, the listing from before is used,
    and if there is none, the error is raised.
    """

    def get(url: str, num_retry: int = 5) -> requests.Response:
        raise ConnectionError("The server cannot be reached.")

    monkeypatch.setattr(hinode.xrt._coalign, "_get", get)

    with pytest.raises(ConnectionError):
        _listing("ufss", 0, tmp_path)

    path = tmp_path / "sdb/hinode/xrt/xrt_msu_coalign/coaldb_u" / _name_listing
    path.parent.mkdir(parents=True)
    path.write_text("20190930_0631 - 1.0")
    _age(path, 2 * _age_check)

    with pytest.warns(UserWarning, match="the listing from before is used"):
        assert _listing("ufss", 0, tmp_path) == {"20190930_0631": 1.0}


def test_listing_empty(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hinode.xrt._coalign, "_get", lambda url, n=5: _response(""))
    with pytest.raises(ValueError, match="has no files of the database"):
        _listing("ufss", 0, tmp_path)


@pytest.mark.parametrize(
    argnames="database,expected",
    argvalues=[
        ("aia", (5.99925, -16.8217, 10)),
        ("ufss", (6.334633, -17.498896, 3)),
    ],
)
def test_lookup(database: str, expected: tuple[float, float, int]) -> None:
    """
    An image within the tolerance of an entry is given it,
    and one further from every entry is given none.
    """
    isot = [_date_obs, "2019-09-30T18:08:00.667", "2019-09-30T18:08:00.877"]
    xcen, ycen, ref_type = _lookup(database, _seconds(isot), hinode.directory_default)
    assert np.allclose(xcen[:2], expected[0])
    assert np.allclose(ycen[:2], expected[1])
    assert np.all(ref_type[:2] == expected[2])
    assert np.isnan(xcen[2]) and np.isnan(ycen[2]) and ref_type[2] == -1


def test_lookup_before() -> None:
    """An image before the first file of a database has no entry."""
    xcen, ycen, ref_type = _lookup(
        "ufss", _seconds(["2006-01-01T00:00"]), hinode.directory_default
    )
    assert np.isnan(xcen[0]) and ref_type[0] == -1


@pytest.mark.parametrize(
    argnames="coalign,expected",
    argvalues=[
        ("aia", (5.99925, -16.8217, 1)),
        ("ufss", (6.334633, -17.498896, 3)),
    ],
)
def test_coalign(coalign: str, expected: tuple[float, float, int]) -> None:
    header = _header("2019-09-30T18:07:59", "2019-09-30T18:08:01", "Al_poly")
    assert header["DATE_OBS"] == _date_obs

    result, calibration = _corrected(header, coalign)

    assert calibration == expected[2]
    assert np.isclose(result["CRVAL1"], expected[0])
    assert np.isclose(result["CRVAL2"], expected[1])
    assert result["CRPIX1"] == result["CRPIX2"] == 192.5
    assert result["CDELT1"] == result["CDELT2"] == 1.0286
    assert np.isclose(result["CROTA2"], header["CROTA2"], rtol=0, atol=1e-6)
    assert result["CROTA1"] == result["CROTA2"]

    # The header given is not changed
    assert header["CRVAL1"] == 6.33463287354


def test_coalign_gband() -> None:
    """The images through G-band have a plate scale of their own."""
    header = _header("2019-09-30T18:09:32", "2019-09-30T18:09:33", "Gband")
    result, calibration = _corrected(header, "aia")
    assert calibration == 1
    assert result["CDELT1"] == result["CDELT2"] == 1.0302 * header["CHIP_SUM"]


def test_coalign_no_entry() -> None:
    """
    An image no database has an entry for keeps the center and the plate
    scale of its header, but is given the roll of ``xrt_rollangle.pro``.
    """
    header = _header("2019-09-30T18:07:59", "2019-09-30T18:08:01", "Al_poly")
    header["DATE_OBS"] = "2006-01-01T00:00:00.000"
    header["CDELT1"] = 1.03

    result, calibration = _corrected(header, "aia")

    assert calibration == -1
    for key in ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2"]:
        assert result[key] == header[key]
    expected = _roll(_seconds(["2006-01-01T00:00:00.000"]), hinode.directory_default)
    assert result["CROTA2"] == expected[0]


def test_coalign_invalid() -> None:
    with pytest.raises(ValueError, match="`coalign` must be one of"):
        _coalign([], [], [], [], [], "aia_cc")  # type: ignore[arg-type]


def test_check_coalign() -> None:
    _check_coalign("aia")
    _check_coalign("ufss")
    for coalign in ["AIA", None, True]:
        with pytest.raises(ValueError, match="`coalign` must be one of"):
            _check_coalign(coalign)


def test_coalign_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database which cannot be downloaded says how to do without it."""

    def lookup(*args: object) -> None:
        raise ConnectionError("The server cannot be reached.")

    monkeypatch.setattr(hinode.xrt._coalign, "_lookup", lookup)

    with pytest.raises(ConnectionError, match="coalign=None"):
        _coalign([_date_obs], [1], [1], [384], [384], "aia")


def test_download_database_unreachable(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A file which cannot be downloaded again is kept, with a warning,
    and one which has never been downloaded raises the error.
    """

    def get(url: str, num_retry: int = 5) -> requests.Response:
        raise ConnectionError("The server cannot be reached.")

    monkeypatch.setattr(hinode.xrt._data, "_get", get)

    url = f"{_url_coalign}coaldb_u/20190930_0631.geny"

    with pytest.raises(ConnectionError):
        _download_database(url, tmp_path)

    path = _path_local(url, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"SR\x00\x04")

    assert _download_database(url, tmp_path) == path
    with pytest.warns(UserWarning, match="the copy downloaded before is used"):
        assert _download_database(url, tmp_path, overwrite=True) == path


def _entries(times: list[str], ref_type: int) -> np.recarray:
    """The entries of a file of a co-alignment database at the given times."""
    num = len(times)
    return np.rec.fromarrays(
        [
            np.array([t.encode() for t in times]),
            np.ones(num),
            np.ones(num),
            np.full(num, ref_type),
        ],
        names=["DATE_OBS", "XCEN", "YCEN", "REF_TYPE"],
    )


@pytest.mark.parametrize(
    argnames="cached,age,overwrite",
    argvalues=[
        # Never downloaded
        (False, 0, False),
        # Downloaded before the listing showed the version on the server
        (True, 10, True),
        # Downloaded after it
        (True, -10, False),
    ],
)
def test_lookup_refresh(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    cached: bool,
    age: float,
    overwrite: bool,
) -> None:
    """
    A file is downloaded again only if the copy is older than the version
    the listing shows.
    """
    listed = time.time() - 100
    url = f"{_url_coalign}coaldb_c/20190930_1503.geny"
    path = _path_local(url, tmp_path)
    if cached:
        path.parent.mkdir(parents=True)
        path.touch()
        os.utime(path, (listed - age, listed - age))

    downloads = []

    def download_database(
        url: str,
        directory: pathlib.Path,
        overwrite: bool = False,
        num_retry: int = 5,
    ) -> pathlib.Path:
        downloads.append(overwrite)
        return path

    isot = "2019-09-30T15:30:00.000"

    monkeypatch.setattr(
        hinode.xrt._coalign, "_listing", lambda *args: {"20190930_1503": listed}
    )
    monkeypatch.setattr(hinode.xrt._coalign, "_download_database", download_database)
    monkeypatch.setattr(
        hinode.xrt._coalign, "_restore", lambda path: {"p0": _entries([isot], 10)}
    )

    _, _, ref_type = _lookup("aia", _seconds([isot]), tmp_path)

    assert downloads == [overwrite]
    assert ref_type[0] == 10


@pytest.mark.parametrize(
    argnames="cached,latest,age,downloads_expected",
    argvalues=[
        # Never downloaded
        (False, 5.0, 0, [False]),
        # After the last measurement, in a database downloaded long ago
        (True, 20.0, 2 * _age_refresh, [True]),
        # Within the database
        (True, 5.0, 2 * _age_refresh, []),
        # After the last measurement, in a new database
        (True, 20.0, 0, []),
        # Within the database, but checked again after a month
        (True, 5.0, 2 * _age_check, [True]),
    ],
)
def test_database_roll_refresh(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    cached: bool,
    latest: float,
    age: float,
    downloads_expected: list[bool],
) -> None:
    """
    The database of the roll is downloaded again only for an image after its
    last measurement, if it is a day old, or for any image, if it is a month
    old.
    """
    path = _path_local(_url_coalign + "xrt_rollangle_db.geny", tmp_path)
    if cached:
        path.parent.mkdir(parents=True)
        path.touch()
        _age(path, age)

    downloads = []

    def download_database(
        url: str,
        directory: pathlib.Path,
        overwrite: bool = False,
        num_retry: int = 5,
    ) -> pathlib.Path:
        downloads.append(overwrite)
        return path

    def restore(path: pathlib.Path) -> dict:
        return {"p0": np.array([0.0, 10.0]), "p1": np.array([1.0, 2.0])}

    monkeypatch.setattr(hinode.xrt._coalign, "_download_database", download_database)
    monkeypatch.setattr(hinode.xrt._coalign, "_restore", restore)

    times, angles = _database_roll(latest, tmp_path)

    assert downloads == downloads_expected
    assert np.all(times == [0, 10]) and np.all(angles == [1, 2])


def test_roll_extrapolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The roll of an image after the last measurement of the database is
    extrapolated, with a warning.
    """
    end = _seconds(["2021-12-27T15:10"])[0]
    times = end + np.array([100.0, 200.0])
    monkeypatch.setattr(
        hinode.xrt._coalign,
        "_database_roll",
        lambda *args: (times, np.array([1.0, 2.0])),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.allclose(_roll(end + np.array([150.0]), pathlib.Path()), -1.5)

    with pytest.warns(UserWarning, match="1 of 2 images began outside"):
        result = _roll(end + np.array([150.0, 300.0]), pathlib.Path())
    assert np.allclose(result, [-1.5, -3.0])


def test_nearest() -> None:
    """The entries may be in any order, and a tie goes to the earlier one."""
    times = np.array([30.0, 10.0, 20.0])
    index, difference = _nearest(times, np.array([0.0, 14.0, 16.0, 25.0, 40.0]))
    assert index.tolist() == [1, 1, 2, 2, 0]
    assert difference.tolist() == [10, 4, 4, 5, 10]


def test_lookup_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Only the files which cover an image without an entry yet are downloaded,
    and not the ones between them,
    or the next one for an image near its start which the file before has
    an entry for.
    """
    isot = [
        "2019-09-30T00:30:00.000",
        "2019-09-30T02:30:00.000",
        "2019-09-30T02:59:59.950",
    ]
    downloads = []

    def download_database(url: str, *args: object) -> str:
        downloads.append(url.split("/")[-1])
        return url

    def restore(path: str) -> dict:
        times = [t for t in isot if t[11:13] == path.split("_")[-1][:2]]
        return {"p0": _entries(times, 3)}

    names = ["20190930_0000", "20190930_0100", "20190930_0200", "20190930_0300"]
    monkeypatch.setattr(
        hinode.xrt._coalign, "_listing", lambda *args: dict.fromkeys(names, 0.0)
    )
    monkeypatch.setattr(hinode.xrt._coalign, "_download_database", download_database)
    monkeypatch.setattr(hinode.xrt._coalign, "_restore", restore)

    _, _, ref_type = _lookup("ufss", _seconds(isot), pathlib.Path("cache"))

    assert downloads == ["20190930_0000.geny", "20190930_0200.geny"]
    assert ref_type.tolist() == [3, 3, 3]
