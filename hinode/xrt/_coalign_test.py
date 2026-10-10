import os
import time
import pathlib
import datetime
import pytest
import requests
import numpy as np
import astropy.io.fits
import hinode
import hinode.xrt._coalign
from hinode.xrt._coalign import (
    _seconds,
    _start,
    _interpol,
    _roll,
    _listing,
    _lookup,
    _calibration,
    _coalign,
    _age_refresh,
    _name_listing,
    _url_coalign,
)

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


def test_listing(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The directory is listed again only if an image is later than the start of
    the last file and the listing is older than a day.
    """
    calls = []

    def get(url: str, num_retry: int = 5) -> requests.Response:
        calls.append(url)
        return _response('<a href="20190930_1503.geny"> <a href="20190930_0631.geny">')

    monkeypatch.setattr(hinode.xrt._coalign, "_get", get)

    early = _seconds(["2019-09-30T12:00"])[0]
    late = _seconds(["2019-10-01T00:00"])[0]

    assert _listing("aia", early, tmp_path) == ["20190930_0631", "20190930_1503"]
    assert calls == [f"{_url_coalign}coaldb_c/"]

    # Kept, since the image is in a file before the last one
    _listing("aia", early, tmp_path)
    # Kept, since the listing is new
    _listing("aia", late, tmp_path)
    assert len(calls) == 1

    path = tmp_path / "sdb/hinode/xrt/xrt_msu_coalign/coaldb_c" / _name_listing
    old = time.time() - 2 * _age_refresh
    os.utime(path, (old, old))
    _listing("aia", early, tmp_path)
    assert len(calls) == 1
    _listing("aia", late, tmp_path)
    assert len(calls) == 2


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


@pytest.mark.parametrize(
    argnames="isot,stale,num_downloads",
    argvalues=[
        # After the last entry of the file, which was downloaded long ago
        ("2019-09-30T16:00:00.000", True, 2),
        # After the last entry, but the file is new
        ("2019-09-30T16:00:00.000", False, 1),
        # Before the last entry, with no entry of its own
        ("2019-09-30T15:30:00.000", True, 1),
    ],
)
def test_lookup_refresh(
    monkeypatch: pytest.MonkeyPatch,
    isot: str,
    stale: bool,
    num_downloads: int,
) -> None:
    """
    A file is downloaded again only if an image later than its last entry has
    no entry, and the file is old.
    """
    downloads = []

    def download_file(
        url: str, directory: pathlib.Path, overwrite: bool, **kwargs: object
    ) -> pathlib.Path:
        downloads.append(overwrite)
        return directory / "file.geny"

    def restore(path: pathlib.Path) -> dict:
        times = ["2019-09-30T15:10:00.000", "2019-09-30T15:50:00.000"]
        if len(downloads) > 1:
            times.append("2019-09-30T16:00:00.000")
        num = len(times)
        entries = np.rec.fromarrays(
            [
                np.array([t.encode() for t in times]),
                np.ones(num),
                np.ones(num),
                np.full(num, 10),
            ],
            names=["DATE_OBS", "XCEN", "YCEN", "REF_TYPE"],
        )
        return {"p0": entries}

    monkeypatch.setattr(
        hinode.xrt._coalign, "_listing", lambda *args: ["20190930_1503"]
    )
    monkeypatch.setattr(hinode.xrt._coalign, "_download_file", download_file)
    monkeypatch.setattr(hinode.xrt._coalign, "_restore", restore)
    monkeypatch.setattr(hinode.xrt._coalign, "_stale", lambda path: stale)

    _, _, ref_type = _lookup("aia", _seconds([isot]), pathlib.Path("cache"))

    assert downloads == [False, True][:num_downloads]
    assert ref_type[0] == (10 if num_downloads == 2 else -1)
