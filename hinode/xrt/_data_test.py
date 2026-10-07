import pathlib
import pytest
import numpy as np
import astropy.time
import astropy.io.fits
import named_arrays as na
import hinode
from hinode.xrt._data import _files, _header, _filter


def _names(urls: na.ScalarArray) -> list[str]:
    """The hour directory and the name of each file."""
    return ["/".join(str(url).split("/")[-2:]) for url in urls.ndarray]


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
    time_start: str,
    time_stop: str,
    filter: str,
    expected: list[str],
):
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


def test_files_missing_hour():
    """An hour with no directory in the archive has no files."""
    assert _files(astropy.time.Time("2000-01-01T00:00")) == []


def test_header(tmp_path: pathlib.Path):
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
def test_filter(filter_1: str, filter_2: str, expected: str):
    header = astropy.io.fits.Header()
    header["EC_FW1_"] = filter_1
    header["EC_FW2_"] = filter_2
    assert _filter(header) == expected


def test_download(tmp_path: pathlib.Path):
    urls = hinode.xrt.urls("2019-09-30T18:08:00", "2019-09-30T18:08:40")

    result = hinode.xrt.download(urls, directory=tmp_path)

    assert isinstance(result, na.ScalarArray)
    assert result.shape == urls.shape

    for url, path in zip(urls.ndarray, result.ndarray):
        path = pathlib.Path(path)
        assert path.is_relative_to(tmp_path)
        assert path.name == str(url).split("/")[~0]
        assert astropy.io.fits.getheader(path)["DATA_LEV"] == 1
        assert not path.with_name(path.name + ".part").exists()

    # The files are not downloaded again
    mtime = [pathlib.Path(path).stat().st_mtime_ns for path in result.ndarray]
    again = hinode.xrt.download(urls, directory=tmp_path)
    assert np.all(again.ndarray == result.ndarray)
    assert mtime == [pathlib.Path(path).stat().st_mtime_ns for path in again.ndarray]
