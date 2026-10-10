from typing import Sequence, Literal
import re
import time
import pathlib
import warnings
import requests
import numpy as np
import hinode
from ._data import _get, _path_local, _write, _download_file

__all__ = []

_url_coalign = "https://sohoftp.nascom.nasa.gov/sdb/hinode/xrt/xrt_msu_coalign/"
"""
The directory of the SolarSoft database which holds the co-alignment
databases of XRT and the database of its roll angle.
"""

_directories_coalign = {
    "aia": "coaldb_c",
    "ufss": "coaldb_u",
}
"""
The directory of each co-alignment database:
``coaldb_c``, from the cross-correlation of the images with AIA 335 Å,
and ``coaldb_u``, from the Ultra Fine Sun Sensors (UFSS) of Hinode.
"""

_pattern_coalign = re.compile(
    r'href="(\d{8}_\d{4})\.geny"(?:[^\n]*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}))?'
)
"""
The name of a file of a co-alignment database in a listing of its directory,
which is the start of the time it covers, as in ``20190930_1503.geny``,
and when the file was last changed, if the listing shows it on the same line.
"""

_name_listing = "listing.txt"
"""
The file in which the names of the files of a co-alignment database are kept
once its directory has been listed,
since the listing of ``coaldb_c`` is several megabytes.
"""

_signature_genx = b"SR\x00\x04"
"""
The first bytes of every IDL save file,
which is what a GENX file of SolarSoft is.
"""

_age_refresh = 86400
"""
How old, in seconds, a listing of a database or the database of the roll
must be before it is downloaded again to look for images it may not have had
yet, since the databases are updated about once a week.
"""

_age_check = 30 * 86400
"""
How old, in seconds, a listing of a database or the database of the roll
must be before it is downloaded again whatever the images,
since the server sometimes changes files it has had for years,
as it did for more than a thousand files of each database on one day in
2015 and again in 2020.
"""

_errors_network = (
    ConnectionError,
    requests.exceptions.RequestException,
)
"""
The errors of a download which mean the server could not be reached or
refused it, so that a copy downloaded before can be used instead.
"""

_tolerance = 0.1
"""
The largest difference, in seconds, between the start of the exposure of an
image and the time of an entry of a database for the entry to be that of the
image, as in ``xrt_read_coaldb.pro``.
"""

_scale_xray = 1.0286
"""The plate scale of the X-ray images, in arcseconds per unsummed pixel."""

_scale_gband = 1.0302
"""The plate scale of the G-band images, in arcseconds per unsummed pixel."""

_position_gband = 3
"""The position of the G-band filter in the second filter wheel, ``EC_FW2``."""

_epoch = np.datetime64("1979-01-01T00:00:00", "us")
"""The time from which ``anytim`` in SolarSoft counts seconds."""

_times_roll = np.array(
    [
        "2012-03-10",
        "2012-06-08",
        "2013-01-21",
        "2021-12-27T13:00",
        "2021-12-27T13:28",
        "2021-12-27T14:40",
        "2021-12-27T15:10",
    ],
    dtype="datetime64[us]",
)
"""
The times at which ``xrt_rollangle.pro`` changes from one model of the roll
to the next,
the last four of them during the attitude anomaly of 2021 December 27.
"""


def _seconds(time: Sequence[str] | np.ndarray) -> np.ndarray:
    """
    The seconds since 1979 January 1 of times in the ISOT format,
    counted without leap seconds, as ``anytim`` in SolarSoft counts them.

    Parameters
    ----------
    time
        The times, in UTC.
    """
    return (np.asarray(time, dtype="datetime64[us]") - _epoch) / np.timedelta64(1, "s")


def _start(names: Sequence[str]) -> np.ndarray:
    """
    The start of the time covered by each file of a co-alignment database,
    in seconds since 1979 January 1, from its name.

    Parameters
    ----------
    names
        The names of the files, without their extension,
        such as ``20190930_1503``.
    """
    isot = [f"{n[:4]}-{n[4:6]}-{n[6:8]}T{n[9:11]}:{n[11:13]}" for n in names]
    return _seconds(isot)


def _isot(seconds: float) -> str:
    """
    A time in seconds since 1979 January 1 in the ISOT format,
    the inverse of :func:`_seconds`.

    Parameters
    ----------
    seconds
        The time, counted as ``anytim`` counts it.
    """
    microseconds = np.timedelta64(round(seconds * 1e6), "us")
    return str(np.datetime_as_string(_epoch + microseconds, unit="s"))


def _stale(
    path: pathlib.Path,
    age: float = _age_refresh,
) -> bool:
    """
    Whether a downloaded file is old enough to be downloaded again.

    Parameters
    ----------
    path
        The file.
    age
        How old, in seconds, the file must be.
    """
    return time.time() - path.stat().st_mtime > age


def _download_database(
    url: str,
    directory: pathlib.Path,
    overwrite: bool = False,
    num_retry: int = 5,
) -> pathlib.Path:
    """
    Download a file of a database to a directory,
    unless it has been downloaded already,
    and keep the copy downloaded before, with a warning,
    if it cannot be downloaded again.

    Parameters
    ----------
    url
        The URL of the file.
    directory
        The directory to place the file in,
        under the same path as on the server.
    overwrite
        Whether to download the file again if it is already in `directory`.
    num_retry
        The number of times to try to connect to the server.
    """
    path = _path_local(url, directory)
    try:
        return _download_file(
            url=url,
            directory=directory,
            overwrite=overwrite,
            num_retry=num_retry,
            signature=_signature_genx,
            kind="an IDL save file",
        )
    except _errors_network as e:
        if not path.is_file():
            raise
        warnings.warn(
            f"Could not download {url} again, "
            f"so the copy downloaded before is used: {e!r}",
            stacklevel=2,
        )
        return path


def _restore(path: pathlib.Path) -> dict:
    """
    The variables of an IDL save file, such as a GENX file of SolarSoft.

    Parameters
    ----------
    path
        The file.
    """
    # Imported here, since only the pointing needs it
    import scipy.io

    return scipy.io.readsav(str(path))


def _interpol(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation which extrapolates beyond the ends from the first and
    the last interval, as ``interpol`` in IDL does.

    Parameters
    ----------
    x
        Where to interpolate.
    xp
        The increasing coordinates of the samples.
    fp
        The value of each sample.
    """
    result = np.interp(x, xp, fp)
    below = x < xp[0]
    above = x > xp[-1]
    slope_below = (fp[1] - fp[0]) / (xp[1] - xp[0])
    slope_above = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    result[below] = fp[0] + slope_below * (x[below] - xp[0])
    result[above] = fp[-1] + slope_above * (x[above] - xp[-1])
    return result


def _database_roll(
    latest: float,
    directory: pathlib.Path,
    num_retry: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The database of the roll of XRT relative to SDO since the attitude anomaly
    of 2021 December 27, which ``xrt_rollangle.pro`` interpolates:
    the time of each measurement, in seconds since 1979 January 1,
    and the roll, in degrees, opposite in sign to ``CROTA2``.

    It is downloaded again if an image is later than its last measurement
    and it was downloaded more than :data:`_age_refresh` ago,
    or if it was downloaded more than :data:`_age_check` ago.

    Parameters
    ----------
    latest
        The start of the latest image which needs the roll,
        in seconds since 1979 January 1.
    directory
        The directory to place the database in.
    num_retry
        The number of times to try to connect to the server.
    """
    url = _url_coalign + "xrt_rollangle_db.geny"
    path = _path_local(url, directory)

    downloaded = path.is_file()
    if downloaded:
        variables = _restore(path)
        after = latest > variables["p0"][-1]
        if not ((after and _stale(path)) or _stale(path, _age_check)):
            return variables["p0"].astype(float), variables["p1"].astype(float)

    path = _download_database(url, directory, downloaded, num_retry)

    variables = _restore(path)
    return variables["p0"].astype(float), variables["p1"].astype(float)


def _roll(
    seconds: np.ndarray,
    directory: pathlib.Path,
    num_retry: int = 5,
) -> np.ndarray:
    """
    The roll of each image, ``CROTA2``, in degrees,
    as ``xrt_rollangle.pro`` in SolarSoft finds it:
    from fits to the roll of XRT relative to SDO before the attitude anomaly
    of 2021 December 27,
    from a model of the roll during the anomaly,
    and from the database of the roll after it, interpolated linearly.

    The models and their ranges are those of ``xrt_rollangle.pro``,
    including its gap at exactly 2012 June 8 00:00,
    when the roll is zero,
    and its linear extrapolation beyond the first and the last measurement
    of the database, which is done with a warning,
    since the measurements reach the server days after the images.

    Parameters
    ----------
    seconds
        The start of each image, in seconds since 1979 January 1,
        as ``anytim`` counts them.
    directory
        The directory to place the database of the roll in.
    num_retry
        The number of times to try to connect to the server.
    """
    t = np.asarray(seconds, dtype=float)
    t1, t2, t3, t4, t5, t6, t7 = _seconds(_times_roll)

    roll = np.zeros(t.shape)

    # Before 2012 March 10
    where = t <= t1
    roll[where] = 0.314067 + 0.0850355 * np.sin(1.99892e-07 * t[where] + 3.45231)

    # From 2012 March 10 to June 8, and from 2013 January 21 to the anomaly
    where = ((t1 < t) & (t < t2)) | ((t3 < t) & (t <= t4))
    roll[where] = 0.314457 - 0.0712843 * np.sin(1.99474e-07 * t[where] + 0.445648)

    # From 2012 June 8 to 2013 January 21, with a variation over each day
    where = (t2 < t) & (t <= t3)
    roll[where] = (
        0.314457
        - 0.0712843 * np.sin(1.99474e-07 * t[where] + 0.445648)
        + 0.00833730
        + 0.0472532 * np.sin(7.27225e-05 * t[where] + 0.661514)
    )

    # During the anomaly
    d = t - t4
    where = (t4 < t) & (t <= t5)
    roll[where] = 5.2946095e-07 * d[where] ** 2 + 0.0038027661 * d[where] + 0.26089384
    where = (t5 < t) & (t <= t6)
    roll[where] = 0.0015707323 * d[where] + 5.6717680
    where = (t6 < t) & (t <= t7)
    roll[where] = 0.0062752709 * d[where] - 22.536388

    # After the anomaly
    where = t > t7
    if np.any(where):
        times, angles = _database_roll(t[where].max(), directory, num_retry)
        roll[where] = _interpol(t[where], times, angles)
        outside = (t[where] < times[0]) | (t[where] > times[-1])
        if np.any(outside):
            warnings.warn(
                f"{np.count_nonzero(outside)} of {t.size} images began "
                f"outside the measurements of the roll, "
                f"from {_isot(times[0])} to {_isot(times[-1])}, "
                f"so their roll is extrapolated, "
                f"as xrt_rollangle.pro extrapolates it, and may be wrong.",
                stacklevel=3,
            )

    return -roll


def _listing(
    database: Literal["aia", "ufss"],
    latest: float,
    directory: pathlib.Path,
    num_retry: int = 5,
) -> dict[str, float]:
    """
    The names of the files of a co-alignment database,
    without their extension, in order of time,
    and when the listing first showed the version of each file the server
    has now, from :func:`time.time`.

    The listing of the directory is kept in :data:`_name_listing`,
    and the directory is listed again if an image is later than the start of
    the last file and it was listed more than :data:`_age_refresh` ago,
    or if it was listed more than :data:`_age_check` ago.
    A file whose time of change in the listing changes,
    such as the last file as the database grows,
    is given the time it was listed again,
    so that a copy downloaded before then is downloaded again.
    If the directory cannot be listed again,
    the listing from before is used, with a warning.

    Parameters
    ----------
    database
        The database, ``"aia"`` or ``"ufss"``.
    latest
        The start of the latest image to be looked up,
        in seconds since 1979 January 1.
    directory
        The directory to place the listing in.
    num_retry
        The number of times to try to connect to the server.

    Raises
    ------
    ValueError
        If the listing of the directory has no files of the database.
    """
    url = f"{_url_coalign}{_directories_coalign[database]}/"
    path = _path_local(url + _name_listing, directory)

    # The name of each file, when it was last changed, and when the listing
    # first showed that change
    before: dict[str, tuple[str, float]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            name, changed, listed = line.split()
            before[name] = (changed, float(listed))

    if before:
        last = _start(list(before)[-1:])[0]
        if not ((latest >= last and _stale(path)) or _stale(path, _age_check)):
            return {name: listed for name, (_, listed) in before.items()}

    try:
        text = _get(url, num_retry).text
    except _errors_network as e:
        if not before:
            raise
        warnings.warn(
            f"Could not list {url} again, so the listing from before is used: {e!r}",
            stacklevel=2,
        )
        return {name: listed for name, (_, listed) in before.items()}

    # A time of change the listing does not show is kept as "-"
    found = {n: c.replace(" ", "T") or "-" for n, c in _pattern_coalign.findall(text)}
    if not found:
        raise ValueError(f"The listing of {url} has no files of the database.")

    now = time.time()
    result: dict[str, float] = {}
    lines = []
    for name in sorted(found):
        changed = found[name]
        changed_before, listed = before.get(name, ("", now))
        if changed != changed_before:
            listed = now
        result[name] = listed
        lines.append(f"{name} {changed} {listed!r}")

    _write(path, "\n".join(lines).encode(), overwrite=True)

    return result


def _lookup(
    database: Literal["aia", "ufss"],
    seconds: np.ndarray,
    directory: pathlib.Path,
    num_retry: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The center of the field of view of each image, in arcseconds,
    and the type of its reference,
    from the entry of a co-alignment database which began within
    :data:`_tolerance` of it, as ``xrt_read_coaldb.pro`` finds it,
    or NaN and -1 for an image the database has no entry for.

    Each file of the database covers the time from its start to the start of
    the next one,
    and only the files which cover an image without an entry yet are read.
    A file is downloaded again if the copy was downloaded before the listing
    of :func:`_listing` showed the version the server has now,
    as for the last file once the database has grown.
    An image may have no entry,
    as an image the cross-correlation with AIA was not made for has none in
    ``coaldb_c``.

    Parameters
    ----------
    database
        The database, ``"aia"`` or ``"ufss"``.
    seconds
        The start of each image, in seconds since 1979 January 1.
    directory
        The directory to place the files of the database in.
    num_retry
        The number of times to try to connect to the server.
    """
    num = len(seconds)
    xcen = np.full(num, np.nan)
    ycen = np.full(num, np.nan)
    ref_type = np.full(num, -1, dtype=int)

    listed = _listing(database, float(np.max(seconds)), directory, num_retry)
    names = list(listed)
    starts = _start(names)

    # The files which cover the time within the tolerance of each image
    first = np.searchsorted(starts, seconds - _tolerance, side="right") - 1
    last = np.searchsorted(starts, seconds + _tolerance, side="right") - 1

    url_directory = f"{_url_coalign}{_directories_coalign[database]}/"

    for i in np.unique(np.concatenate([first, last])):
        images = np.flatnonzero((first <= i) & (i <= last) & (ref_type == -1))
        if i < 0 or images.size == 0:
            continue

        name = names[i]
        url = f"{url_directory}{name}.geny"
        path = _path_local(url, directory)
        overwrite = path.is_file() and path.stat().st_mtime < listed[name]
        path = _download_database(url, directory, overwrite, num_retry)

        entries = _restore(path)["p0"]
        times = _seconds([t.decode() for t in entries["DATE_OBS"]])

        index, difference = _nearest(times, seconds[images])
        match = difference <= _tolerance
        images = images[match]
        index = index[match]
        xcen[images] = entries["XCEN"][index]
        ycen[images] = entries["YCEN"][index]
        ref_type[images] = entries["REF_TYPE"][index]

    return xcen, ycen, ref_type


def _nearest(
    times: np.ndarray,
    seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The index of the entry nearest in time to each image,
    the earlier one if two are as near,
    and how far it is, in seconds.

    Parameters
    ----------
    times
        The time of each entry, in any order.
    seconds
        The start of each image.
    """
    order = np.argsort(times, kind="stable")
    ordered = times[order]
    position = np.searchsorted(ordered, seconds)
    before = np.clip(position - 1, 0, ordered.size - 1)
    after = np.clip(position, 0, ordered.size - 1)
    difference_before = np.abs(ordered[before] - seconds)
    difference_after = np.abs(ordered[after] - seconds)
    nearest = np.where(difference_after < difference_before, after, before)
    return order[nearest], np.minimum(difference_before, difference_after)


def _calibration(ref_type: np.ndarray) -> np.ndarray:
    """
    The type of calibration of the pointing of each image,
    ``CALIBRATION_TYPE`` of ``xrt_read_coaldb.pro``,
    from the type of the reference of its entry in a co-alignment database.

    Parameters
    ----------
    ref_type
        The type of the reference, ``REF_TYPE``, or -1 for no entry.
    """
    result = ref_type.copy()
    result[ref_type >= 10] = 1
    result[ref_type == 1] = 2
    result[ref_type == 5] = 4
    return result


def _check_coalign(coalign: object) -> None:
    """
    Check that a co-alignment database is one of those of
    :data:`_directories_coalign`,
    so that a wrong one is found before any image is downloaded.

    Parameters
    ----------
    coalign
        The database, which :obj:`None` is not.

    Raises
    ------
    ValueError
        If `coalign` is not ``"aia"`` or ``"ufss"``.
    """
    if coalign not in tuple(_directories_coalign):
        raise ValueError(
            f"`coalign` must be one of {list(_directories_coalign)} or None, "
            f"got {coalign!r}."
        )


def _coalign(
    time: Sequence[str],
    filter_2: Sequence[int],
    chip_sum: Sequence[int],
    num_x: Sequence[int],
    num_y: Sequence[int],
    coalign: Literal["aia", "ufss"],
    directory: None | pathlib.Path = None,
    num_retry: int = 5,
) -> tuple[list[dict[str, float | str]], np.ndarray]:
    """
    The keywords to change in the header of each Level 1 image to correct its
    pointing as ``xrt_read_coaldb.pro`` in SolarSoft corrects it,
    and the type of calibration of the pointing of each image.

    The center of the field of view of an image found in a co-alignment
    database is placed at the center of the image,
    with the plate scale of ``xrt_read_coaldb.pro``,
    and every image is given the roll of ``xrt_rollangle.pro``.
    An image the databases have no entry for keeps the center and the plate
    scale of its header.

    Parameters
    ----------
    time
        The start of the exposure of each image, ``DATE_OBS``,
        in UTC in the ISOT format.
    filter_2
        The position of the second filter wheel of each image, ``EC_FW2``.
    chip_sum
        The number of pixels of the CCD summed along each axis into each
        pixel of each image, ``CHIP_SUM``.
    num_x
        The number of columns of each image, ``NAXIS1``.
    num_y
        The number of rows of each image, ``NAXIS2``.
    coalign
        Which co-alignment database to find the center in:
        ``"aia"`` for the cross-correlation with AIA,
        or the UFSS database for an image it has no entry for,
        as ``xrt_read_coaldb.pro`` does with ``/aia_cc``,
        or ``"ufss"`` for the UFSS database alone,
        as it does by default.
    directory
        The directory to place the databases in.
        If :obj:`None` (the default), :data:`hinode.directory_default` is used.
    num_retry
        The number of times to try to connect to the server.

    Raises
    ------
    ValueError
        If `coalign` is not ``"aia"`` or ``"ufss"``.
    ConnectionError
        If a file of the databases which has not been downloaded before
        cannot be downloaded.
    """
    _check_coalign(coalign)

    root: pathlib.Path = hinode.directory_default if directory is None else directory

    num = len(time)
    seconds = _seconds(time)

    databases: list[Literal["aia", "ufss"]] = ["ufss"]
    if coalign == "aia":
        databases = ["aia", "ufss"]

    xcen = np.full(num, np.nan)
    ycen = np.full(num, np.nan)
    calibration = np.full(num, -1, dtype=int)
    try:
        for database in databases:
            missing = calibration == -1
            if not np.any(missing):
                break
            x, y, ref_type = _lookup(database, seconds[missing], root, num_retry)
            xcen[missing] = x
            ycen[missing] = y
            calibration[missing] = _calibration(ref_type)

        roll = _roll(seconds, root, num_retry)
    except _errors_network as e:
        raise ConnectionError(
            f"Could not download the co-alignment databases of SolarSoft "
            f"from {_url_coalign}. "
            f"Pass coalign=None to keep the pointing of the headers."
        ) from e

    keywords: list[dict[str, float | str]] = []
    for i in range(num):
        result: dict[str, float | str] = {
            "CROTA1": float(roll[i]),
            "CROTA2": float(roll[i]),
        }
        if calibration[i] != -1:
            scale = _scale_xray
            if filter_2[i] == _position_gband:
                scale = _scale_gband
            scale = scale * chip_sum[i]
            for axis, center, size in (
                ("1", xcen[i], num_x[i]),
                ("2", ycen[i], num_y[i]),
            ):
                result[f"CRVAL{axis}"] = float(center)
                # Integer division, as in ``xrt_read_coaldb.pro``
                result[f"CRPIX{axis}"] = size // 2 + 0.5
                result[f"CDELT{axis}"] = scale
                result[f"CUNIT{axis}"] = "arcsec"
        keywords.append(result)

    return keywords, calibration
