import os
import re
import time
import typing
import pathlib
import datetime
import tempfile
import concurrent.futures
import requests
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import named_arrays as na
import hinode

__all__ = [
    "urls",
    "download",
]

_url_level_1 = "https://umbra.nascom.nasa.gov/hinode/xrt/level1/"
"""
The archive of Level 1 XRT images at the Solar Data Analysis Center,
with one directory for each hour.
"""

_pattern_file = re.compile(r'href="(L1_XRT(\d{8})_(\d{6}\.\d)\.fits)"')
"""
The name of a Level 1 file in a directory listing of the archive,
which holds the date and the start time of the exposure,
as in ``L1_XRT20190930_180837.5.fits``.
"""

_size_block = 2880
"""The number of bytes in a block of a FITS file."""

_size_card = 80
"""The number of bytes in a card of a FITS header."""

_num_workers = 8
"""
The number of requests made to the archive at once,
since most of the time of each one is spent waiting for the server.
"""

_delay_retry = 1
"""
The number of seconds to wait before trying a request again,
which is doubled before each further try.
"""

_signature_fits = b"SIMPLE  ="
"""The first bytes of every FITS file."""


def _get(
    url: str,
    num_retry: int = 5,
    headers: None | dict[str, str] = None,
) -> requests.Response:
    """
    Get a URL, trying again if the connection fails, the server is busy or
    has an error, or the response is cut short.

    Each try waits twice as long as the one before it,
    starting from :data:`_delay_retry`.

    Parameters
    ----------
    url
        The URL to get.
    num_retry
        The number of times to try to connect to the server.
    headers
        Additional HTTP headers to send with the request.

    Raises
    ------
    FileNotFoundError
        If the server does not have the URL.
    requests.exceptions.HTTPError
        If the server refuses the request for a reason which trying again
        cannot change, such as a 403 status.
    ConnectionError
        If no attempt succeeded.
    """
    error = None
    for i in range(num_retry):
        if i > 0:
            time.sleep(_delay_retry * 2 ** (i - 1))
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = None if e.response is None else e.response.status_code
            if status == 404:
                raise FileNotFoundError(url) from e
            if status is not None and status < 500 and status != 429:
                raise
            error = e
            continue
        except requests.exceptions.RequestException as e:  # pragma: no cover
            error = e
            continue

        # The archive sometimes ends a response early,
        # which is only noticed by comparing its length to the one promised.
        length = response.headers.get("Content-Length")
        if "Content-Encoding" not in response.headers and length is not None:
            if int(length) != len(response.content):  # pragma: no cover
                error = ConnectionError(
                    f"Received {len(response.content)} of {length} bytes of {url}."
                )
                continue

        return response

    raise ConnectionError(f"Could not get {url}.") from error  # pragma: no cover


def _hours(
    start: str | astropy.time.Time,
    stop: str | astropy.time.Time,
) -> list[datetime.datetime]:
    """
    The hours of the directories of the archive which hold the files that
    began during a time range.

    The directories are named in UTC,
    and the hours are counted as :class:`datetime.datetime` objects,
    which, unlike UTC times, have no leap seconds,
    so an hour with a leap second at its end is not counted twice.

    Parameters
    ----------
    start
        The start of the time range.
    stop
        The end of the time range, which is not included.
    """
    format_hour = "%Y-%m-%dT%H"

    def hour_utc(time: str | astropy.time.Time) -> datetime.datetime:
        text = str(astropy.time.Time(time, scale="utc").strftime(format_hour))
        return datetime.datetime.strptime(text, format_hour)

    hour = hour_utc(start)
    last = hour_utc(stop)

    result = []
    while hour <= last:
        result.append(hour)
        hour += datetime.timedelta(hours=1)

    return result


def _files(
    hour: datetime.datetime,
    num_retry: int = 5,
) -> list[tuple[str, str]]:
    """
    The URL and the start time, from the name, of every Level 1 file in the
    directory of the archive which holds a given hour.

    The start time is in UTC, in the ISOT format.

    Parameters
    ----------
    hour
        The hour, in UTC.
    num_retry
        The number of times to try to connect to the server.
    """
    url = f"{_url_level_1}{hour.strftime('%Y/%m/%d/H%H00')}/"

    try:
        response = _get(url, num_retry)
    except FileNotFoundError:
        return []

    result = []
    for name, date, clock in dict.fromkeys(_pattern_file.findall(response.text)):
        isot = f"{date[:4]}-{date[4:6]}-{date[6:]}T{clock[:2]}:{clock[2:4]}:{clock[4:]}"
        result.append((url + name, isot))

    return result


def _header_string(
    url: str,
    num_retry: int = 5,
) -> str:
    """
    The text of the primary header of a FITS file on a server,
    read from the start of the file without downloading its data.

    Parameters
    ----------
    url
        The URL of the FITS file.
    num_retry
        The number of times to try to connect to the server.
    """
    num_blocks = 8

    while True:
        size = num_blocks * _size_block
        response = _get(
            url=url,
            num_retry=num_retry,
            headers={"Range": f"bytes=0-{size - 1}"},
        )
        content = response.content

        for start in range(0, len(content) - _size_card + 1, _size_card):
            card = content[start : start + _size_card]
            if card.rstrip() == b"END":
                return content[: start + _size_card].decode("ascii")

        if len(content) < size:
            raise ValueError(f"The header of {url} has no END card.")

        num_blocks *= 2


_header_string_cached = hinode.memory.cache(_header_string, ignore=["num_retry"])
"""
:func:`_header_string`, cached in :data:`hinode.memory`,
since the archive does not change.
"""


def _header(
    url: str,
    num_retry: int = 5,
) -> astropy.io.fits.Header:
    """
    The primary header of a FITS file on a server,
    cached in :data:`hinode.memory`, since the archive does not change.

    Parameters
    ----------
    url
        The URL of the FITS file.
    num_retry
        The number of times to try to connect to the server.
    """
    text = typing.cast(str, _header_string_cached(url, num_retry))
    return astropy.io.fits.Header.fromstring(text)


def _filter(header: astropy.io.fits.Header) -> str:
    """
    The filters an image was taken through,
    named as in :func:`urls`.

    Parameters
    ----------
    header
        The primary header of an XRT file.
    """
    names = [str(header[key]).strip() for key in ["EC_FW1_", "EC_FW2_"]]
    names = [name for name in names if name != "Open"]
    if not names:
        return "Open"
    return "/".join(names)


def urls(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    filter: str = "Al_poly",
    axis_time: str = "time",
    num_retry: int = 5,
) -> na.ScalarArray:
    """
    Find the URLs of the Level 1 XRT images in the archive of the Solar Data
    Analysis Center which began during a given time range and were taken
    through a given filter.

    The archive does not say which filter each image was taken through,
    so the start of each file is downloaded to read its header,
    and the headers are cached in :data:`hinode.memory`.
    Only the images whose type, ``EC_IMTY_``, is ``normal`` are found,
    which leaves out the dark frames.

    Parameters
    ----------
    time_start
        The earliest start time of the images.
    time_stop
        The time before which an image must begin.
    filter
        The filters in the two filter wheels of XRT,
        named as in the ``EC_FW1_`` and ``EC_FW2_`` header keywords,
        with the open positions left out and the filters joined by a slash.
        For example ``"Al_poly"``, ``"Ti_poly"``, ``"Al_poly/Ti_poly"``,
        ``"Gband"``, or ``"Open"`` if both wheels are open.
    axis_time
        The logical axis corresponding to changes in time.
    num_retry
        The number of times to try to connect to the server.

    Examples
    --------

    Find the Al_poly images captured while the EUV Snapshot Imaging
    Spectrograph (ESIS) was observing the Sun on 2019 September 30.

    .. jupyter-execute::

        import hinode

        hinode.xrt.urls(
            time_start="2019-09-30T18:06:11",
            time_stop="2019-09-30T18:11:01",
            filter="Al_poly",
        )
    """
    start = astropy.time.Time(time_start)
    stop = astropy.time.Time(time_stop)

    result: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(_num_workers) as executor:

        listings = executor.map(
            lambda hour: _files(hour, num_retry),
            _hours(time_start, time_stop),
        )
        files = dict(file for listing in listings for file in listing)

        # The name holds the start time cut to a tenth of a second,
        # so it can be a little earlier than the start time in the header.
        candidates = []
        if files:
            time_name = astropy.time.Time(list(files.values()), scale="utc")
            margin = astropy.time.TimeDelta(1 * u.s)
            where = (start - margin <= time_name) & (time_name < stop)
            candidates = [url for url, w in zip(files, where) if w]

        # The first header is read before the others, so that the cache
        # records the code of the function it caches from one thread,
        # rather than from every thread at once.
        headers = [_header(url, num_retry) for url in candidates[:1]]
        headers += executor.map(lambda url: _header(url, num_retry), candidates[1:])

    selected = [
        (url, header["DATE_OBS"])
        for url, header in zip(candidates, headers)
        if _filter(header) == filter and header.get("EC_IMTY_") == "normal"
    ]

    if selected:
        urls_selected, dates = zip(*selected)
        time_header = astropy.time.Time(list(dates), scale="utc")
        where = (start <= time_header) & (time_header < stop)
        order = np.argsort(np.asarray(time_header.jd))
        result = [urls_selected[i] for i in order if where[i]]

    return na.ScalarArray(np.array(result, dtype=str), axes=axis_time)


def download(
    urls: na.AbstractScalarArray,
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
    num_retry: int = 5,
) -> na.ScalarArray:
    """
    Download the given URLs to a directory,
    unless they have been downloaded already.

    The files are placed under `directory` with the same paths as on the
    server, and each one is checked against the length the server promised,
    since the archive sometimes ends a download early,
    and checked to be a FITS file.

    Parameters
    ----------
    urls
        The URLs to download.
    directory
        The directory to place the downloaded files in.
        If :obj:`None` (the default), :data:`hinode.directory_default` is used.
    overwrite
        Boolean flag controlling whether to download files which are already
        in `directory`.
    num_retry
        The number of times to try to connect to the server.

    Examples
    --------

    Download one of the Al_poly images captured while ESIS was observing
    the Sun.

    .. jupyter-execute::

        import hinode

        urls = hinode.xrt.urls(
            time_start="2019-09-30T18:08:30",
            time_stop="2019-09-30T18:08:40",
        )

        hinode.xrt.download(urls)
    """
    if directory is None:
        directory = hinode.directory_default

    ndarray = np.asarray(urls.ndarray)

    def get(url: str) -> str:
        path = directory / "/".join(url.split("/")[3:])

        if overwrite or not path.exists():
            content = _get(url, num_retry).content

            if not content.startswith(_signature_fits):
                raise ValueError(
                    f"{url} is not a FITS file, it starts with {content[:40]!r}."
                )

            path.parent.mkdir(parents=True, exist_ok=True)

            # Written next to its final place, under a name no other download
            # uses, and then moved,
            # so an interrupted download never looks finished.
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f"{path.name}.",
                suffix=".part",
                delete=False,
            ) as file:
                file.write(content)
            try:
                os.replace(file.name, path)
            except PermissionError:  # pragma: no cover
                # Windows does not replace a file which another process has
                # open, as it can if that process downloaded the file first.
                os.remove(file.name)
                if not path.exists():
                    raise

        return str(path)

    # Each URL is downloaded once, even if it is given more than once.
    unique = list(dict.fromkeys(str(url) for url in ndarray.flat))

    with concurrent.futures.ThreadPoolExecutor(_num_workers) as executor:
        paths = dict(zip(unique, executor.map(get, unique)))

    paths = [paths[str(url)] for url in ndarray.flat]
    paths = np.array(paths, dtype=str).reshape(ndarray.shape)

    return na.ScalarArray(paths, axes=urls.axes)
