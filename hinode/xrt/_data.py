import re
import typing
import pathlib
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


def _get(
    url: str,
    num_retry: int = 5,
    headers: None | dict[str, str] = None,
) -> requests.Response:
    """
    Get a URL, trying again if the connection fails or the response is cut
    short.

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
    ConnectionError
        If no attempt succeeded.
    """
    error = None
    for _ in range(num_retry):
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise FileNotFoundError(url) from e
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


def _files(
    hour: astropy.time.Time,
    num_retry: int = 5,
) -> list[tuple[str, astropy.time.Time]]:
    """
    The URL and the start time, from the name, of every Level 1 file in the
    directory of the archive which holds a given hour.

    Parameters
    ----------
    hour
        A time during the hour.
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
        time = astropy.time.Time(
            f"{date[:4]}-{date[4:6]}-{date[6:]}T{clock[:2]}:{clock[2:4]}:{clock[4:]}",
            scale="utc",
        )
        result.append((url + name, time))

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
    cached = hinode.memory.cache(_header_string, ignore=["num_retry"])
    text = typing.cast(str, cached(url, num_retry))
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

    hour = astropy.time.Time(start.strftime("%Y-%m-%dT%H:00:00"), scale="utc")

    files = []
    while hour < stop:
        files += _files(hour, num_retry)
        hour = hour + 1 * u.hour

    # The name holds the start time cut to a tenth of a second,
    # so it can be a little earlier than the start time in the header.
    files = [url for url, time in files if start - 1 * u.s <= time < stop]

    with concurrent.futures.ThreadPoolExecutor(_num_workers) as executor:
        headers = list(executor.map(lambda url: _header(url, num_retry), files))

    result = []
    for url, header in zip(files, headers):

        if _filter(header) != filter:
            continue

        time = astropy.time.Time(header["DATE_OBS"], scale="utc")

        if start <= time < stop:
            result.append((time, url))

    result = [url for time, url in sorted(result, key=lambda r: r[0].jd)]

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
    since the archive sometimes ends a download early.

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
            response = _get(url, num_retry)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Written next to its final place and then moved,
            # so an interrupted download never looks finished.
            partial = path.with_name(path.name + ".part")
            partial.write_bytes(response.content)
            partial.replace(path)

        return str(path)

    with concurrent.futures.ThreadPoolExecutor(_num_workers) as executor:
        paths = list(executor.map(get, [str(url) for url in ndarray.flat]))

    paths = np.array(paths, dtype=str).reshape(ndarray.shape)

    return na.ScalarArray(paths, axes=urls.axes)
