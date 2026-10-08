import pathlib
import astropy.time
from ._filtergrams import Filtergram

__all__ = [
    "open",
]


def open(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    filter: str = "Al_poly",
    axis_time: str = "time",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
    num_retry: int = 5,
    leak: bool = False,
) -> Filtergram:
    """
    Download the Level 1 XRT images which began during a given time range
    and were taken through a given filter,
    and load them into memory as an instance of
    :class:`~hinode.xrt.Filtergram`.

    This is a shortcut for :meth:`hinode.xrt.Filtergram.from_time_range`.

    Parameters
    ----------
    time_start
        The earliest start time of the images.
    time_stop
        The time before which an image must begin.
    filter
        The filters the images were taken through,
        named as in :func:`hinode.xrt.urls`.
    axis_time
        The logical axis corresponding to changes in time.
    axis_detector_x
        The logical axis corresponding to changes in detector :math:`x`-coordinate.
    axis_detector_y
        The logical axis corresponding to changes in detector :math:`y`-coordinate.
    directory
        The directory to place the downloaded files in.
        If :obj:`None` (the default), :data:`hinode.directory_default` is used.
    overwrite
        Boolean flag controlling whether to download files which are already
        in `directory`.
    num_retry
        The number of times to try to connect to the server.
    leak
        Whether to load the visible light leaking into each pixel,
        :attr:`~hinode.xrt.Filtergram.leak`.
    """
    return Filtergram.from_time_range(
        time_start=time_start,
        time_stop=time_stop,
        filter=filter,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        directory=directory,
        overwrite=overwrite,
        num_retry=num_retry,
        leak=leak,
    )
