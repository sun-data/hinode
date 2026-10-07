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
    num_retry: int = 5,
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
    num_retry
        The number of times to try to connect to the server.
    """
    return Filtergram.from_time_range(
        time_start=time_start,
        time_stop=time_stop,
        filter=filter,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        num_retry=num_retry,
    )
