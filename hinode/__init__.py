"""
Download and analyze observations from the Hinode satellite.
"""

import pathlib
import joblib

__all__ = [
    "directory_default",
    "memory",
    "xrt",
]

directory_default = pathlib.Path.home() / ".hinode/cache"
"""The default directory for downloaded files."""

memory = joblib.Memory(location=directory_default / "joblib", verbose=0)
"""
A representation of the cache which stores intermediate results,
such as the headers of the files in the archives.

The cache has a directory of its own, apart from the downloaded files,
so that :meth:`joblib.Memory.clear` does not delete them.
Assign another :class:`joblib.Memory` to this attribute to move the cache,
or one whose location is :obj:`None` to stop caching.
"""

# Imported after the attributes above, which the subpackages use
from . import xrt  # noqa: E402
