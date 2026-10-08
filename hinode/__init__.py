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

memory = joblib.Memory(location=directory_default, verbose=0)
"""A representation of the cache which stores intermediate results."""

# Imported after the attributes above, which the subpackages use
from . import xrt  # noqa: E402
