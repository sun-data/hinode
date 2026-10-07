import pathlib

__all__ = [
    "directory_default",
]


directory_default = pathlib.Path.home() / ".hinode/cache"
"""The default directory for downloaded files."""
