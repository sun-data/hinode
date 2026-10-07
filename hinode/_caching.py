import joblib
import hinode

__all__ = [
    "memory",
]

memory = joblib.Memory(location=hinode.directory_default, verbose=0)
"""A representation of the cache which stores intermediate results."""
