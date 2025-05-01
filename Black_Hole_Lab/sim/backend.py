# sim/backend.py

try:
    import cupy as _cp
except ImportError:
    _cp = None
import numpy as _np

# pick our “xp” namespace
xp = _cp if _cp is not None else _np

# expose ndarray type and common routines
ndarray = xp.ndarray
ones     = xp.ones
zeros    = xp.zeros
clip     = xp.clip
log      = xp.log
sum      = xp.sum
random   = xp.random

def as_backend(arr):
    """
    Convert any array-like to an xp.ndarray.
    """
    if isinstance(arr, xp.ndarray):
        return arr
    return xp.array(arr)

def asnumpy(arr):
    """
    Bring a GPU array back to CPU, or return NumPy array as-is.
    """
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return arr
