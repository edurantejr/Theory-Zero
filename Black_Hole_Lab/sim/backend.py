# sim/backend.py

import numpy as _np
try:
    import cupy as _cp
except ImportError:
    _cp = None

# Choose our array library
xp = _cp if _cp is not None else _np

def as_backend(x):
    """
    Convert a NumPy array to CuPy if xp is CuPy,
    or a CuPy array to NumPy if xp is NumPy.
    Leave everything else alone.
    """
    if xp is _cp and isinstance(x, _np.ndarray):
        return _cp.asarray(x)
    if xp is _np and hasattr(x, "__cuda_array_interface__"):
        return _np.asarray(x)
    return x

def asnumpy(x):
    """Always return a NumPy ndarray."""
    if xp is _cp:
        return _cp.asnumpy(x)
    return x
