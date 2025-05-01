"""
Backend utilities: transparently handle NumPy (CPU) vs CuPy (GPU).

If CuPy is importable **and** a CUDA device exists, we treat CuPy as the
“native” array type; otherwise we fall back to NumPy everywhere.

Code elsewhere should:
    from .backend import xp, as_backend
and then work with ``xp.ndarray`` just like NumPy.
"""
from __future__ import annotations

import os
import numpy as _np

try:
    import cupy as _cp
    _GPU_OK = _cp.is_available()
except ImportError:     # CuPy not installed → CPU only
    _cp = None
    _GPU_OK = False

# --- public aliases ----------------------------------------------------------
xp = _cp if _GPU_OK else _np            # “numpy-like” backend

def as_backend(arr):
    """
    Convert *arr* to the currently-selected backend (NumPy or CuPy).

    • If arr is already an xp.ndarray, it is returned unchanged.  
    • Scalars / lists become 0-D / 1-D xp arrays.
    """
    if _GPU_OK and isinstance(arr, _np.ndarray):
        return _cp.asarray(arr)
    if (not _GPU_OK) and _cp is not None and isinstance(arr, _cp.ndarray):
        return _np.asarray(arr)
    if _GPU_OK and not isinstance(arr, _cp.ndarray):
        return _cp.asarray(arr)
    if (not _GPU_OK) and not isinstance(arr, _np.ndarray):
        return _np.asarray(arr)
    return arr  # already correct type

def backend_name() -> str:          # convenience for logging
    return "cupy/GPU" if _GPU_OK else "numpy/CPU"
