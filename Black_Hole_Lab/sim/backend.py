# sim/backend.py

import numpy as _np
try:
    import cupy as _cp
except ImportError:
    _cp = None

# pick our array library
xp = _cp if _cp is not None else _np

# “fake” numpy API so downstream code/tests that do
#   import sim.backend as np
# will see np.ndarray, np.ones, np.clip, etc.
ndarray = xp.ndarray
ones    = xp.ones
zeros   = xp.zeros
clip    = xp.clip
log     = xp.log
sum     = xp.sum

def as_backend(x):
    """
    Convert any array‐like into an xp.ndarray.
    """
    if isinstance(x, xp.ndarray):
        return x
    return xp.array(x)

def asnumpy(x):
    """
    Bring a GPU (cupy) array back to a NumPy ndarray,
    or just return a NumPy array untouched.
    """
    if xp is not _np:
        return xp.asnumpy(x)
    return x
