"""
Unified numeric backend (NumPy or CuPy), *and* a NumPy‐like module
so that `import sim.backend as np` just works in your tests.
"""

# Attempt to use CuPy if available, otherwise fall back to NumPy
try:
    import cupy as _cp
    xp = _cp
except ImportError:
    xp = None

# Always import real NumPy under the hood
import numpy as _np

# Re-export *all* of NumPy’s public API at the top level of this module.
# That way: `import sim.backend as np` gives you everything you expect.
for _name in dir(_np):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_np, _name)

# Finalize xp so that it’s either CuPy or NumPy
if xp is None:
    xp = _np

# Make sure np.ndarray refers to the real NumPy array type
ndarray = _np.ndarray

def as_backend(arr):
    """
    Convert a NumPy ndarray into the current xp backend, or leave xp ndarrays alone.
    """
    if isinstance(arr, xp.ndarray):
        return arr
    # xp.asarray works for both NumPy and CuPy
    return xp.asarray(arr)
