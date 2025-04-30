"""
Backend selector: NumPy by default, CuPy when --gpu is on **and** cupy is
installed with a compatible CUDA runtime.

Usage in your code:
    import sim.backend as xp
    a = xp.zeros((100,100))
    b = xp.exp(a) + 1

Running:
    python -m sim.run_phase3 ...               # CPU / NumPy
    python -m sim.run_phase3 --gpu ...         # GPU / CuPy (if available)
"""
import importlib
import os
import sys

_use_gpu = "--gpu" in sys.argv
if _use_gpu:
    sys.argv.remove("--gpu")

if _use_gpu:
    try:
        xp = importlib.import_module("cupy")   # GPU!
    except ModuleNotFoundError:
        print("⚠️  CuPy not installed – falling back to NumPy")
        xp = importlib.import_module("numpy")
else:
    xp = importlib.import_module("numpy")

# re-export the backend as this module’s public API
globals().update(xp.__dict__)
__all__ = xp.__dict__.keys()
