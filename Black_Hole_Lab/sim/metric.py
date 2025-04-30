import numpy as np
from numba import njit, prange

@njit(fastmath=True, parallel=False)        # ← parallel off = simpler
def evolve_metric(g: np.ndarray,
                  R00: np.ndarray,
                  dt: float,
                  damping: float = 0.1):
    """
    Relaxation: ∂t g = −damping * R00 .
    Shapes must match: g.shape == R00.shape  (here = (L-2)³).
    """
    n = g.size
    for idx in range(n):                    # scalar loop → auto-vectorised
        g.flat[idx] += -damping * R00.flat[idx] * dt
