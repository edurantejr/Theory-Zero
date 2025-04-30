"""
Metric evolution and forces (Phase-3).
Works on either NumPy (CPU) or CuPy (GPU).
"""
from __future__ import annotations
import math
import numba as nb
import sim.backend as xp      # NumPy or CuPy
import numpy as np            # always NumPy for Numba kernels

# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(cache=True, fastmath=True)
def safe_step(g: np.ndarray,
              R00: np.ndarray,
              dt: float,
              dx: float,
              damping: float = 0.1) -> np.ndarray:
    """
    Semi-implicit forward-Euler with a simple stability limit.
    """
    max_delta = np.abs(R00).max() * dt
    if max_delta > 0.4:           # clip large steps
        dt_eff = 0.4 / np.abs(R00).max()
    else:
        dt_eff = dt

    g_new = g + dt_eff * (R00 - damping * g)
    return g_new


# ─────────────────────────────────────────────────────────────────────────────
def ricci_tensor(S: xp.ndarray,
                 dx: float = 1.0,
                 kappa: float = -0.138475,
                 gamma: float = -1.0) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Returns (R00, Rii) on the interior grid (shape (L-2)³).
    """
    lap = (
        S[:-2, 1:-1, 1:-1] + S[2:, 1:-1, 1:-1] +
        S[1:-1, :-2, 1:-1] + S[1:-1, 2:, 1:-1] +
        S[1:-1, 1:-1, :-2] + S[1:-1, 1:-1, 2:] -
        6.0 * S[1:-1, 1:-1, 1:-1]
    ) / dx**2

    dxx = (S[2:, 1:-1, 1:-1] - 2*S[1:-1, 1:-1, 1:-1] + S[:-2, 1:-1, 1:-1]) / dx**2
    dyy = (S[1:-1, 2:, 1:-1] - 2*S[1:-1, 1:-1, 1:-1] + S[1:-1, :-2, 1:-1]) / dx**2
    dzz = (S[1:-1, 1:-1, 2:] - 2*S[1:-1, 1:-1, 1:-1] + S[1:-1, 1:-1, :-2]) / dx**2

    R00 = kappa * (dxx + dyy + dzz + gamma * lap)
    Rii = kappa * (-dxx - dyy + (1 + gamma) * lap)   # spatial trace
    return R00, Rii


# ─────────────────────────────────────────────────────────────────────────────
def evolve_metric(g: xp.ndarray,
                  S: xp.ndarray,
                  dt: float,
                  dx: float,
                  *,
                  kappa: float = -0.138475,
                  gamma: float = -1.0,
                  damping: float = 0.1) -> xp.ndarray:
    """
    Single forward-Euler step for g00 using the Ricci tensor.
    Handles NumPy (CPU) and CuPy (GPU) transparently.
    """
    on_gpu = xp is not np

    # make sure NumPy views for Numba kernels
    if on_gpu:
        S_np = S.get()
        g_np = g.get()
    else:
        S_np = S
        g_np = g

    R00_np, _ = ricci_tensor(S_np, dx, kappa=kappa, gamma=gamma)
    g_new_np  = safe_step(g_np, R00_np, dt, dx, damping)

    # send back to GPU if necessary
    if on_gpu:
        return xp.asarray(g_new_np)
    else:
        return g_new_np


# ─────────────────────────────────────────────────────────────────────────────
def force_phase3(g00: xp.ndarray,
                 pos: xp.ndarray,
                 dx: float = 1.0,
                 kappa: float = -0.138475,
                 gamma: float = -1.0) -> xp.ndarray:
    """
    Simple force: F = −∇g00  (gradient on the staggered grid)
    """
    # trilinear index
    i = xp.clip(((pos[:, 0] + (g00.shape[0] // 2)) / dx).astype(int), 1, g00.shape[0]-2)
    j = xp.clip(((pos[:, 1] + (g00.shape[1] // 2)) / dx).astype(int), 1, g00.shape[1]-2)
    k = xp.clip(((pos[:, 2] + (g00.shape[2] // 2)) / dx).astype(int), 1, g00.shape[2]-2)

    gx = (g00[i+1, j, k] - g00[i-1, j, k]) / (2*dx)
    gy = (g00[i, j+1, k] - g00[i, j-1, k]) / (2*dx)
    gz = (g00[i, j, k+1] - g00[i, j, k-1]) / (2*dx)

    return -xp.stack((gx, gy, gz), axis=1)
