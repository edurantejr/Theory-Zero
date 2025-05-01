"""
Low-level Phase-3 single-file physics helpers:
    • ricci_tensor      – discrete ∇∇S based Ricci (R00, Rii)
    • force_phase3      – gradient of R00  (drives particle motion)
    • evolve_metric     – explicit Euler update of the g00 lattice
"""

from __future__ import annotations

import numpy as np
from .backend import xp, as_backend

# -----------------------------------------------------------------------------


def ricci_tensor(S: np.ndarray | xp.ndarray,
                 dx: float = 1.0,
                 kappa: float = -0.138475,
                 gamma: float = -1.0):
    """
    Returns (R00, Rii) evaluated on the interior (shape = (L-2)³).

    Notes
    -----
    • Accepts either NumPy or CuPy arrays (auto-converted to backend).  
    • We use a 6-point finite-difference Laplacian.
    """
    S = as_backend(S)
    inv_dx2 = 1.0 / dx**2

    lap = (
        S[:-2, 1:-1, 1:-1] + S[2:, 1:-1, 1:-1] +
        S[1:-1, :-2, 1:-1] + S[1:-1, 2:, 1:-1] +
        S[1:-1, 1:-1, :-2] + S[1:-1, 1:-1, 2:]
        - 6.0 * S[1:-1, 1:-1, 1:-1]
    ) * inv_dx2

    dxx = (S[2:, 1:-1, 1:-1] - 2*S[1:-1, 1:-1, 1:-1] + S[:-2, 1:-1, 1:-1]) * inv_dx2
    dyy = (S[1:-1, 2:, 1:-1] - 2*S[1:-1, 1:-1, 1:-1] + S[1:-1, :-2, 1:-1]) * inv_dx2
    dzz = (S[1:-1, 1:-1, 2:] - 2*S[1:-1, 1:-1, 1:-1] + S[1:-1, 1:-1, :-2]) * inv_dx2

    R00 = kappa * (dxx + dyy + dzz + gamma * lap)
    Rii = kappa * ((gamma - 1.0) * xp.stack([dxx, dyy, dzz])).sum(axis=0)
    return R00, Rii


def force_phase3(g00: np.ndarray | xp.ndarray, dx: float):
    """
    3-vector ∂R00/∂x.  Result has same shape as *g00*.

    Works on both CPU and GPU arrays thanks to ``as_backend``.
    """
    g00 = as_backend(g00)
    Fx = (xp.zeros_like(g00))
    Fy = Fx.copy()
    Fz = Fx.copy()

    inv_2dx = 1.0 / (2.0 * dx)

    Fx[1:-1, :, :] = (g00[2:, :, :] - g00[:-2, :, :]) * inv_2dx
    Fy[:, 1:-1, :] = (g00[:, 2:, :] - g00[:, :-2, :]) * inv_2dx
    Fz[:, :, 1:-1] = (g00[:, :, 2:] - g00[:, :, :-2]) * inv_2dx
    return Fx, Fy, Fz


def evolve_metric(g00: np.ndarray | xp.ndarray,
                  S:  np.ndarray | xp.ndarray,
                  dt: float,
                  dx: float,
                  kappa: float = -0.138475,
                  gamma: float = -1.0,
                  damping: float = 0.1):
    """
    Explicit Euler:

        gⁿ⁺¹ = gⁿ + dt * ( R00(S) - damping * gⁿ )

    Returns the new lattice (same backend as input).
    """
    g = as_backend(g00)
    R00, _ = ricci_tensor(S, dx, kappa, gamma)
    g_new = g + dt * (R00 - damping * g)
    return g_new
