"""
Low-level Phase-3 physics:
  • ricci_tensor   – ∇∇S → (R00, Rii)
  • force_phase3   – ∂R00/∂x → (Fx,Fy,Fz)
  • evolve_metric  – gₙ₊₁ = gₙ + dt*(R00 – damping*gₙ)
"""

from __future__ import annotations
from .backend import xp, as_backend
import numpy as _np  # for type hints only


def ricci_tensor(
    S: _np.ndarray | xp.ndarray,
    dx: float = 1.0,
    h:  float | None = None,      # legacy tests pass `h=` instead of `dx=`
    kappa: float = -0.138475,
    gamma: float = -1.0,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Returns (R00, Rii) on the interior grid (shape (L-2, L-2, L-2)).
    Accepts either NumPy or CuPy arrays (auto-converted).
    Legacy: if `h` is provided, we treat it as `dx`.
    """
    if h is not None:
        dx = h

    S = as_backend(S)
    inv_dx2 = 1.0 / dx**2

    lap = (
        S[:-2,1:-1,1:-1] + S[2:,1:-1,1:-1] +
        S[1:-1,:-2,1:-1] + S[1:-1,2:,1:-1] +
        S[1:-1,1:-1,:-2] + S[1:-1,1:-1,2:] -
        6.0 * S[1:-1,1:-1,1:-1]
    ) * inv_dx2

    dxx = (S[2:,1:-1,1:-1] - 2*S[1:-1,1:-1,1:-1] + S[:-2,1:-1,1:-1]) * inv_dx2
    dyy = (S[1:-1,2:,1:-1] - 2*S[1:-1,1:-1,1:-1] + S[1:-1,:-2,1:-1]) * inv_dx2
    dzz = (S[1:-1,1:-1,2:] - 2*S[1:-1,1:-1,1:-1] + S[1:-1,1:-1,:-2]) * inv_dx2

    R00 = kappa * (dxx + dyy + dzz + gamma * lap)
    Rii = kappa * ((gamma - 1.0) * xp.stack([dxx, dyy, dzz])).sum(axis=0)

    return R00, Rii


def force_phase3(
    g00: _np.ndarray | xp.ndarray,
    dx:  float,
) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Compute the grid‐based 3-vector ∂R00/∂x on the same domain as g00.
    """
    G = as_backend(g00)
    inv_2dx = 1.0 / (2.0 * dx)

    Fx = xp.zeros_like(G)
    Fy = xp.zeros_like(G)
    Fz = xp.zeros_like(G)

    Fx[1:-1, :, :] = (G[2:,   :, :] - G[:-2,   :, :]) * inv_2dx
    Fy[:, 1:-1, :] = (G[:,  2:, :] - G[:, :-2, :]) * inv_2dx
    Fz[:, :, 1:-1] = (G[:, :,  2:] - G[:, :, :-2]) * inv_2dx

    return Fx, Fy, Fz


def evolve_metric(
    g00: _np.ndarray | xp.ndarray,
    S:   _np.ndarray | xp.ndarray,
    dt:  float,
    dx:  float,
    kappa:  float = -0.138475,
    gamma:  float = -1.0,
    damping: float = 0.1,
) -> xp.ndarray:
    """
    Explicit Euler update of the g00 lattice:
       gⁿ⁺¹ = gⁿ + dt*(R00(S) - damping*gⁿ)
    """
    g   = as_backend(g00)
    Sbg = as_backend(S)

    R00, _ = ricci_tensor(Sbg, dx=dx, kappa=kappa, gamma=gamma)
    return g + dt * (R00 - damping * g)
