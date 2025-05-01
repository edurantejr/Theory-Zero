# sim/physics.py

from __future__ import annotations
from .backend import xp, as_backend

def ricci_tensor(
    S:  xp.ndarray | object,
    *,
    dx: float = 1.0,
    h:  float | None = None,
    kappa: float = -0.138475,
    gamma: float = -1.0,
):
    """
    Compute (R00, Rii) on the interior grid.
    Accepts keyword `h=` for legacy tests.
    """
    if h is not None:
        dx = h

    S = as_backend(S)
    inv_dx2 = 1.0 / dx**2

    # 6-point Laplacian
    lap = (
        S[:-2,1:-1,1:-1] + S[2:,1:-1,1:-1] +
        S[1:-1,:-2,1:-1] + S[1:-1,2:,1:-1] +
        S[1:-1,1:-1,:-2] + S[1:-1,1:-1,2:]
        - 6.0 * S[1:-1,1:-1,1:-1]
    ) * inv_dx2

    dxx = (S[2:,1:-1,1:-1] - 2*S[1:-1,1:-1,1:-1] + S[:-2,1:-1,1:-1]) * inv_dx2
    dyy = (S[1:-1,2:,1:-1] - 2*S[1:-1,1:-1,1:-1] + S[1:-1,:-2,1:-1]) * inv_dx2
    dzz = (S[1:-1,1:-1,2:] - 2*S[1:-1,1:-1,1:-1] + S[1:-1,1:-1,:-2]) * inv_dx2

    R00 = kappa * (dxx + dyy + dzz + gamma * lap)
    Rii = kappa * ((gamma - 1.0) * xp.stack([dxx, dyy, dzz])).sum(axis=0)
    return R00, Rii


def force_phase3(
    g00: xp.ndarray,
    dx: float
) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Compute ∂R00/∂x,∂R00/∂y,∂R00/∂z on the interior grid.
    """
    g = as_backend(g00)
    Fx = xp.zeros_like(g)
    Fy = Fx.copy()
    Fz = Fx.copy()

    inv_2dx = 1.0 / (2.0 * dx)
    Fx[1:-1,:,:] = (g[2:,:,:] - g[:-2,:,:]) * inv_2dx
    Fy[:,1:-1,:] = (g[:,2:,:] - g[:,:-2,:]) * inv_2dx
    Fz[:,:,1:-1] = (g[:,:,2:] - g[:,:,:-2]) * inv_2dx

    return Fx, Fy, Fz


def evolve_metric(
    g00: xp.ndarray,
    S:    xp.ndarray,
    dt:   float,
    dx:   float,
    kappa: float = -0.138475,
    gamma: float = -1.0,
    damping: float = 0.1,
) -> xp.ndarray:
    """
    Explicit Euler update of g00:

      gⁿ⁺¹ = gⁿ + dt * [ R00(S) - damping * gⁿ ]
    """
    g = as_backend(g00)
    R00, _ = ricci_tensor(S, dx=dx, kappa=kappa, gamma=gamma)
    return g + dt * (R00 - damping * g)
