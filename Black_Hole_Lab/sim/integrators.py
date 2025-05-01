# sim/integrators.py

"""
Phase-3 integrators: evolve the metric and move particles under ∂R00/∂x.
"""

from .physics     import force_phase3, evolve_metric
from .backend     import xp, as_backend

def step(state: dict[str, xp.ndarray],
         particles: xp.ndarray,
         dt: float,
         dx: float) -> xp.ndarray:
    """
    1) Evolve the metric in-place via explicit Euler:
         g → g + dt * ( R00(S) − damping * g )
    2) Compute ∂R00/∂x = (Fx, Fy, Fz)
    3) Look up the force at each particle’s lattice index,
       update velocity and then position.

    Args:
        state:    {"S": S_field (L³), "g": g00_field ((L−2)³)}
        particles: (N,6) array [x,y,z,vx,vy,vz] in lattice units
        dt:       timestep
        dx:       lattice spacing

    Returns:
        Updated (N,6) particle array on the same backend.
    """
    # ————————————————
    # 1) Evolve metric
    g_new = evolve_metric(state["g"], state["S"], dt, dx)
    state["g"] = g_new

    # ————————————————
    # 2) Ensure particles live on the same backend
    particles = as_backend(particles)

    # ————————————————
    # 3) Get force field ∂R00/∂x
    Fx, Fy, Fz = force_phase3(g_new, dx)

    # ————————————————
    # 4) Turn continuous positions → integer lattice indices,
    #    clamp so we never hit the boundary
    idx = (particles[:, :3] / dx).astype(int)
    i   = xp.clip(idx[:, 0], 1, g_new.shape[0] - 2)
    j   = xp.clip(idx[:, 1], 1, g_new.shape[1] - 2)
    k   = xp.clip(idx[:, 2], 1, g_new.shape[2] - 2)

    # ————————————————
    # 5) Kick velocities: v += F * dt
    particles[:, 3] += Fx[i, j, k] * dt
    particles[:, 4] += Fy[i, j, k] * dt
    particles[:, 5] += Fz[i, j, k] * dt

    # ————————————————
    # 6) Drift positions: x += v * dt
    particles[:, :3] += particles[:, 3:6] * dt

    return particles
