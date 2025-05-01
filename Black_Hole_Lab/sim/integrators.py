# sim/integrators.py

"""
Phase-3 integrators: evolve the metric and move particles under ∂R00/∂x.
"""

from .physics import force_phase3, evolve_metric
from .backend import xp, as_backend

def step(state: dict[str, xp.ndarray],
         particles: xp.ndarray,
         dt: float,
         dx: float) -> xp.ndarray:
    """
    1) Advance the metric g₀₀ via explicit Euler:
         g → g + dt * (R00(S) - damping*g)
    2) Compute ∂R00/∂x = (Fx, Fy, Fz)
    3) At each particle’s lattice index, lookup the force and
       update velocity and position.

    Args:
        state:  {"S": entropy_field (L³), "g": g00_field (L-2)³}
        particles:  (N,6) array [[x,y,z,vx,vy,vz], …] in lattice units
        dt:      time step
        dx:      lattice spacing

    Returns:
        updated particles array of shape (N,6)
    """
    # 1) evolve the metric in state
    g_new = evolve_metric(state["g"], state["S"], dt, dx)
    state["g"] = g_new

    # 2) make sure our particle array is on the right backend
    #    (NumPy on CPU or CuPy on GPU)
    particles = as_backend(particles)

    # 3) compute the force field
    Fx, Fy, Fz = force_phase3(g_new, dx)

    # 4) compute integer lattice indices for each particle
    #    clamp into [1, L-3] so we never index the boundary
    idx = (particles[:, :3] / dx).astype(int)
    i = xp.clip(idx[:, 0], 1, g_new.shape[0] - 2)
    j = xp.clip(idx[:, 1], 1, g_new.shape[1] - 2)
    k = xp.clip(idx[:, 2], 1, g_new.shape[2] - 2)

    # 5) update velocities: v += F * dt
    particles[:, 3] += Fx[i, j, k] * dt
    particles[:, 4] += Fy[i, j, k] * dt
    particles[:, 5] += Fz[i, j, k] * dt

    # 6) update positions: x += v * dt
    particles[:, :3] += particles[:, 3:6] * dt

    return particles
