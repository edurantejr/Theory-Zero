# sim/integrators.py

from .physics import force_phase3, evolve_metric
from .backend import xp

def kick_particles(
    particles: xp.ndarray,
    g00:       xp.ndarray,
    dt:        float,
    dx:        float,
) -> None:
    """
    In-place velocity kick: v ← v + dt * ∂R00/∂x at each particle's grid cell.
    """
    Fx, Fy, Fz = force_phase3(g00, dx)

    # grid indices
    idx = (particles[:,0:3] / dx).astype(int)
    N = g00.shape[0]
    idx = xp.clip(idx, 1, N) - 1  # clamp + shift into [0..N-1]

    # apply to vx,vy,vz
    particles[:,3] += Fx[idx[:,0], idx[:,1], idx[:,2]] * dt
    particles[:,4] += Fy[idx[:,0], idx[:,1], idx[:,2]] * dt
    particles[:,5] += Fz[idx[:,0], idx[:,1], idx[:,2]] * dt


def step(
    state:     dict[str, xp.ndarray],
    particles: xp.ndarray,
    dt:        float,
    dx:        float,
) -> xp.ndarray:
    """
    One full step:
      1) evolve metric
      2) kick velocities
      3) drift positions
    """
    # 1) metric update
    state["g"] = evolve_metric(state["g"], state["S"], dt, dx)

    # 2) kick
    kick_particles(particles, state["g"], dt, dx)

    # 3) drift
    particles[:,0:3] += particles[:,3:6] * dt

    return particles
