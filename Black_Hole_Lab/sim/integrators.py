# sim/integrators.py

from .physics import force_phase3, evolve_metric
from .backend import xp

def kick_particles(
    particles: xp.ndarray,
    g00: xp.ndarray,
    dt: float,
    dx: float
) -> None:
    """
    In-place update of particle velocities via ∂R00/∂x from g00.
    particles: shape (M,6), where columns 0–2 = x,y,z and 3–5 = vx,vy,vz
    g00: 3D array on the interior grid
    """
    Fx, Fy, Fz = force_phase3(g00, dx)

    # Convert each particle's (x,y,z) → integer grid index
    idx = (particles[:, 0:3] / dx).astype(int)

    # clamp into [1..g00.shape[0]] then shift to [0..shape-1]
    N = g00.shape[0]
    idx = xp.clip(idx, 1, N) - 1

    # apply force*dt to the velocities (in-place)
    particles[:, 3] += Fx[idx[:,0], idx[:,1], idx[:,2]] * dt
    particles[:, 4] += Fy[idx[:,0], idx[:,1], idx[:,2]] * dt
    particles[:, 5] += Fz[idx[:,0], idx[:,1], idx[:,2]] * dt


def step(
    state: dict[str, xp.ndarray],
    particles: xp.ndarray,
    dt: float,
    dx: float
) -> xp.ndarray:
    """
    One integrator step:
      1) evolve the metric g00 field
      2) kick particle velocities from ∂R00/∂x
      3) drift particle positions by vx*dt,vy*dt,vz*dt

    state: {"S": S_field, "g": g00_field}
    particles: shape (M,6) array
    """
    # 1) metric update
    state["g"] = evolve_metric(state["g"], state["S"], dt, dx)

    # 2) velocity kick
    kick_particles(particles, state["g"], dt, dx)

    # 3) drift positions
    particles[:, 0:3] += particles[:, 3:6] * dt

    return particles
