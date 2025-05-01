"""
Time integrators for Phase-3:
  • step            – evolve metric & particles together
  • kick_particles  – update particles under ∇R00 force
"""

import numpy as _np
from .physics import evolve_metric, force_phase3


def step(
    state: dict[str, _np.ndarray | 'xp.ndarray'],
    particles: _np.ndarray,
    dt: float,
    dx: float,
) -> None:
    """
    Evolve one time‐step:
      1) Update the metric g in-place: state['g'] = gⁿ⁺¹
      2) Kick the particles under ∂R00/∂x
    """
    # 1) metric
    state["g"] = evolve_metric(state["g"], state["S"], dt, dx)

    # 2) particles
    kick_particles(particles, state["g"], dt, dx)


def kick_particles(
    p: _np.ndarray,
    g00: 'xp.ndarray',
    dt: float,
    dx: float,
) -> None:
    """
    Advance positions & velocities of shape-(N,6) array p:
      p[:,0:3] = (x,y,z), p[:,3:6] = (vx,vy,vz).
    """
    Fx, Fy, Fz = force_phase3(g00, dx)
    N = p.shape[0]
    Lx, Ly, Lz = g00.shape

    for n in range(N):
        x, y, z, vx, vy, vz = p[n]

        # map physical coords → grid indices (centered lattice)
        i = int(_np.clip(_np.round(x/dx + Lx/2), 0, Lx-1))
        j = int(_np.clip(_np.round(y/dx + Ly/2), 0, Ly-1))
        k = int(_np.clip(_np.round(z/dx + Lz/2), 0, Lz-1))

        # pull out a Python float (handles both np & cp scalars)
        ax = float(Fx[i, j, k])
        ay = float(Fy[i, j, k])
        az = float(Fz[i, j, k])

        # update velocity then position
        p[n, 3] += ax * dt
        p[n, 4] += ay * dt
        p[n, 5] += az * dt

        p[n, 0] += p[n, 3] * dt
        p[n, 1] += p[n, 4] * dt
        p[n, 2] += p[n, 5] * dt
