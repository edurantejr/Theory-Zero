"""
Integrator utilities: update metric + particle positions one time-step.
Designed to be minimal so it can be Numba- or CuPy-accelerated.
"""
from __future__ import annotations

import numpy as np
from math import ceil

from .physics  import ricci_tensor, force_phase3
from .metric   import evolve_metric
from .backend  import xp, as_backend

# -----------------------------------------------------------------------------

def kick_particles(p: xp.ndarray,
                   g00: xp.ndarray,
                   dt: float,
                   dx: float):
    """
    Velocity-Verlet “kick” – update velocities and positions in-place.
    p is (N,6):  [x,y,z, vx,vy,vz]
    """
    Fx, Fy, Fz = force_phase3(g00, dx)

    Lx = g00.shape[0]
    for n in range(p.shape[0]):
        # integer voxel indices – clip to avoid OOB
        ix = int(xp.clip(p[n, 0] / dx, 0, Lx - 2))
        iy = int(xp.clip(p[n, 1] / dx, 0, Lx - 2))
        iz = int(xp.clip(p[n, 2] / dx, 0, Lx - 2))

        ax = Fx[ix, iy, iz]
        ay = Fy[ix, iy, iz]
        az = Fz[ix, iy, iz]

        p[n, 3] += ax * dt
        p[n, 4] += ay * dt
        p[n, 5] += az * dt

        p[n, 0:3] += p[n, 3:6] * dt


def step(state: dict[str, xp.ndarray],
         particles: xp.ndarray,
         dt: float,
         dx: float):
    """
    Single integrator step:
        • evolve the metric g00 = state["g"]
        • kick / drift particles
    """
    # --- metric update --------------------------------------------------------
    state["g"] = evolve_metric(state["g"], state["S"], dt, dx)

    # --- particle update ------------------------------------------------------
    kick_particles(particles, state["g"], dt, dx)

# -----------------------------------------------------------------------------
# helper to choose a stable dt  (optional, used in run_phase3.py)
def safe_dt(g00: xp.ndarray, R00: xp.ndarray, dx: float, cfl: float = 0.4):
    """
    Courant-style limit for explicit Euler:  dt < cfl * dx / max|R00|
    """
    max_delta = xp.abs(R00).max()
    if max_delta == 0:
        return 1e10
    return cfl * dx / float(max_delta)
