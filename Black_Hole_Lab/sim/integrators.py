# sim/integrators.py
from __future__ import annotations

import numpy as _np
from .backend import xp
from .physics import evolve_metric, force_phase3

def step(
    state: dict[str, _np.ndarray | xp.ndarray],
    particles: _np.ndarray,
    dt: float,
    dx: float,
) -> None:
    """
    Evolve one timestep of metric and particles in place.
    state must have keys "S" (entropy field) and "g" (current g00 lattice).
    particles is an N×6 array: [x, y, z, vx, vy, vz].
    """
    # 1) update the metric
    state["g"] = evolve_metric(state["g"], state["S"], dt, dx)

    # 2) push the particles
    kick_particles(particles, state["g"], dt, dx)


def kick_particles(
    particles: _np.ndarray,
    g00: xp.ndarray,
    dt: float,
    dx: float,
) -> None:
    """
    Update velocities & positions of each particle in place,
    using the local ∇R00 force lookup at the nearest grid‐index.
    """
    # compute ∂R00/∂x on the lattice
    Fx, Fy, Fz = force_phase3(g00, dx)

    # lattice shape & world‐to‐grid converter
    Lx, Ly, Lz = g00.shape
    halfx = Lx // 2
    halfy = Ly // 2
    halfz = Lz // 2

    for n in range(particles.shape[0]):
        x, y, z, vx, vy, vz = particles[n]

        # map world coords → interior grid indices [1 .. L-2]
        i = int(_np.clip(round(x/dx) + halfx, 1, Lx-2))
        j = int(_np.clip(round(y/dx) + halfy, 1, Ly-2))
        k = int(_np.clip(round(z/dx) + halfz, 1, Lz-2))

        # update velocity by local force
        vx += Fx[i, j, k] * dt
        vy += Fy[i, j, k] * dt
        vz += Fz[i, j, k] * dt

        # simple Euler step for position
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # write back
        particles[n, 0:3] = (x, y, z)
        particles[n, 3:6] = (vx, vy, vz)
