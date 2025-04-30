# ── sim/integrators.py ─────────────────────────────────────────────────────
import numpy as np
import cupy  as cp
from sim.backend import xp                     # NumPy ↔ CuPy switch
from sim.physics import force_phase3

# --------------------------------------------------------------------------
@xp.fuse()                                    # fused kernel for CuPy
def kick_particles(p: xp.ndarray, g00: xp.ndarray,
                   dt: float, dx: float):
    """
    Velocity-Verlet kick + drift for all particles in one pass.
    `p` shape (N,6): x,y,z,vx,vy,vz
    """
    for n in range(p.shape[0]):
        # --- read current state -------------------------------------------
        x, y, z, vx, vy, vz = p[n]

        # gravitational “force”
        Fx, Fy, Fz = force_phase3(g00, xp.array([x, y, z], xp.float32), dx)

        # --- kick + drift --------------------------------------------------
        vx = vx + Fx * dt
        vy = vy + Fy * dt
        vz = vz + Fz * dt

        x  = x + vx * dt
        y  = y + vy * dt
        z  = z + vz * dt

        # write back
        p[n] = (x, y, z, vx, vy, vz)

# ------------------------------------------------------------------
def step(state: dict[str, np.ndarray],
         particles: np.ndarray,
         dt: float,
         dx: float):
    """Advance metric then particles by one Euler step."""
    state["g"] = evolve_metric(state["g"], state["S"], dt, dx)
    kick_particles(particles, state["g"], dt, dx)
