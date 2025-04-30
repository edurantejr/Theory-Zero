import numpy as np
from sim.physics import evolve_metric, force_phase3

# =============================================================================
#  Leap‑frog / Velocity‑Verlet integrator
#  -------------------------------------
#  * half‑kick → full drift → recalc force → half‑kick
#  * second‑order in time, symplectic, far better energy behaviour than Euler
#  * API identical to the old `step(state, particles, dt, dx)` so existing
#    drivers / tests need no changes.
# =============================================================================

def _sample_force(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray,
                  pos: np.ndarray,
                  L:   int,
                  dx:  float) -> np.ndarray:
    """Nearest‑neighbour force lookup for a single particle position."""
    i = int(np.clip(round(pos[0] + L/2 - 1), 0, L-3))
    j = int(np.clip(round(pos[1] + L/2 - 1), 0, L-3))
    k = int(np.clip(round(pos[2] + L/2 - 1), 0, L-3))
    return np.array([Fx[i,j,k], Fy[i,j,k], Fz[i,j,k]], np.float32)

# -----------------------------------------------------------------------------
#  The public integrator
# -----------------------------------------------------------------------------

def step(state: dict[str, np.ndarray],
         particles: np.ndarray,
         dt: float,
         dx: float):
    """Velocity‑Verlet update for metric + particles (in‑place)."""

    # 1) update metric first (explicit Euler with stability guard inside)
    state["g"] = evolve_metric(state["g"], state["S"], dt, dx)

    # 2) precompute force field on the interior grid
    Fx, Fy, Fz = force_phase3(state["g"], dx)
    L          = state["g"].shape[0] + 2               # full lattice size

    # 3) loop over particles (few thousand, plain Python fine)
    for p in particles:
        # unpack
        x, y, z, vx, vy, vz = p
        pos = np.array([x, y, z], np.float32)
        vel = np.array([vx, vy, vz], np.float32)

        # a) half‑kick
        acc  = _sample_force(Fx, Fy, Fz, pos, L, dx)
        vel += 0.5 * dt * acc

        # b) full drift
        pos += dt * vel

        # c) recompute forces at new position
        acc_new = _sample_force(Fx, Fy, Fz, pos, L, dx)

        # d) half‑kick again
        vel += 0.5 * dt * acc_new

        # write back
        p[0:3] = pos
        p[3:6] = vel
