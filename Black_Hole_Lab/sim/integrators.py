import sim.backend as np   # or 'xp' if you prefer the alias
from sim.physics import force_phase3, evolve_metric

# ------------------------------------------------------------------
def kick_particles(p: np.ndarray, g00: np.ndarray, dt: float, dx: float):
    """Euler kick & drift using the interior force field."""
    Fx, Fy, Fz = force_phase3(g00, dx)
    L = g00.shape[0] + 2                         # original lattice size

    for n in range(p.shape[0]):
        x, y, z, vx, vy, vz = p[n]
        # nearest interior voxel indices
        i = int(np.clip(round(x + L/2 - 1), 0, L-3))
        j = int(np.clip(round(y + L/2 - 1), 0, L-3))
        k = int(np.clip(round(z + L/2 - 1), 0, L-3))

        vx += Fx[i, j, k] * dt
        vy += Fy[i, j, k] * dt
        vz += Fz[i, j, k] * dt

        p[n, 0:3] += dt * np.array([vx, vy, vz], np.float32)
        p[n, 3:6]  = [vx, vy, vz]

# ------------------------------------------------------------------
def step(state: dict[str, np.ndarray],
         particles: np.ndarray,
         dt: float,
         dx: float):
    """Advance metric then particles by one Euler step."""
    state["g"] = evolve_metric(state["g"], state["S"], dt, dx)
    kick_particles(particles, state["g"], dt, dx)
