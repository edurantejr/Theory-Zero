from .backend import xp, as_backend, asnumpy

def kick_particles(particles, g, dt, dx):
     """…your existing Numba/CuPy kernels…"""
     # make sure you convert index arrays to ints
     i = (particles[:, 0] / dx).astype(int)
     j = (particles[:, 1] / dx).astype(int)
     k = (particles[:, 2] / dx).astype(int)
     Fx, Fy, Fz = force_phase3(g, dx)
     particles[:, 3] += dt * Fx[i, j, k]
     particles[:, 4] += dt * Fy[i, j, k]
     particles[:, 5] += dt * Fz[i, j, k]
     return particles

def step(state: dict[str, xp.ndarray], dt, dx):
     """
     One explicit‐Euler update of both the metric and the
     N×6 particle array in `state`.
     """
     # 1) evolve the metric
     state["g"] = evolve_metric(
         state["g"], state["S"], dt, dx
     )
     # 2) evolve the particles on CPU arrays
     particles_np = asnumpy(state["particles"])
     state["particles"] = kick_particles(
         particles_np, state["g"], dt, dx
     )
     return state
