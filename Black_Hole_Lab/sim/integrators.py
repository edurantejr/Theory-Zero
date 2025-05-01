# sim/integrators.py

from .physics import force_phase3, evolve_metric
from .backend import xp, as_backend

def step(state: dict[str, xp.ndarray],
         particles: xp.ndarray,
         dt: float,
         dx: float) -> xp.ndarray:
    # 1) update metric
    g_new = evolve_metric(state["g"], state["S"], dt, dx)
    state["g"] = g_new

    # 2) coerce particles → correct backend
    particles = as_backend(particles)

    # 3) gradient field
    Fx, Fy, Fz = force_phase3(g_new, dx)

    # 4) sample force at each particle's cell
    idx = (particles[:, :3] / dx).astype(int)
    i = xp.clip(idx[:,0], 1, g_new.shape[0]-2)
    j = xp.clip(idx[:,1], 1, g_new.shape[1]-2)
    k = xp.clip(idx[:,2], 1, g_new.shape[2]-2)

    # 5) kick
    particles[:,3] += Fx[i,j,k] * dt
    particles[:,4] += Fy[i,j,k] * dt
    particles[:,5] += Fz[i,j,k] * dt

    # 6) drift
    particles[:,:3] += particles[:,3:6] * dt

    return particles
