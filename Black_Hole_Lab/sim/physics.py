# sim/physics.py

from __future__ import annotations
from .backend import xp, as_backend

def ricci_tensor(S, *, dx=1.0, h=None, kappa=-0.138475, gamma=-1.0):
    # accept legacy `h=` too
    if h is not None:
        dx = h
    S = as_backend(S)
    inv_dx2 = 1.0 / (dx*dx)
    lap = (
        S[:-2,1:-1,1:-1] + S[2:,1:-1,1:-1] +
        S[1:-1,:-2,1:-1] + S[1:-1,2:,1:-1] +
        S[1:-1,1:-1,:-2] + S[1:-1,1:-1,2:] -
        6.0*S[1:-1,1:-1,1:-1]
    ) * inv_dx2

    dxx = (S[2:,1:-1,1:-1]  - 2*S[1:-1,1:-1,1:-1] + S[:-2,1:-1,1:-1]) * inv_dx2
    dyy = (S[1:-1,2:,1:-1]  - 2*S[1:-1,1:-1,1:-1] + S[1:-1,:-2,1:-1]) * inv_dx2
    dzz = (S[1:-1,1:-1,2:]  - 2*S[1:-1,1:-1,1:-1] + S[1:-1,1:-1,:-2]) * inv_dx2

    R00 = kappa * (dxx + dyy + dzz + gamma*lap)
    Rii = kappa * ((gamma - 1.0)*xp.stack((dxx,dyy,dzz))).sum(axis=0)
    return R00, Rii

def force_phase3(g00, dx):
    g00 = as_backend(g00)
    inv2dx = 1.0/(2*dx)
    Fx = xp.zeros_like(g00); Fy = Fx.copy(); Fz = Fx.copy()
    Fx[1:-1,:,:] = (g00[2:,:,:] - g00[:-2,:,:]) * inv2dx
    Fy[:,1:-1,:] = (g00[:,2:,:] - g00[:,:-2,:]) * inv2dx
    Fz[:,:,1:-1] = (g00[:,:,2:] - g00[:,:,:-2]) * inv2dx
    return Fx, Fy, Fz

def evolve_metric(g00, S, dt, dx, kappa=-0.138475, gamma=-1.0, damping=0.1):
    g = as_backend(g00)
    R00, _ = ricci_tensor(S, dx=dx, kappa=kappa, gamma=gamma)
    return g + dt*(R00 - damping*g)
