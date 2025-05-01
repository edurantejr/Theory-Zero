# sim/run_phase3.py

import argparse, pathlib
from .backend import xp, as_backend, asnumpy
from .integrators import step

def main():
    p = argparse.ArgumentParser(...)
    # your existing flags...
    args = p.parse_args()

    # build S, g and initial particle array in xp arrays
    S = as_backend( xp.random.rand(args.nodes, args.nodes, args.nodes, dtype=xp.float32) )
    g = xp.zeros((args.nodes-2,)*3, dtype=xp.float32)

    particles = xp.zeros((args.nodes, 6), dtype=xp.float32)
    # fill positions & velocities...

    for _ in range(args.frames):
        particles = step({"S":S, "g":g}, particles, args.dt, args.dx)

    # at the very end, if you need to save trajectories:
    traj = asnumpy(particles)
    xp.savez("phase3_traj.npz", traj=traj)

if __name__=="__main__":
    main()
