# sim/run_phase3.py

import argparse
import numpy as _np
from .backend import xp, as_backend, asnumpy
from .integrators import step

def main():
    p = argparse.ArgumentParser(prog="Phase-3 sim")
    p.add_argument("--gpu",    action="store_true", help="use Cupy if available")
    p.add_argument("--nodes",  type=int,   default=1000)
    p.add_argument("--frames", type=int,   default=250)
    p.add_argument("--dt",     type=float, default=0.05)
    p.add_argument("--dx",     type=float, default=1.0)
    args = p.parse_args()

    # build S & g on xp
    S = as_backend(
        xp.random.random(
            (args.nodes, args.nodes, args.nodes),
            dtype = xp.float32
        )
    )
    g = xp.zeros((args.nodes - 2,)*3, dtype=xp.float32)

    # initialise M = nodes test particles
    particles = xp.zeros((args.nodes, 6), dtype=xp.float32)
    # example: line of particles along x, centered
    particles[:,0] = xp.linspace(-args.nodes/2, args.nodes/2, args.nodes)

    state = {"S": S, "g": g}

    # run
    for _ in range(args.frames):
        particles = step(state, particles, args.dt, args.dx)

    # save final trajectory
    traj = asnumpy(particles)
    _np.savez("phase3_traj.npz", traj=traj)
    print("Done → phase3_traj.npz")

if __name__ == "__main__":
    main()
