# sim/run_phase3.py

import argparse
from . import backend
from .backend import xp, as_backend, asnumpy
from .integrators import step

def main():
    p = argparse.ArgumentParser(
        description="Phase-3 Emergent Gravity Simulation"
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="use the CuPy backend if available"
    )
    p.add_argument(
        "--nodes", type=int, default=100,
        help="number of grid points per side (N³ lattice)"
    )
    p.add_argument(
        "--frames", type=int, default=100,
        help="number of timesteps to simulate"
    )
    p.add_argument(
        "--dt", type=float, default=0.1,
        help="time step Δt"
    )
    p.add_argument(
        "--dx", type=float, default=1.0,
        help="spatial grid spacing Δx"
    )
    args = p.parse_args()

    # ——— pick backend ———
    if args.gpu:
        if backend._cp is None:
            print("⚠️  cupy not installed, falling back to NumPy")
        else:
            backend.xp = backend._cp

    # ——— allocate fields ———
    N = args.nodes
    # entanglement-entropy field on an N×N×N grid
    S = xp.random.rand(N, N, N).astype(xp.float32)
    S = as_backend(S)

    # metric g00 lives on the interior (N-2)³
    g = xp.zeros((N-2, N-2, N-2), dtype=xp.float32)

    # ——— setup particle array ———
    # each particle: [x, y, z, vx, vy, vz]
    particles = xp.zeros((N, 6), dtype=xp.float32)
    # e.g. initialize one test particle at (−N/2,0,0) with vx=+0.5
    particles[0, 0:3] = xp.array([-N/2, 0.0, 0.0], dtype=xp.float32)
    particles[0, 3]   = 0.5

    # ——— time-march ———
    for frame in range(args.frames):
        # step updates particles in-place
        step({"S": S, "g": g}, particles, args.dt, args.dx)

    # ——— save final trajectories ———
    # bring back to NumPy
    traj = asnumpy(particles)
    # and save with either xp (np or cp)
    xp.savez("phase3_traj.npz", traj=traj)
    print(f"✔️  Done: saved phase3_traj.npz with shape {traj.shape}")

if __name__ == "__main__":
    main()
