"""
Phase-3 curvature-driven particle simulation
===========================================

Run on CPU:

    python -m sim.run_phase3 --nodes 5000 --frames 250

Run on GPU (CuPy):

    python -m sim.run_phase3 --gpu --nodes 5000 --frames 250
"""
from __future__ import annotations
import argparse, time, math

# dynamic backend ─────────────────────────────────────────────────────────────
import sim.backend as xp            # NumPy *or* CuPy, selected by --gpu
import numpy as np                  # always plain NumPy (for Numba kernels)

from sim.entropy_field import (
    load_wij,          # phase-2 weights  (optional)
    node_entropy,      # S(i) from wij
    splash_to_lattice, # scatter S onto 3-D lattice
)

from sim.physics      import evolve_metric
from sim.integrators  import step        # Numba CPU kernels


def parse_args() -> argparse.Namespace:
    ap  = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true",
                    help="use CuPy backend if available")
    ap.add_argument("--nodes",  type=int, default=5000)
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--wij",    type=str, help="phase-2 weight matrix .npy")
    ap.add_argument("--out",    type=str, default="phase3_traj.npz")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    N      = args.nodes
    F      = args.frames
    fps    = 20
    dt     = 1.0 / fps
    dx     = 1.0            # lattice spacing
    L      = 64             # lattice side (adjust as you like)

    print(f"▶ Phase-3 sim  N={N:,}  frames={F}  fps={fps}")

    # ------------------------------------------------------------------ entropy
    if args.wij:
        print(f"• loading Phase-2 weights from {args.wij}")
        wij = load_wij(args.wij)           # → backend array
        S_nodes = node_entropy(wij)        # shape (N,)
    else:
        # no weights → simple demo: single entropy 'dent'
        S_nodes = xp.ones(N, dtype=xp.float32)

    # random particle positions in a cube of side L-4 centred at origin
    rng = xp.random.default_rng(0)
    pos = rng.uniform(-(L - 4) / 2, +(L - 4) / 2, size=(N, 3)).astype(xp.float32)

    # ---------------------------------------------------------------- lattice S
    S_lattice = splash_to_lattice(S_nodes, pos, L, sigma=1.0, xp=xp)

    # metric tensor g00 on (L-2)³ interior grid
    g   = xp.zeros((L - 2, L - 2, L - 2), dtype=xp.float32)

    # particles: (x, y, z, vx, vy, vz)
    particles = xp.zeros((N, 6), dtype=xp.float32)
    particles[:, 0:3] = pos
    # give them an outward radial kick
    r = xp.linalg.norm(pos, axis=1, keepdims=True) + 1e-6
    particles[:, 3:6] =  0.1 * pos / r

    # ----------------------------------------------------------------- simulate
    t0 = time.time()
    for frame in range(F):
        # 1. update metric on GPU / CPU backend
        g = evolve_metric(g, S_lattice, dt, dx)

        # 2. Numba integrator needs **NumPy** arrays
        if xp is not np:                         # CuPy in use
            particles_np = xp.asnumpy(particles)
            g00_np       = xp.asnumpy(g)
            S_np        = xp.asnumpy(S_lattice)
        else:
            particles_np = particles
            g00_np       = g
            S_np        = S_lattice

        # kick + drift (CPU)
        step({"S": S_np, "g": g00_np}, particles_np, dt, dx)

        # 3. copy back to GPU if necessary
        if xp is not np:
            particles = xp.asarray(particles_np)

    print(f"⏱  completed in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------- save traj
    np.savez_compressed(args.out,
                        particles=xp.asnumpy(particles),
                        g00=xp.asnumpy(g))
    print(f"✅  wrote {args.out}")


if __name__ == "__main__":
    main()
