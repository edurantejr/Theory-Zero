# sim/run_phase3.py
"""
Phase-3 Black-Hole Simulator driver
----------------------------------
• If --wij <file> is given, converts Phase-2 weight matrix w_ij → S(x)
  and splashes each node on a 3-D lattice at a **random location**
  (4-voxel margin).  Otherwise uses a single-dent demo.
• Evolves metric + particles and writes phase3_traj.npz
"""

from __future__ import annotations
import argparse, time, sim.backend as np
from pathlib import Path

from sim.integrators   import step
from sim.io            import save_npz
from sim.entropy_field import load_wij, node_entropy, splash_to_lattice


# ── helper ──────────────────────────────────────────────────────
def build_entropy_field(L: int, wij_path: Path | None) -> np.ndarray:
    """
    Returns S(x) on an L×L×L lattice.
    Randomly scatters nodes with a 4-cell safety margin.
    """
    if wij_path and wij_path.exists():
        print(f"• loading Phase-2 weights from {wij_path}")
        wij    = load_wij(str(wij_path))
        S_node = node_entropy(wij)
        S_node += 0.05 * np.random.rand(*S_node.shape).astype(np.float32)
        print(f"  node-entropy min/max  {S_node.min():.4f}  {S_node.max():.4f}")

        # normalise 0–1
        S_node -= S_node.min()
        if S_node.max() > 0:
            S_node /= S_node.max()

        # ---------------------------------------------------------------
        #  Randomly scatter each Phase-2 node, then push half of them
        #  to the left and half to the right so the field is asymmetric.
        # ---------------------------------------------------------------

        margin = 4                                  # keep bumps away from lattice edge
        coords = np.random.uniform(margin, L - margin,
                                    (wij.shape[0], 3)).astype(np.float32)

        # Hard split: move first half left, second half right
        half   = coords.shape[0] // 2
        shift  = min(10, (L // 2) - margin)          # never leave the lattice
        coords[:half, 0]  -= shift                   # x – shift  (left cluster)
        coords[half:, 0] += shift                    # x + shift  (right cluster)

        # Optional: make the right cluster stronger
        S_node[half:] *= 2.0                         # double entropy for right cluster


        S = splash_to_lattice(S_node, coords, (L, L, L), sigma=1.0)
        print(f"  lattice S range       {S.min():.4f}  {S.max():.4f}")

        if np.isclose(S.max() - S.min(), 0.0):
            print("  ⚠ flat entropy field → reverting to single dent")
            S.fill(0.0)
            S[L // 2, L // 2, L // 2] = 1.0
    else:
        print("• no weights provided → single-dent demo")
        S = np.zeros((L, L, L), np.float32)
        S[L // 2, L // 2, L // 2] = 1.0

    return S


# ── CLI and main loop ───────────────────────────────────────────
def positive_int(x: str) -> int: return max(1, int(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes",  type=positive_int, default=5000)
    ap.add_argument("--frames", type=positive_int, default=250)
    ap.add_argument("--fps",    type=positive_int, default=20)
    ap.add_argument("--out",    default="phase3_traj.npz")
    ap.add_argument("--wij",    type=Path, default=None, help=".npy / .csv weight matrix")
    args = ap.parse_args()

    dt, dx = 1.0 / args.fps, 1.0
    L      = int(round(args.nodes ** (1 / 3)))

    S = build_entropy_field(L, args.wij)
    g = np.zeros((L - 2, L - 2, L - 2), np.float32)     # metric interior
    state = dict(S=S, g=g)

    particles = np.zeros((args.nodes, 6), np.float32)   # [x y z vx vy vz]
    particles[:, 0:3] = np.random.uniform(-0.4, 0.4, (args.nodes, 3))

    traj = np.empty((args.frames, args.nodes, 3), np.float32)

    print(f"▶ Phase-3 sim  N={args.nodes:,}  frames={args.frames}  fps={args.fps}")
    t0 = time.time()
    for f in range(args.frames):
        traj[f] = particles[:, 0:3]
        step(state, particles, dt, dx)
    print(f"⏱  completed in {time.time() - t0:.1f}s")

    save_npz(traj, args.fps, args.out)


if __name__ == "__main__":
    main()
