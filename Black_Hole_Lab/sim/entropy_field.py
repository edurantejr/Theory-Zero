import sim.backend as np   # or 'xp' if you prefer the alias

# ───────────────────────── helpers ────────────────────────────
def load_wij(path: str) -> np.ndarray:
    """Load Phase-2 weight matrix (.npy, .csv, .txt)."""
    if path.endswith(".npy"):
        return np.load(path)
    return np.loadtxt(path, delimiter=",")

def node_entropy(wij: np.ndarray) -> np.ndarray:
    """
    Phase-2 normalisation:
        S(i) = −Σ w log w
        S_norm = ( S / log N − 1 ) / 20
    so perfectly uniform weights ⇒ 0.
    """
    w  = np.clip(wij, 1e-12, 1.0)
    S  = -(w * np.log(w)).sum(axis=1)
    N  = wij.shape[0]
    return (S / np.log(N) - 1.0) / 20.0

# ───────────────────────── node → lattice helper ──────────────────────────
def splash_to_lattice(S_nodes: np.ndarray,
                      positions: np.ndarray,
                      L: int,
                      sigma: float = 1.0,
                      *,
                      xp=np):
    """
    Very light-weight dispatcher used by run_phase3.py:

        • `positions` shape (N,3) in world space with origin at centre
        • Places a Gaussian 'splash' of total entropy S_nodes[i] onto the
          regular L×L×L lattice.
        • Uses the same backend (NumPy or CuPy) via `xp`.

    This is *not* a high-quality kernel; it’s just enough to unblock imports.
    """
    grid = xp.zeros((L, L, L), dtype=xp.float32)

    # precompute Gaussian weights in a 3σ cube
    r   = int(3 * sigma) + 1
    ax  = xp.arange(-r, r + 1, dtype=xp.float32)
    kern = xp.exp(-(ax[:, None, None]**2 +
                    ax[None, :, None]**2 +
                    ax[None, None, :]**2) / (2 * sigma * sigma))

    for (x, y, z), S in zip(positions, S_nodes):
        ix = int(x + L / 2)
        iy = int(y + L / 2)
        iz = int(z + L / 2)

        xs = slice(max(ix - r, 0), min(ix + r + 1, L))
        ys = slice(max(iy - r, 0), min(iy + r + 1, L))
        zs = slice(max(iz - r, 0), min(iz + r + 1, L))

        # slice the kernel to fit the grid edges
        kx = slice(xs.start - (ix - r), xs.stop - (ix - r))
        ky = slice(ys.start - (iy - r), ys.stop - (iy - r))
        kz = slice(zs.start - (iz - r), zs.stop - (iz - r))

        grid[xs, ys, zs] += S * kern[kx, ky, kz]

    return grid
