import sim.backend as np   # or 'xp' if you prefer the alias

# ───────────────────────── Ricci tensor ───────────────────────
def ricci_tensor(S: np.ndarray,
                 dx: float = 1.0,
                 *,               # force keyword args from here
                 h: float | None = None,      # legacy alias
                 kappa: float = -0.138475,
                 gamma: float = -1.0):
    """
    Returns (R00, Rii) on the (L-2)³ interior grid.

    Phase-2 core relation
        Rμν = κ ( ∇μ∇ν S − γ gμν ∇²S ),   g00 = −1.
    The static tables used dx = 0.9.
    """
    if h is not None:
        dx = h

    # 6-point Laplacian and second derivatives
    lap = (
          S[ :-2,1:-1,1:-1] + S[2:,1:-1,1:-1]
        + S[1:-1, :-2,1:-1] + S[1:-1,2:,1:-1]
        + S[1:-1,1:-1, :-2] + S[1:-1,1:-1,2:]
        - 6.0 * S[1:-1,1:-1,1:-1]
    ) / dx**2

    dxx = (S[2:,1:-1,1:-1]  - 2*S[1:-1,1:-1,1:-1] + S[ :-2,1:-1,1:-1]) / dx**2
    dyy = (S[1:-1,2:,1:-1]  - 2*S[1:-1,1:-1,1:-1] + S[1:-1, :-2,1:-1]) / dx**2
    dzz = (S[1:-1,1:-1,2:]  - 2*S[1:-1,1:-1,1:-1] + S[1:-1,1:-1, :-2]) / dx**2

    R00 = kappa * lap                            # g00 = −1, γ = −1
    Rii = -kappa * (dxx + dyy + dzz) / 3.0       # isotropic spatial diag
    return R00, Rii

# ───────────────────────── stability helper ──────────────────────
def safe_step(g00: np.ndarray,
              R00: np.ndarray,
              dt:  float,
              dx:  float,
              safety: float = 0.4):
    """
    CFL-style guard: limit dt so   |Δg00| ≤ safety · dx²

    If the proposed dt would change any g00 value by more than
    safety·dx², we scale dt down before applying the update.
    """
    max_delta = np.abs(R00).max() * dt
    limit     = safety * dx * dx
    if max_delta > limit and max_delta != 0.0:
        dt *= limit / max_delta     # shrink dt
    return g00 - dt * R00, dt       # return (new_g00, actual_dt)

# ───────────────────────── metric update ──────────────────────
def evolve_metric(g: np.ndarray,
                  S: np.ndarray,
                  dt: float,
                  dx: float,
                  *,
                  kappa: float = -0.138475,
                  gamma: float = -1.0,
                  damping: float = 0.1) -> np.ndarray:

    R00, _ = ricci_tensor(S, dx=dx, kappa=kappa, gamma=gamma)
    g_new, dt_eff = safe_step(g, R00, dt, dx)        # stability guard
    return (1.0 - damping) * g_new                  # optional damping

# ───────────────────────── Phase-3 particle force ─────────────
def force_phase3(g00: np.ndarray, dx: float = 1.0):
    """F = −½ ∇g00  on the interior lattice."""
    Fx = -(g00[2:,1:-1,1:-1] - g00[:-2,1:-1,1:-1]) / (2*dx) * 0.5
    Fy = -(g00[1:-1,2:,1:-1] - g00[1:-1,:-2,1:-1]) / (2*dx) * 0.5
    Fz = -(g00[1:-1,1:-1,2:] - g00[1:-1,1:-1,:-2]) / (2*dx) * 0.5
    return Fx, Fy, Fz
