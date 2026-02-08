import numpy as np

def save_npz(traj: np.ndarray, fps: int, path: str):
    """Lightweight compressed archive."""
    np.savez_compressed(path, traj=traj, fps=fps)
    print("✅  wrote", path)

# stub for Alembic export (fill in later)
def save_alembic(traj: np.ndarray, path: str, fps: int):
    raise NotImplementedError(
        "Install alembic-python and implement save_alembic() when needed.")
