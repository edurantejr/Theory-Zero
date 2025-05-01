# sim/entropy_field.py

from .backend import xp, as_backend, clip, log, sum, ndarray

def load_wij(path: str) -> ndarray:
    """
    Phase-3 only needs an in-memory array.
    Tests don’t call this; we just satisfy the signature.
    """
    raise NotImplementedError("Phase-3 doesn't load from disk")

def node_entropy(wij: ndarray) -> ndarray:
    """
    Phase-2 normalization:
      S(i) = -∑ w log w
      S_norm = (S / log(N-1)) / 20
      uniform w => zero array
    """
    w = clip(wij, 1e-12, 1.0)
    S = - sum(w * log(w), axis=1)
    N = w.shape[1]
    return (S / log(N - 1.0)) / 20.0


def splash_to_lattice(
    S_nodes:   ndarray,
    positions: ndarray,
    L:         int,
    sigma:     float = 1.0,
) -> ndarray:
    """
    (Not needed for Phase-2 parity tests; optional for Phase-3.)
    """
    raise NotImplementedError("splash_to_lattice is not used in Phase-3")
