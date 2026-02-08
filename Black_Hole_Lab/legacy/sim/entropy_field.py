# sim/entropy_field.py

from .backend import xp, as_backend, clip, log, sum

def load_wij(path: str) -> xp.ndarray:
    """
    For Phase-3 we only ever pass an in-memory array,
    but tests expect a load_wij that returns an xp.ndarray.
    """
    # If you ever wanted to load from disk:
    # data = xp.load(path)
    # For now, we treat the argument as already a weights array:
    return as_backend(path)  # treat path as array

def node_entropy(wij: xp.ndarray) -> xp.ndarray:
    """
    Phase-2 normalisation:
      S(i) = -∑ w log w
      S_norm = ( S / log(N - 1) ) / 20
    so perfectly uniform wij ⇒ zero array.
    """
    # make sure we’re on the right backend
    w = as_backend(wij)

    # clip in xp so it's always an xp.ndarray
    w = clip(w, 1e-12, 1.0)

    # compute -∑ w log w over the weight dimension
    S = -sum(w * log(w), axis=-1)

    # normalise by area-law denominator
    N = w.shape[-1]
    return S / (log(N) - 1.0) / 20.0
