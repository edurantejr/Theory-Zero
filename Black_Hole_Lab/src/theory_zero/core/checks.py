"""Invariant checks and guardrails."""

from __future__ import annotations

import numpy as np


def ensure_finite(array: np.ndarray, *, name: str) -> None:
    """Raise if an array contains NaN or Inf."""
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf values")


def ensure_dtype(array: np.ndarray, *, dtype: np.dtype, name: str) -> None:
    """Raise if an array dtype is not as expected."""
    if array.dtype != dtype:
        raise TypeError(f"{name} dtype {array.dtype} != {dtype}")


def ensure_stable(array: np.ndarray, *, threshold: float, name: str) -> None:
    """Raise if an array norm exceeds a divergence threshold."""
    if np.linalg.norm(array) > threshold:
        raise ValueError(f"{name} diverged beyond threshold {threshold}")
