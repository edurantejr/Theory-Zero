"""Diagnostics and metrics."""

from __future__ import annotations

import numpy as np


def energy_harmonic(state: np.ndarray, omega: float) -> float:
    """Compute energy for harmonic oscillator."""
    x, v = state
    return 0.5 * (v**2 + (omega * x) ** 2)
