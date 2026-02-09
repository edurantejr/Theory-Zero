"""Seed utilities."""

from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for numpy."""
    np.random.seed(seed)
