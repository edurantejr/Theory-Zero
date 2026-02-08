"""Model interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Model(Protocol):
    """Protocol for dynamical system models."""

    name: str

    def initial_state(self) -> np.ndarray:
        ...

    def derivative(self, state: np.ndarray, time: float) -> np.ndarray:
        ...


@dataclass(frozen=True)
class HarmonicOscillator:
    """1D harmonic oscillator model."""

    omega: float
    x0: float
    v0: float
    xp: object = np
    name: str = "harmonic_oscillator"

    def initial_state(self) -> np.ndarray:
        return self.xp.array([self.x0, self.v0], dtype=float)

    def derivative(self, state: np.ndarray, time: float) -> np.ndarray:  # noqa: ARG002
        x, v = state
        dx = v
        dv = -(self.omega**2) * x
        return self.xp.array([dx, dv], dtype=float)
