"""Integrator interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


DerivativeFn = Callable[[np.ndarray, float], np.ndarray]


class Integrator(Protocol):
    name: str

    def step(self, state: np.ndarray, time: float, dt: float, deriv: DerivativeFn) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EulerIntegrator:
    name: str = "euler"

    def step(self, state: np.ndarray, time: float, dt: float, deriv: DerivativeFn) -> np.ndarray:
        return state + dt * deriv(state, time)


@dataclass(frozen=True)
class RK4Integrator:
    name: str = "rk4"

    def step(self, state: np.ndarray, time: float, dt: float, deriv: DerivativeFn) -> np.ndarray:
        k1 = deriv(state, time)
        k2 = deriv(state + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = deriv(state + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = deriv(state + dt * k3, time + dt)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
