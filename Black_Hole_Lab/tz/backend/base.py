"""Backend abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ArrayBackend(Protocol):
    """Protocol for array backends."""

    name: str
    xp: object

    def asarray(self, data, dtype=None):  # noqa: ANN001
        ...


@dataclass(frozen=True)
class NumpyBackend:
    """Numpy backend implementation."""

    name: str = "numpy"
    xp: object = np

    def asarray(self, data, dtype=None):  # noqa: ANN001
        return np.asarray(data, dtype=dtype)


@dataclass(frozen=True)
class CupyBackend:
    """Cupy backend implementation (optional)."""

    name: str = "cupy"
    xp: object = None

    def __post_init__(self) -> None:
        if self.xp is None:
            import cupy as cp  # type: ignore

            object.__setattr__(self, "xp", cp)

    def asarray(self, data, dtype=None):  # noqa: ANN001
        return self.xp.asarray(data, dtype=dtype)


BACKENDS = {
    "numpy": NumpyBackend,
    "cupy": CupyBackend,
}


def get_backend(name: str) -> ArrayBackend:
    """Return backend instance by name."""
    name = name.lower()
    if name not in BACKENDS:
        raise ValueError(f"Unsupported backend: {name}")
    return BACKENDS[name]()
