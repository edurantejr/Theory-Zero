"""Model registry."""

from __future__ import annotations

from typing import Any, Dict

from theory_zero.models.base import HarmonicOscillator, Model


def build_model(config: Dict[str, Any], *, xp: object = None) -> Model:
    """Build model from config dictionary."""
    name = config.get("name", "harmonic_oscillator")
    xp = xp or __import__("numpy")
    if name == "harmonic_oscillator":
        return HarmonicOscillator(
            omega=float(config.get("omega", 1.0)),
            x0=float(config.get("x0", 1.0)),
            v0=float(config.get("v0", 0.0)),
            xp=xp,
        )
    raise ValueError(f"Unknown model {name}")


__all__ = ["build_model", "HarmonicOscillator", "Model"]
