"""Integrator registry."""

from __future__ import annotations

from typing import Any, Dict

from tz.integrators.base import EulerIntegrator, Integrator, RK4Integrator


def build_integrator(config: Dict[str, Any]) -> Integrator:
    """Build integrator from config dict."""
    name = config.get("name", "rk4")
    if name == "euler":
        return EulerIntegrator()
    if name == "rk4":
        return RK4Integrator()
    raise ValueError(f"Unknown integrator {name}")


__all__ = ["build_integrator", "EulerIntegrator", "Integrator", "RK4Integrator"]
