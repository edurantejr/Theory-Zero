"""Deprecated wrapper for legacy script."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path

warnings.warn(
    "blackhole_simulation.py is deprecated. Use `python -m experiments.run --config experiments/configs/baseline.yaml`.",
    DeprecationWarning,
)

legacy_path = Path(__file__).resolve().parent / "legacy" / "blackhole_simulation.py"
runpy.run_path(str(legacy_path), run_name="__main__")
