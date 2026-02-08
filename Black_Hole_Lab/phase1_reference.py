"""Deprecated wrapper for legacy script."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path

warnings.warn(
    "phase1_reference.py is deprecated. Use experiments/configs YAML files.",
    DeprecationWarning,
)

legacy_path = Path(__file__).resolve().parent / "legacy" / "phase1_reference.py"
runpy.run_path(str(legacy_path), run_name="__main__")
