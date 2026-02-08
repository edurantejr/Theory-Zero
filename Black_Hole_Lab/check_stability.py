"""Deprecated wrapper for legacy script."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path

warnings.warn(
    "check_stability.py is deprecated. Use tz.core.checks instead.",
    DeprecationWarning,
)

legacy_path = Path(__file__).resolve().parent / "legacy" / "check_stability.py"
runpy.run_path(str(legacy_path), run_name="__main__")
