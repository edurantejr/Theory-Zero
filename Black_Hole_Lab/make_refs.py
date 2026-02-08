"""Deprecated wrapper for legacy script."""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path

warnings.warn(
    "make_refs.py is deprecated. Use tz.db ingestion instead.",
    DeprecationWarning,
)

legacy_path = Path(__file__).resolve().parent / "legacy" / "make_refs.py"
runpy.run_path(str(legacy_path), run_name="__main__")
