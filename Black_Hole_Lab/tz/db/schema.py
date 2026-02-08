"""Database schema utilities."""

from __future__ import annotations

from pathlib import Path


def schema_sql() -> str:
    """Return SQL schema definition."""
    return Path(__file__).resolve().parents[2].joinpath("db", "schema.sql").read_text()
