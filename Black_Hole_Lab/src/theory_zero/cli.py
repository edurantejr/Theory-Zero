"""Top-level CLI for Theory Zero."""

from __future__ import annotations

from experiments.run import main as run_main


def main() -> None:
    """Dispatch to the experiment runner."""
    run_main()
