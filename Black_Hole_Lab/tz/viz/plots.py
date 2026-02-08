"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def plot_metric(steps: Iterable[int], values: Iterable[float], *, title: str, ylabel: str, outpath: Path) -> None:
    """Create a simple line plot for a metric."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(steps), list(values), linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
