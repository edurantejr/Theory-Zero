import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_smoke(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "experiments" / "configs" / "smoke.yaml"
    outdir = tmp_path / "runs"
    env = os.environ.copy()
    env["TZ_DB_PATH"] = str(tmp_path / "findings.sqlite")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.run",
            "--config",
            str(config),
            "--outdir",
            str(outdir),
        ],
        check=True,
        env=env,
    )
    run_dirs = sorted(outdir.glob("*"))
    assert run_dirs, "Run directory not created"
    return run_dirs[-1]


def test_cli_smoke_run_outputs(tmp_path: Path):
    run_dir = run_smoke(tmp_path)
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "artifacts" / "trajectory.npz").exists()


def test_cli_regression_shape(tmp_path: Path):
    run_dir = run_smoke(tmp_path)
    data = np.load(run_dir / "artifacts" / "trajectory.npz")
    trajectory = data["trajectory"]
    assert trajectory.shape == (11, 2)
