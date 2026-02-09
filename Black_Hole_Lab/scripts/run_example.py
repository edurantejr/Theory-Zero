"""Run a minimal example experiment in a temporary directory."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    config = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "smoke.yaml"
    with tempfile.TemporaryDirectory() as temp_dir:
        outdir = Path(temp_dir) / "runs"
        env = os.environ.copy()
        env["TZ_DB_PATH"] = str(Path(temp_dir) / "findings.sqlite")
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
        if run_dirs:
            print(f"Example run created: {run_dirs[-1]}")
        else:
            raise RuntimeError("Example run did not produce output")


if __name__ == "__main__":
    main()
