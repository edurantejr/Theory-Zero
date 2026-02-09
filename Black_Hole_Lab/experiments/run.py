"""Experiment entrypoint for Theory Zero."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from theory_zero.backend import get_backend
from theory_zero.core.checks import ensure_dtype, ensure_finite, ensure_stable
from theory_zero.core.constants import DEFAULT_DTYPE, DIVERGENCE_THRESHOLD
from theory_zero.core.seed import set_seed
from theory_zero.db.api import ingest_legacy, log_artifact, log_metric, log_run
from theory_zero.integrators import build_integrator
from theory_zero.io import build_run_dir, get_env_info, get_git_info, write_json, write_yaml
from theory_zero.metrics import energy_harmonic
from theory_zero.models import build_model


@dataclass
class RunConfig:
    name: str
    seed: int
    backend: str
    device: str
    notes: str
    model: Dict[str, Any]
    integrator: Dict[str, Any]
    metrics: Dict[str, Any]


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    return yaml.safe_load(path.read_text())


def resolve_config(raw: Dict[str, Any], overrides: Dict[str, Any]) -> RunConfig:
    """Resolve config with CLI overrides."""
    merged = {**raw, **{k: v for k, v in overrides.items() if v is not None}}
    return RunConfig(
        name=merged.get("name", "baseline"),
        seed=int(merged.get("seed", 0)),
        backend=merged.get("backend", "numpy"),
        device=merged.get("device", "cpu"),
        notes=str(merged.get("notes", "")),
        model=merged.get("model", {}),
        integrator=merged.get("integrator", {}),
        metrics=merged.get("metrics", {}),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Theory Zero experiment runner")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--backend")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
    overrides = {
        "seed": args.seed,
        "device": args.device,
        "backend": args.backend,
        "notes": args.notes,
    }
    config = resolve_config(raw_config, overrides)

    repo_root = REPO_ROOT
    git_info = get_git_info(repo_root)

    run_root = args.outdir or repo_root / "runs"
    run_dir = build_run_dir(run_root, config.name, git_info.sha)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(run_dir / "logs.txt"),
            logging.StreamHandler(),
        ],
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logging.info("Starting run %s", run_dir.name)
    if args.resume:
        logging.warning("Resume flag provided but resume logic is not implemented yet.")
    set_seed(config.seed)

    backend = get_backend(config.backend)
    model = build_model(config.model, xp=backend.xp)
    integrator = build_integrator(config.integrator)
    dt = float(config.integrator.get("dt", 0.01))
    steps = int(config.integrator.get("steps", 1000))
    record_every = int(config.metrics.get("record_every", 1))

    state = backend.asarray(model.initial_state(), dtype=DEFAULT_DTYPE)
    time_value = 0.0

    ensure_dtype(state, dtype=DEFAULT_DTYPE, name="state")

    metrics_path = run_dir / "metrics.csv"
    tracemalloc.start()
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time", "x", "v", "energy", "step_time_ms"])  # header

        trajectory: List[np.ndarray] = []
        step_times: List[float] = []
        energy_log: List[tuple[int, float]] = []

        start_time = time.perf_counter()
        for step in range(steps + 1):
            step_start = time.perf_counter()
            if step < steps:
                state = integrator.step(state, time_value, dt, model.derivative)
                time_value += dt
                ensure_finite(state, name="state")
                ensure_stable(state, threshold=DIVERGENCE_THRESHOLD, name="state")
            step_time_ms = (time.perf_counter() - step_start) * 1000.0
            step_times.append(step_time_ms)

            if step % record_every == 0:
                energy = energy_harmonic(state, float(config.model.get("omega", 1.0)))
                writer.writerow([step, time_value, float(state[0]), float(state[1]), energy, step_time_ms])
                energy_log.append((step, float(energy)))
                trajectory.append(state.copy())

        runtime = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    config_payload = {
        **raw_config,
        "resolved": {
            "seed": config.seed,
            "backend": config.backend,
            "device": config.device,
            "notes": config.notes,
        },
    }
    write_yaml(run_dir / "config_resolved.yaml", config_payload)
    write_json(run_dir / "env.json", get_env_info())
    write_json(
        run_dir / "git.json",
        {"sha": git_info.sha, "branch": git_info.branch, "dirty": git_info.dirty},
    )

    summary = {
        "final_state": state.tolist(),
        "runtime_seconds": runtime,
        "mean_step_ms": float(np.mean(step_times)) if step_times else 0.0,
        "memory_current_bytes": current,
        "memory_peak_bytes": peak,
        "seed": config.seed,
        "backend": config.backend,
        "device": config.device,
    }
    write_json(run_dir / "summary.json", summary)

    trajectory_arr = np.stack(trajectory) if trajectory else np.zeros((0, 2))
    artifacts_path = run_dir / "artifacts" / "trajectory.npz"
    np.savez(artifacts_path, trajectory=trajectory_arr)

    config_hash = hashlib.sha256((run_dir / "config_resolved.yaml").read_bytes()).hexdigest()
    run_id = log_run(
        timestamp=datetime.now(timezone.utc),
        git_sha=git_info.sha,
        config_hash=config_hash,
        seed=config.seed,
        backend=config.backend,
        device=config.device,
        runtime=runtime,
        status="completed",
        params={"config_name": config.name, "notes": config.notes},
    )

    for step, step_time in enumerate(step_times):
        log_metric(run_id, step, "step_time_ms", float(step_time))
    for step, energy in energy_log:
        log_metric(run_id, step, "energy", float(energy))

    artifact_hash = hashlib.sha256(artifacts_path.read_bytes()).hexdigest()
    try:
        artifact_path = artifacts_path.relative_to(repo_root)
    except ValueError:
        artifact_path = artifacts_path
    log_artifact(run_id, "trajectory", str(artifact_path), artifact_hash)

    ingest_legacy(repo_root / "legacy")

    logging.info("Run complete: %s", run_dir.name)


if __name__ == "__main__":
    main()
