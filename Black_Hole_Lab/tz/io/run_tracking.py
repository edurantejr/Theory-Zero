"""Run tracking and metadata utilities."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class GitInfo:
    sha: str
    branch: str
    dirty: bool


def get_git_info(repo_root: Path) -> GitInfo:
    """Collect git metadata for the repository."""
    def run_git(args: list[str]) -> str:
        return subprocess.check_output(["git", *args], cwd=repo_root).decode().strip()

    sha = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(run_git(["status", "--porcelain"]))
    return GitInfo(sha=sha, branch=branch, dirty=dirty)


def get_env_info() -> Dict[str, Any]:
    """Collect environment metadata."""
    packages = ["numpy", "pyyaml", "matplotlib", "pytest", "hypothesis"]
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "env": {"TZ": os.environ.get("TZ")},
    }


def build_run_dir(root: Path, shortname: str, git_sha: str, timestamp: Optional[datetime] = None) -> Path:
    """Create deterministic run directory name."""
    timestamp = timestamp or datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{stamp}_{shortname}_{git_sha}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
