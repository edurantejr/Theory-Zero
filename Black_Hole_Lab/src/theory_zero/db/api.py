"""SQLite findings database API."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from theory_zero.db.schema import schema_sql


DB_PATH = Path(os.environ.get("TZ_DB_PATH", "")) if os.environ.get("TZ_DB_PATH") else None


def get_db_path() -> Path:
    """Return the configured database path."""
    if DB_PATH is not None:
        return DB_PATH
    return Path(__file__).resolve().parents[3] / "db" / "findings.sqlite"


def connect() -> sqlite3.Connection:
    """Connect to the SQLite database, creating schema if needed."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(schema_sql())
    conn.commit()
    return conn


def hash_config(config: Dict[str, Any]) -> str:
    """Hash config dictionary for traceability."""
    payload = repr(sorted(config.items())).encode()
    return hashlib.sha256(payload).hexdigest()


def log_run(
    *,
    timestamp: datetime,
    git_sha: str,
    config_hash: str,
    seed: int,
    backend: str,
    device: str,
    runtime: float,
    status: str,
    params: Optional[Dict[str, Any]] = None,
) -> int:
    """Insert a run record and optional params."""
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (timestamp, git_sha, config_hash, seed, backend, device, runtime, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp.replace(tzinfo=timezone.utc).isoformat(),
                git_sha,
                config_hash,
                seed,
                backend,
                device,
                runtime,
                status,
            ),
        )
        run_id = cur.lastrowid
        if params:
            conn.executemany(
                "INSERT INTO params (run_id, key, value) VALUES (?, ?, ?)",
                [(run_id, key, str(value)) for key, value in params.items()],
            )
        conn.commit()
        return int(run_id)


def log_metric(run_id: int, step: int, key: str, value: float) -> None:
    """Insert a metric record."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO metrics (run_id, step, key, value) VALUES (?, ?, ?, ?)",
            (run_id, step, key, value),
        )
        conn.commit()


def log_artifact(run_id: int, kind: str, path: str, hash_value: str) -> None:
    """Insert artifact record."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO artifacts (run_id, kind, path, hash) VALUES (?, ?, ?, ?)",
            (run_id, kind, path, hash_value),
        )
        conn.commit()


def add_finding(
    *,
    title: str,
    description: str,
    evidence_run_id: Optional[int] = None,
    tags: Optional[Iterable[str]] = None,
) -> int:
    """Add a finding record."""
    tags_value = ",".join(tags) if tags else None
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO findings (created_at, title, description, evidence_run_id, tags)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                title,
                description,
                evidence_run_id,
                tags_value,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def query(sql: str, params: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
    """Run a query and return rows as dictionaries."""
    with connect() as conn:
        cur = conn.execute(sql, params or [])
        rows = cur.fetchall()
        return [dict(row) for row in rows]


def ingest_legacy(legacy_root: Path) -> None:
    """Ingest legacy references as findings."""
    candidates = [
        legacy_root / "phase1_reference.json",
        legacy_root / "phase2_reference.json",
    ]
    with connect() as conn:
        for path in candidates:
            if not path.exists():
                continue
            title = f"Legacy reference: {path.name}"
            existing = conn.execute(
                "SELECT 1 FROM findings WHERE title = ? LIMIT 1",
                (title,),
            ).fetchone()
            if existing:
                continue
            conn.execute(
                """
                INSERT INTO findings (created_at, title, description, evidence_run_id, tags)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    title,
                    path.read_text()[:4000],
                    None,
                    "legacy,reference",
                ),
            )
        conn.commit()
