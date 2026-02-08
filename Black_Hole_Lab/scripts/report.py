"""Generate markdown report from findings database."""

from __future__ import annotations

import argparse
from pathlib import Path

from tz.db.api import query
from tz.viz.plots import plot_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate run report")
    parser.add_argument("--last", type=int, default=10)
    parser.add_argument("--outdir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs = query("SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", [args.last])
    lines = ["# Theory Zero Report", "", "## Recent Runs", ""]
    if runs:
        lines.append("| id | timestamp | git_sha | seed | backend | device | status | runtime |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for run in runs:
            lines.append(
                f"| {run['id']} | {run['timestamp']} | {run['git_sha']} | {run['seed']} |"
                f" {run['backend']} | {run['device']} | {run['status']} | {run['runtime']:.3f} |"
            )
    else:
        lines.append("No runs recorded yet.")

    findings = query("SELECT * FROM findings ORDER BY created_at DESC LIMIT 20")
    lines.extend(["", "## Findings", ""])
    if findings:
        for finding in findings:
            lines.append(f"- **{finding['title']}** ({finding['created_at']}): {finding['description']}")
    else:
        lines.append("No findings yet.")

    report_path = args.outdir / "report.md"
    report_path.write_text("\n".join(lines))

    if runs:
        last_run_id = runs[0]["id"]
        metrics = query(
            "SELECT step, value FROM metrics WHERE run_id = ? AND key = ? ORDER BY step",
            [last_run_id, "step_time_ms"],
        )
        if metrics:
            steps = [row["step"] for row in metrics]
            values = [row["value"] for row in metrics]
            plot_metric(steps, values, title="Step Time", ylabel="ms", outpath=args.outdir / "step_time.png")


if __name__ == "__main__":
    main()
