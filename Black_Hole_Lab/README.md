# Theory Zero Research OS

Theory Zero is now structured as a reproducible research codebase with a single experiment entrypoint, versioned configs, and a findings database.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart run

```bash
python -m experiments.run --config experiments/configs/baseline.yaml
```

## How configs work

Configs live in `experiments/configs/` as YAML files. The CLI resolves the config and stores a fully-resolved copy in the run folder for traceability.

## Outputs

Runs are stored in `runs/YYYYMMDD_HHMMSS_<shortname>_<gitsha>/` and include:

- `config_resolved.yaml`
- `env.json`
- `git.json`
- `metrics.csv`
- `summary.json`
- `artifacts/`
- `logs.txt`

## Findings database

The SQLite database is stored in `db/findings.sqlite`. Query it with:

```bash
python -m scripts.report --last 10
```

This generates a markdown report and plots in the `reports/` folder.

## Make targets

```bash
make setup
make run
make test
make report
```

## Legacy work

All previous scripts and assets live under `legacy/` with a migration map in `legacy/README_MIGRATION.md`.
