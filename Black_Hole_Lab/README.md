# Theory Zero Research OS

Theory Zero is now structured as a reproducible research codebase with a single experiment entrypoint, versioned configs, and a findings database.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
```

## Quickstart run

```bash
python -m experiments.run --config experiments/configs/baseline.yaml
```

You can also use the CLI entry point:

```bash
theory-zero --config experiments/configs/baseline.yaml
```

## Minimal demo

```bash
python -m scripts.run_example
```

This runs a fast smoke configuration and writes output artifacts into a temporary run directory.

## Main simulations

Create or edit YAML configs in `experiments/configs/` and run them via:

```bash
python -m experiments.run --config experiments/configs/<name>.yaml
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
make install
make lint
make format
make run
make run-example
make test
make report
```

## Troubleshooting

- **`ModuleNotFoundError: numpy`**: ensure you've installed `requirements.txt` and the editable package install succeeded.
- **Database locked**: delete `db/findings.sqlite-shm` and `db/findings.sqlite-wal` if a prior run crashed.
- **GPU backend errors**: install the optional `.[gpu]` extras and verify your CUDA toolkit matches the CuPy build.

## Legacy work

All previous scripts and assets live under `legacy/` with a migration map in `legacy/README_MIGRATION.md`.
