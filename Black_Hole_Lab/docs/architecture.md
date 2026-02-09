# Architecture

## Module map

- `theory_zero.core`: constants, typing, seed control, invariant checks
- `theory_zero.backend`: backend abstraction (numpy/cupy)
- `theory_zero.models`: physics models and operators
- `theory_zero.integrators`: time-stepping algorithms
- `theory_zero.metrics`: metrics and diagnostics
- `theory_zero.viz`: plotting utilities
- `theory_zero.io`: run metadata and serialization
- `theory_zero.db`: findings database API

## Data flow

1. `experiments.run` loads a YAML config.
2. The run is materialized in `runs/<timestamp>_<name>_<gitsha>/` with metadata.
3. The simulation loop logs metrics, artifacts, and summary outputs.
4. Results are inserted into `db/findings.sqlite` for querying/reporting.
