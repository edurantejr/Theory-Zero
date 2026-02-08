# Architecture

## Module map

- `tz.core`: constants, typing, seed control, invariant checks
- `tz.backend`: backend abstraction (numpy/cupy)
- `tz.models`: physics models and operators
- `tz.integrators`: time-stepping algorithms
- `tz.metrics`: metrics and diagnostics
- `tz.viz`: plotting utilities
- `tz.io`: run metadata and serialization
- `tz.db`: findings database API

## Data flow

1. `experiments.run` loads a YAML config.
2. The run is materialized in `runs/<timestamp>_<name>_<gitsha>/` with metadata.
3. The simulation loop logs metrics, artifacts, and summary outputs.
4. Results are inserted into `db/findings.sqlite` for querying/reporting.
