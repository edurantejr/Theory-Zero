# Legacy Migration Map

## File mapping

- `auto_setup_phase5.py` → `experiments/run.py` (new entrypoint)
- `blackhole_simulation.py` → `tz/models` + `tz/integrators`
- `check_stability.py` → `tz/core/checks.py`
- `make_refs.py` → `tz/db/api.py` (legacy findings ingest)
- `phase1_reference.py` → `experiments/configs/*.yaml`
- `phase5_blackhole_bake.py` → `experiments/run.py`

## CLI mapping

- Old runners: `python auto_setup_phase5.py`, `python blackhole_simulation.py`
- New runner: `python -m experiments.run --config experiments/configs/baseline.yaml`

## Notes

All original files and assets are preserved under `legacy/`. New wrappers at the repo root call into the legacy scripts and emit a deprecation warning.
