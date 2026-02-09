# Legacy Migration Map

## File mapping

- `auto_setup_phase5.py` → `experiments/run.py` (new entrypoint)
- `blackhole_simulation.py` → `src/theory_zero/models` + `src/theory_zero/integrators`
- `check_stability.py` → `src/theory_zero/core/checks.py`
- `make_refs.py` → `src/theory_zero/db/api.py` (legacy findings ingest)
- `phase1_reference.py` → `experiments/configs/*.yaml`
- `phase5_blackhole_bake.py` → `experiments/run.py`

## CLI mapping

- Old runners: `python auto_setup_phase5.py`, `python blackhole_simulation.py`
- New runner: `python -m experiments.run --config experiments/configs/baseline.yaml`

## Notes

All original files and assets are preserved under `legacy/`. New wrappers at the repo root call into the legacy scripts and emit a deprecation warning.
