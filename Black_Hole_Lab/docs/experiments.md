# Experiments

## Configs

Configs live in `experiments/configs` and are YAML files with model, integrator, and metrics settings.

## Sweeps

Sweep definitions live in `experiments/sweeps` and can be used by future automation.

## Reports

Generate a report of recent runs with:

```bash
python -m scripts.report --last 10
```
