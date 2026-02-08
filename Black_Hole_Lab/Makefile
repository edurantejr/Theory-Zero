setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e .[dev]

run:
	python -m experiments.run --config experiments/configs/baseline.yaml

test:
	pytest

report:
	python -m scripts.report --last 10
