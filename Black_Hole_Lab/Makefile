install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -e .[dev]

lint:
	ruff src tests experiments scripts
	black --check src tests experiments scripts
	isort --check-only src tests experiments scripts

format:
	black src tests experiments scripts
	isort src tests experiments scripts

test:
	pytest -q

run-example:
	python -m scripts.run_example

run:
	python -m experiments.run --config experiments/configs/baseline.yaml

report:
	python -m scripts.report --last 10
