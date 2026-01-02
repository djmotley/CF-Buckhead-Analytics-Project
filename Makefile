.PHONY: setup fmt lint test cov

setup:
	pyenv local 3.13.9
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"
	pre-commit install

fmt:
	black .
	ruff format .
	isort .

lint:
	ruff check .
	mypy src

test:
	pytest

cov:
	pytest --cov=cf_buckhead_analytics --cov-report=html
