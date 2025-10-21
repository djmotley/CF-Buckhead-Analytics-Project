.PHONY: setup fmt lint test cov

setup:
\tpyenv local 3.13.9
\tpython -m venv .venv
\t. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"
\tpre-commit install

fmt:
\tblack .
\truff format .
\tisort .

lint:
\truff check .
\tmypy src

test:
\tpytest

cov:
\tpytest --cov=cf_buckhead_analytics --cov-report=html
