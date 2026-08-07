.PHONY: help install sync lint format typecheck test test-cov serve docs-build docs-serve clean

help:
	@echo "agent-evals development management"
	@echo "-----------------------------------"
	@echo "make install      : Setup virtual environment and install all dependencies using uv"
	@echo "make lint         : Run ruff linter"
	@echo "make format       : Format code using black and ruff"
	@echo "make typecheck    : Run mypy static type analysis"
	@echo "make test         : Run pytest suite"
	@echo "make test-cov     : Run pytest with coverage report"
	@echo "make serve        : Start FastAPI backend dev server"
	@echo "make docs-serve   : Serve MkDocs documentation locally"
	@echo "make clean        : Clean cache directories and build artifacts"

install:
	uv venv
	uv sync --extra dev --extra docs

lint:
	uv run ruff check .

format:
	uv run ruff check --fix .
	uv run black .

typecheck:
	uv run mypy src/agent_evals

test:
	uv run pytest

test-cov:
	uv run pytest --cov=agent_evals --cov-report=html

serve:
	uv run agent-evals serve --reload

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache htmlcov site dist build *.egg-info
