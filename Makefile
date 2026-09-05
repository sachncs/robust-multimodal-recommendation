.PHONY: help install dev install-pre-commit test test-fast lint lint-fix format typecheck build clean demo bench serve reproduce fidelity docs lock

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

dev: ## Install with development dependencies
	pip install -e ".[dev]"

test: ## Run tests with coverage
	pytest tests/

test-fast: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linter
	ruff check .

lint-fix: ## Run linter with auto-fix
	ruff check --fix .

format: ## Format code
	ruff format .

typecheck: ## Run type checker
	mypy morel/

build: ## Build the package
	python -m build

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

demo: ## Run the demo script
	python examples/end_to_end_demo.py

bench: ## Run benchmarks
	pytest benchmarks/ --benchmark-only --benchmark-min-rounds=20

serve: ## Run the inference server
	python -m morel serve --host 0.0.0.0 --port 8080

reproduce: ## Run end-to-end reproduction
	python -m morel reproduce configs/reproduce.yaml

fidelity: ## Render the fidelity report
	python -m morel render-fidelity

docs: ## Build documentation
	mkdocs build --strict

lock: ## Generate requirements.lock from pyproject.toml
	pip-compile pyproject.toml -o requirements.lock

check: lint test ## Run lint and tests
