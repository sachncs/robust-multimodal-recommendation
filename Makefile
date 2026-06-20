.PHONY: help install dev test lint format typecheck build clean demo install-pre-commit

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

dev: ## Install with development dependencies
	pip install -e ".[dev]"

install-pre-commit: ## Install pre-commit hooks
	pre-commit install

test: ## Run tests with coverage
	pytest tests/ -v --cov=rmr --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linter
	ruff check .

lint-fix: ## Run linter with auto-fix
	ruff check --fix .

format: ## Format code
	ruff format .

typecheck: ## Run type checker
	mypy rmr/ --ignore-missing-imports

build: ## Build the package
	python -m build

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

demo: ## Run the demo script
	python demo.py

lint-all: lint typecheck ## Run all linting checks

check: lint test ## Run lint and tests
