# Contributing to RMR

Thank you for your interest in contributing to RMR (Robust Multimodal Recommendation)! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Naming](#branch-naming)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/rmr.git
   cd rmr
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/sachin-cs/robust-multimodal-recommendation.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip or poetry for package management
- Git

### Installation

1. **Install the package in editable mode with dev dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Verify installation**:
   ```bash
   python -c "import rmr; print('Installation successful')"
   ```

3. **Run the test suite** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

## Branch Naming

Use descriptive branch names with prefixes:

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Adding or updating tests
- `chore/description` - Maintenance tasks

Examples:
- `feature/add-new-metric`
- `fix/retrieval-collision-handling`
- `docs/update-installation-guide`

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Other changes that don't modify src or test files

### Examples

```bash
# Feature
git commit -m "feat: add support for batch inference"

# Bug fix
git commit -m "fix: resolve collision handling in ACS algorithm"

# Documentation
git commit -m "docs: update installation guide for Python 3.12"

# Breaking change
git commit -m "feat: change model architecture

BREAKING CHANGE: model weights format has changed"
```

## Pull Request Process

1. **Update your fork** with the latest upstream changes:
   ```bash
   git fetch upstream
   git checkout master
   git merge upstream/master
   ```

2. **Create a pull request** from your feature branch to `master`

3. **Fill out the PR template** completely, including:
   - Summary of changes
   - Related issue (if applicable)
   - Testing performed
   - Checklist items

4. **Ensure CI passes** - all tests and linting must pass

5. **Request review** from a maintainer

6. **Address feedback** promptly and update your PR as needed

7. **Squash commits** if requested before merging

## Coding Standards

### Code Style

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use `ruff` for linting and formatting (configured in `pyproject.toml`)
- Maximum line length: 80 characters
- Use type hints for all public functions

### Code Quality

- Write clear, concise comments
- Avoid unnecessary comments (don't over-comment)
- Use docstrings for public APIs (Google style)
- Keep functions focused and reasonably short
- Use meaningful variable and function names

### Example Code Style

```python
"""Module for handling graph retrieval operations."""

import numpy as np
import scipy.sparse as sp


def anchor_retrieval(
    features: np.ndarray,
    mask: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Retrieve top-K nearest anchors for each item.

    Args:
        features: Feature matrix of shape (n_items, n_features).
        mask: Binary mask indicating missing modalities.
        k: Number of anchors to retrieve.

    Returns:
        Array of shape (n_items, k) containing anchor indices.
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=rmr --cov-report=term-missing

# Run specific test file
pytest tests/test_codebook.py -v

# Run specific test
pytest tests/test_codebook.py::TestCodebook::test_forward -v
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the existing test patterns in `tests/`
- Aim for high coverage, especially for core functionality
- Test edge cases and error conditions

### Test Structure

```python
"""Tests for module name."""

import pytest
import numpy as np


class TestClassName:
    """Tests for ClassName."""

    def test_method_name(self):
        """Test description of what this test verifies."""
        # Arrange
        input_data = ...
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_output

    def test_error_condition(self):
        """Test that error is raised for invalid input."""
        with pytest.raises(ValueError, match="expected error message"):
            function_under_test(invalid_input)
```

## Documentation

### Writing Documentation

- Update documentation for any changed functionality
- Use clear, concise language
- Include code examples where appropriate
- Follow the existing documentation style in `docs/`

### Documentation Structure

- `README.md` - Project overview and quick start
- `docs/SETUP.md` - Detailed installation and data preparation
- `docs/USAGE.md` - Usage instructions and examples
- `docs/ARCHITECTURE.md` - System architecture overview
- `CHANGELOG.md` - Version history (Keep a Changelog format)

## Reporting Bugs

When reporting bugs, please include:

1. **Clear title** describing the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment details**:
   - Python version
   - OS and version
   - Package versions (`pip list`)
5. **Error messages** or stack traces
6. **Minimal code example** that reproduces the issue

Use the bug report template when creating an issue.

## Requesting Features

When suggesting features:

1. **Clear description** of the proposed feature
2. **Use case** explaining why it's needed
3. **Expected behavior** and user experience
4. **Alternatives considered** (if any)
5. **Additional context** (references, examples, etc.)

Use the feature request template when creating an issue.

## Questions?

If you have questions about contributing, feel free to:

1. Check existing issues and documentation
2. Open a discussion issue
3. Contact the maintainers

Thank you for contributing to RMR!
