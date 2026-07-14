# RMR: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

[![CI](https://github.com/sachncs/robust-multimodal-recommendation/actions/workflows/ci.yml/badge.svg)](https://github.com/sachncs/robust-multimodal-recommendation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sachncs/robust-multimodal-recommendation/branch/main/graph/badge.svg)](https://codecov.io/gh/sachncs/robust-multimodal-recommendation)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reproduction of the paper **"Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion"** (GRE-MC) by Li et al. (NUS, 2026).

## Features

- **Modality-Aware Subgraph Retrieval** — Retrieves semantically relevant subgraphs from the item-item graph using anchor retrieval, ACS, and MAGE algorithms
- **Joint-Encoding Graph Transformer** — Encodes query nodes with retrieved context and Laplacian positional encodings
- **Sparse-Routing Codebook** — Compresses latent representations via Gumbel-Softmax and Top-P selection
- **Modality Completion** — Reconstructs missing modalities using per-modality MLP decoders
- **Downstream Recommendation** — LightGCN trained on the completed features
- **Comprehensive Test Suite** — Full test coverage with pytest and continuous integration
- **Modular Design** — Clean, maintainable codebase with clear separation of concerns

## Overview

RMR addresses missing modalities in multimodal recommender systems by:

1. **Modality-Aware Subgraph Retrieval** (anchor retrieval + ACS + MAGE) — retrieves semantically relevant subgraphs from the item-item graph.
2. **Joint-Encoding Graph Transformer** — encodes query nodes with retrieved context and Laplacian positional encodings.
3. **Sparse-Routing Codebook** — compresses latent representations via Gumbel-Softmax and Top-P selection.
4. **Modality Completion** — reconstructs missing modalities using per-modality MLP decoders.
5. **Downstream Recommendation** — LightGCN trained on the completed features.

> **Fidelity**: See `docs/FIDELITY_REPORT.md` for a section-by-section audit of this reproduction versus the paper. Every component is labeled EXACT, APPROXIMATE, or UNKNOWN, with deviations documented.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or poetry for package management

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sachncs/robust-multimodal-recommendation.git
cd rmr

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Alternative Installation

```bash
# Install without dev dependencies
pip install .

# Install with specific Python version
python3.10 -m pip install -e ".[dev]"
```

## Usage

### Quick Demo

Run a complete end-to-end pipeline on synthetic data:

```bash
python demo.py
```

### Data Pipeline

```bash
# Download Amazon 5-core review and metadata
python -m rmr.data.download

# Extract text and visual features
python -m rmr.data.features

# Build user-item and item-item graphs
python -m rmr.data.graph_builder

# Apply random modality masking (40%)
python -m rmr.data.masking
```

### Training

```bash
# Stage 1: Train the modality completion model
python -m rmr.scripts.train_completion --data-dir data/processed --epochs 100

# Stage 2: Train the downstream recommender
python -m rmr.scripts.train_recommender --data-dir data/processed --epochs 100
```

### Evaluation

```bash
python -m rmr.scripts.evaluate --predictions results/scores.npy --labels results/labels.npy
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVICE` | Device for training (cpu, cuda) | `cpu` |
| `DATA_PROCESSED_DIR` | Path to processed data | `data/processed` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Configuration Files

- `pyproject.toml` — Project configuration, dependencies, and tool settings
- `.editorconfig` — Editor configuration for consistent formatting
- `.gitattributes` — Git line ending normalization

## Project Structure

```
rmr/
├── data/              # Data pipeline (download, features, graphs, masking)
├── models/            # Core model components
│   ├── retrieval.py       # Anchor retrieval + ACS + MAGE algorithms
│   ├── positional_encoding.py  # Laplacian PE
│   ├── transformer.py     # Joint-encoding graph transformer
│   ├── codebook.py        # Sparse-routing codebook
│   ├── decoder.py         # Modality-specific decoders
│   ├── downstream.py      # LightGCN recommender
│   └── gre_mc.py          # End-to-end GRE-MC model
├── training/          # Training loops
├── evaluation/        # Metrics (Recall@K, NDCG@K)
├── scripts/           # CLI training and evaluation scripts
└── tests/             # Comprehensive test suite
```

## Development

### Available Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run development server (if applicable)
python demo.py

# Build the package
python -m build

# Run tests
pytest tests/ -v --cov=rmr --cov-report=term-missing

# Run linter
ruff check .

# Format code
ruff format .

# Run type checker (if configured)
mypy rmr/
```

### Pre-commit Hooks

This project uses pre-commit hooks for code quality. Install them with:

```bash
pip install pre-commit
pre-commit install
```

## Tech Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.10 | Programming language |
| PyTorch | >= 2.0.0 | Deep learning framework |
| NumPy | >= 1.24.0 | Numerical computing |
| SciPy | >= 1.10.0 | Sparse matrices, scientific computing |
| scikit-learn | >= 1.3.0 | ML utilities |
| sentence-transformers | >= 2.2.0 | Text embedding extraction |
| torchvision | >= 0.15.0 | Image features (ResNet) |
| pandas | >= 2.0.0 | Data manipulation |
| tqdm | >= 4.65.0 | Progress bars |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >= 7.4.0 | Test framework |
| pytest-cov | >= 4.1.0 | Coverage reporting |
| ruff | >= 0.1.0 | Linter and formatter |
| pre-commit | Latest | Git hooks |

### Tools & Infrastructure

- **CI/CD**: GitHub Actions
- **Code Coverage**: Codecov
- **Package Management**: pip with pyproject.toml
- **Version Control**: Git

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

### Test Structure

Tests are organized by module:

- `test_codebook.py` — Tests for sparse-routing codebook
- `test_completion_trainer.py` — Tests for completion training
- `test_dataset.py` — Tests for dataset handling
- `test_decoder.py` — Tests for modality decoders
- `test_downstream.py` — Tests for downstream recommender
- `test_download.py` — Tests for data download
- `test_end_to_end.py` — End-to-end integration tests
- `test_evaluator.py` — Tests for evaluation metrics
- `test_features.py` — Tests for feature extraction
- `test_gre_mc.py` — Tests for GRE-MC model
- `test_graph_builder.py` — Tests for graph construction
- `test_masking.py` — Tests for modality masking
- `test_metrics.py` — Tests for evaluation metrics
- `test_positional_encoding.py` — Tests for Laplacian PE
- `test_retrieval.py` — Tests for retrieval algorithms
- `test_transformer.py` — Tests for graph transformer
- `test_utils.py` — Tests for utility functions

## Code Style

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Linting is enforced with `ruff`:

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Code Quality Tools

- **ruff** — Fast Python linter and formatter
- **pytest** — Testing framework with coverage
- **pre-commit** — Git hooks for code quality
- **mypy** — Static type checking (optional)

## Documentation

- [Setup & Data Preparation](docs/SETUP.md)
- [Usage & Inference](docs/USAGE.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Fidelity Report](docs/FIDELITY_REPORT.md)
- [Changelog](CHANGELOG.md)

## Roadmap

- [x] Initial GRE-MC reproduction baseline
- [x] Comprehensive test suite
- [x] CI/CD pipeline
- [x] Documentation
- [ ] Pre-commit hooks
- [ ] Type hints coverage
- [ ] Performance benchmarks
- [ ] Additional datasets support
- [ ] Model export/import utilities
- [ ] Web demo interface
- [ ] Docker support
- [ ] Kubernetes deployment guides

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- How to fork and set up the project
- Branch naming conventions
- Commit message format
- Pull request process
- Coding standards
- Testing requirements

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Security

For security vulnerabilities, please see our [Security Policy](SECURITY.md).

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{li2026gremc,
  title={Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion},
  author={Li, Yuan and Hu, Jun and Jiang, Jiaxin and Hooi, Bryan and He, Bingsheng},
  journal={arXiv preprint arXiv:2605.00670},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Original paper authors for the GRE-MC algorithm
- Contributors and maintainers
- Open source community
