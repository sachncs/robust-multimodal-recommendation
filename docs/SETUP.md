# Setup

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- sentence-transformers, torchvision, pandas, tqdm

## Installation

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode with development dependencies
(pytest, pytest-cov, ruff).

## Data Preparation

```bash
# Download Amazon review & metadata
python -m rmr.data.download

# Extract text and visual features
python -m rmr.data.features

# Build graphs
python -m rmr.data.graph_builder

# Apply modality masking (default 40%)
python -m rmr.data.masking
```

All processed files are written to `data/processed/`.
