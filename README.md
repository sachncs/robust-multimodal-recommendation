# RMR: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion

[![CI](https://github.com/sachin/rmr/actions/workflows/ci.yml/badge.svg)](https://github.com/sachin/rmr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sachin/rmr/branch/main/graph/badge.svg)](https://codecov.io/gh/sachin/rmr)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reproduction of the paper **"Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion"** (GRE-MC) by Li et al. (NUS, 2026).

## Overview

RMR addresses missing modalities in multimodal recommender systems by:

1. **Modality-Aware Subgraph Retrieval** (anchor retrieval + ACS + MAGE) — retrieves semantically relevant subgraphs from the item-item graph.
2. **Joint-Encoding Graph Transformer** — encodes query nodes with retrieved context and Laplacian positional encodings.
3. **Sparse-Routing Codebook** — compresses latent representations via Gumbel-Softmax and Top-P selection.
4. **Modality Completion** — reconstructs missing modalities using per-modality MLP decoders.
5. **Downstream Recommendation** — LightGCN trained on the completed features.

> **Fidelity**: See `docs/FIDELITY_REPORT.md` for a section-by-section audit of this reproduction versus the paper. Every component is labeled EXACT, APPROXIMATE, or UNKNOWN, with deviations documented.

## Installation

```bash
# Clone the repository
git clone https://github.com/sachin/rmr.git
cd rmr

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Data Pipeline

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

## Training

```bash
# Stage 1: Train the modality completion model
python -m rmr.scripts.train_completion --data-dir data/processed --epochs 100

# Stage 2: Train the downstream recommender
python -m rmr.scripts.train_recommender --data-dir data/processed --epochs 100
```

## Evaluation

```bash
python -m rmr.scripts.evaluate --predictions results/scores.npy --labels results/labels.npy
```

## Quick Demo

Run a complete end-to-end pipeline on synthetic data:

```bash
python demo.py
```

## Architecture

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
│   └── gre_mc.py          # End-to-end GRE-MC model (retrieval + transformer + codebook + decoders)
├── training/          # Training loops
├── evaluation/        # Metrics (Recall@K, NDCG@K)
├── scripts/           # CLI training and evaluation scripts
└── tests/             # Comprehensive test suite
```

## Testing

Run the full test suite with coverage:

```bash
pytest tests/ -v --cov=rmr --cov-report=term-missing
```

## Code Style

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Linting is enforced with `ruff`:

```bash
ruff check .
```

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

## Documentation

- [Setup & Data Preparation](docs/SETUP.md)
- [Usage & Inference](docs/USAGE.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Fidelity Report](docs/FIDELITY_REPORT.md)
- [Changelog](CHANGELOG.md)

## License

MIT License — see [LICENSE](LICENSE) for details.
