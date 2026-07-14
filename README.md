<p align="center">
  <h1 align="center">RMR</h1>
  <p align="center">Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/robust-multimodal-recommendation/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/robust-multimodal-recommendation/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/rmr/"><img src="https://img.shields.io/pypi/v/rmr" alt="PyPI"></a>
    <a href="https://github.com/sachncs/robust-multimodal-recommendation/stargazers"><img src="https://img.shields.io/github/stars/sachncs/robust-multimodal-recommendation" alt="Stars"></a>
  </p>
</p>

**Reproduction of "Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion" (GRE-MC) — Li et al., NUS, 2026.**

RMR addresses missing modalities in multimodal recommender systems through five stages: (1) modality-aware subgraph retrieval from the item-item graph (anchor retrieval + ACS + MAGE), (2) joint-encoding graph transformer with Laplacian positional encodings, (3) sparse-routing codebook (Gumbel-Softmax + Top-P selection), (4) per-modality MLP completion of missing modalities, and (5) downstream LightGCN trained on the completed features.

> **Fidelity**: See `docs/FIDELITY_REPORT.md` for a section-by-section audit of this reproduction versus the paper. Every component is labeled EXACT, APPROXIMATE, or UNKNOWN, with deviations documented.

## Features

- **Modality-aware subgraph retrieval** — Retrieves semantically relevant subgraphs from the item-item graph using anchor retrieval, ACS, and MAGE algorithms
- **Joint-encoding graph transformer** — Encodes query nodes with retrieved context and Laplacian positional encodings
- **Sparse-routing codebook** — Compresses latent representations via Gumbel-Softmax and Top-P selection
- **Modality completion** — Reconstructs missing modalities using per-modality MLP decoders
- **Downstream recommendation** — LightGCN trained on the completed features
- **Comprehensive test suite** — Full test coverage with pytest and continuous integration
- **Modular design** — Clean, maintainable codebase with clear separation of concerns

## Installation

### From PyPI

```bash
pip install rmr
```

### From source

```bash
git clone https://github.com/sachncs/robust-multimodal-recommendation.git
cd robust-multimodal-recommendation
pip install -e ".[dev]"
```

For a minimal install without dev tooling: `pip install .`

## Quick Start

### CLI

```bash
python demo.py                                # end-to-end pipeline on synthetic data
python -m rmr.data.download                   # Amazon 5-core review + metadata
python -m rmr.data.features                   # text + visual feature extraction
python -m rmr.data.graph_builder              # user-item + item-item graphs
python -m rmr.data.masking                    # random modality masking (40%)
python -m rmr.scripts.train_completion --data-dir data/processed --epochs 100
python -m rmr.scripts.train_recommender --data-dir data/processed --epochs 100
python -m rmr.scripts.evaluate --predictions results/scores.npy --labels results/labels.npy
```

### Python API

```python
import torch
from rmr.models import GREMC, ModalityConfig
from rmr.training import CompletionTrainer, RecommendationTrainer

# Stage 1 — modality completion
model = GREMC(modality_config=ModalityConfig.from_env())
completion_trainer = CompletionTrainer(model, data_dir="data/processed")
completion_trainer.train(epochs=100)

# Stage 2 — downstream recommender on the completed features
rec_trainer = RecommendationTrainer(model.recommender, data_dir="data/processed")
rec_trainer.train(epochs=100)
```

## Configuration

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `DEVICE` | yes | `cpu` | Training device (`cpu` / `cuda`) |
| `DATA_PROCESSED_DIR` | yes | `data/processed` | Path to processed data |
| `LOG_LEVEL` | yes | `INFO` | Logging level |

Copy `.env.example` to `.env` and edit; `pyproject.toml` and `.editorconfig` carry tool configuration.

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `rmr.models.GREMC` | class | End-to-end GRE-MC model |
| `rmr.models.retrieval` | module | Anchor retrieval, ACS, MAGE algorithms |
| `rmr.models.positional_encoding` | module | Laplacian PE |
| `rmr.models.transformer` | module | Joint-encoding graph transformer |
| `rmr.models.codebook` | module | Sparse-routing codebook |
| `rmr.models.decoder` | module | Per-modality decoders |
| `rmr.models.downstream` | module | LightGCN recommender |
| `rmr.data.download` / `features` / `graph_builder` / `masking` | modules | Data pipeline |
| `rmr.training.CompletionTrainer` | class | Stage-1 trainer |
| `rmr.training.RecommendationTrainer` | class | Stage-2 trainer |
| `rmr.evaluation` | module | Recall@K, NDCG@K metrics |
| `rmr.scripts.train_completion` / `train_recommender` / `evaluate` | CLIs | Pipeline entry points |

## Examples

A complete end-to-end pipeline (synthetic data, no Amazon download):

```bash
python demo.py
```

Stage 1 + Stage 2 on real Amazon data:

```bash
python -m rmr.data.download
python -m rmr.data.features
python -m rmr.data.graph_builder
python -m rmr.data.masking
python -m rmr.scripts.train_completion --data-dir data/processed --epochs 100
python -m rmr.scripts.train_recommender --data-dir data/processed --epochs 100
python -m rmr.scripts.evaluate --predictions results/scores.npy --labels results/labels.npy
```

## Project Structure

```
rmr/
├── data/                       # Data pipeline (download, features, graphs, masking)
├── models/                     # Core model components
│   ├── retrieval.py            #   - Anchor retrieval + ACS + MAGE algorithms
│   ├── positional_encoding.py  #   - Laplacian PE
│   ├── transformer.py          #   - Joint-encoding graph transformer
│   ├── codebook.py             #   - Sparse-routing codebook
│   ├── decoder.py              #   - Modality-specific decoders
│   ├── downstream.py           #   - LightGCN recommender
│   └── gre_mc.py               #   - End-to-end GRE-MC model
├── training/                   # Training loops
├── evaluation/                 # Metrics (Recall@K, NDCG@K)
├── scripts/                    # CLI training and evaluation scripts
├── tests/                      # Comprehensive test suite (one file per module)
├── docs/                       # SETUP, USAGE, ARCHITECTURE, FIDELITY_REPORT
├── demo.py                     # End-to-end pipeline on synthetic data
└── pyproject.toml
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=rmr --cov-report=term-missing
ruff check .
ruff format .
mypy rmr/
pre-commit install
```

## Testing

```bash
pytest tests/ -v                                      # full suite with coverage
pytest tests/test_codebook.py -v                      # one module
pytest tests/test_codebook.py::TestCodebook::test_forward -v  # one test
```

Test suite covers: codebook, completion trainer, dataset, decoder, downstream recommender, download, end-to-end, evaluator, features, GRE-MC model, graph builder, masking, metrics, Laplacian PE, retrieval, graph transformer, and utility helpers.

## Build

```bash
pip install build
python -m build
```

## Release

```bash
pytest && ruff check . && mypy rmr/
git tag vX.Y.Z && git push origin vX.Y.Z
# .github/workflows/release.yml publishes to PyPI via trusted publishing
```

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python ≥ 3.10 |
| Deep learning | PyTorch ≥ 2.0 |
| Numerical | NumPy ≥ 1.24, SciPy ≥ 1.10 |
| ML utilities | scikit-learn ≥ 1.3 |
| Text embeddings | sentence-transformers ≥ 2.2 |
| Vision features | torchvision ≥ 0.15 (ResNet) |
| Data | pandas ≥ 2.0, tqdm ≥ 4.65 |
| Lint / format | ruff |
| Tests | pytest, pytest-cov |
| Hooks | pre-commit |
| CI/CD | GitHub Actions |
| Coverage | Codecov |

## Roadmap

- **v0.3.0** — Current: GRE-MC reproduction baseline, comprehensive test suite, CI/CD, documentation.
- **v0.4.0** — Planned: pre-commit hooks, expanded type-hint coverage, performance benchmarks.
- **v0.5.0** — Planned: additional dataset support, model export/import utilities, web demo interface.
- **v1.0.0** — Planned: Docker support, Kubernetes deployment guides.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

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

[MIT](LICENSE) © 2026 Sachin

## Acknowledgments

- Original paper authors for the GRE-MC algorithm
- Contributors and maintainers
- Open source community
