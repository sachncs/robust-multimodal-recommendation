# morel

**Modality-aware recommendation via graph retrieval-enhanced completion.**

Production-grade, research-grade reproduction of "Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion" (GRE-MC).

The repo implements a multimodal recommendation pipeline that handles missing modalities through graph-conditioned codebook completion and a downstream LightGCN ranker.

## Status

- **Version:** `0.1.0` (rebrand from `rmr`)
- **Python:** `>=3.10`
- **PyTorch:** `>=2.0,<3`

## Install

```bash
pip install -e ".[dev]"
```

For the inference service:

```bash
pip install -e ".[dev,serve]"
```

## Quick start

```python
import morel
from morel import Config, Pipeline, seed

seed(0)
config = Config.default()
pipeline = Pipeline(config)
pipeline.run(items=[0, 1, 2], mask=None)
```

## CLI

```bash
morel data download --category Beauty
morel data extract
morel data build
morel data mask
morel train completion
morel train recommendation
morel eval rank
morel eval robustness
morel reproduce configs/reproduce.yaml
morel serve --host 0.0.0.0 --port 8080
```

## Development

```bash
make install      # install with dev extras
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy
make test         # pytest with coverage
make bench        # pytest-benchmark
make reproduce    # end-to-end reproduction on synthetic data
```

## Documentation

See `docs/`:

- `ARCHITECTURE.md` — module layering, data flow
- `METHOD.md` — mathematical pipeline
- `FIDELITY.md` — paper-fidelity report (machine-rendered)
- `REPRODUCE.md` — reproduction workflow
- `LIMITATIONS.md` — known scope boundaries
- `API.md` — public API reference

## License

MIT © Sachin
