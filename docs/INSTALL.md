# morel — Installation

```bash
pip install -e ".[dev]"          # library + dev tooling
pip install -e ".[dev,serve]"   # + FastAPI inference server
pip install -e ".[dev,bench]"   # + benchmark extras
```

## Python version

Python 3.10 or newer.

## Core dependencies

- `torch>=2.0,<3`
- `numpy>=1.24,<3`
- `scipy>=1.10,<2`
- `scikit-learn>=1.3,<2`
- `sentence-transformers>=2.2,<3`
- `torchvision>=0.15,<1`
- `Pillow>=10.0,<12`
- `pandas>=2.0,<3`
- `pydantic>=2.0,<3`
- `pyyaml>=6.0,<7`

## Optional extras

- `[serve]` — FastAPI, uvicorn, prometheus-client, httpx
- `[bench]` — pytest-benchmark
- `[text]` — sentence-transformers
- `[vision]` — torchvision

## Determinism

`morel.core.seed.seed(value)` configures deterministic seeding for torch,
torch CUDA, numpy, Python `random`, `PYTHONHASHSEED`, and cuDNN. The CI
matrix sets `cudnn.deterministic = True` and `cudnn.benchmark = False` by
default.

## CUDA

If `torch.cuda.is_available()`, `morel.core.device.device(None)` resolves
to CUDA. The CLI scripts accept `--device cuda` explicitly.
