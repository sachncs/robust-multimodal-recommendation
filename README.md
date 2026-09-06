<p align="center">
  <h1 align="center">morel</h1>
  <p align="center">A Python package for robust multimodal recommendation via graph retrieval-enhanced modality completion.</p>
  <p align="center">
    <a href="#before-you-start"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/morel/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/morel/ci.yml" alt="CI"></a>
    <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="mypy"></a>
  </p>
</p>

---

## What is this?

morel is a Python implementation of the GRE-MC architecture — "Robust
Multimodal Recommendation via Graph Retrieval-Enhanced Modality
Completion". It answers one question:

> *"Given user–item interactions where some items are missing part of
> their data, which items should we recommend to each user?"*

You feed it a user–item interaction graph and per-item text/visual
features (which may be partially missing). It retrieves related items
from the graph, **completes** the missing modalities through a
graph-conditioned codebook, and ranks items with a LightGCN-style
recommender.

It implements the architecture from a paper
([CITATION.cff](CITATION.cff)). You don't need to read the paper to use
the package — the [demo](examples/end_to_end_demo.py) runs on synthetic
data in seconds.

---

## Who is this for?

You, even if:

- You've never run a recommendation experiment before.
- You don't know what "modality completion" or "codebook" means.
- You don't know what LightGCN is.

If you can install Python and type commands into a terminal, you can
run morel. When the docs use a word you don't know, look it up in
[docs/METHOD.md](docs/METHOD.md) or
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

If you've used Python before, you'll be productive in five minutes.

---

## What can it do?

- **Data lifecycle CLI** — `download`, `extract`, `build`, `mask`,
  `verify` over the raw dataset. ([morel.data](morel/data))
- **Graph retrieval** — BFS, anchor, ACS, and MAGE retrieval over the
  item co-occurrence graph. ([morel.retrieve](morel/retrieve))
- **Missing-modality completion** — transformer encoding plus VQ /
  Gumbel-VQ codebooks with usage and balance losses.
  ([morel.complete](morel/complete))
- **Recommendation** — BPR training and a LightGCN-style ranker.
  ([morel.recommend](morel/recommend))
- **Evaluation** — recall@k / ndcg@k, robustness sweeps, and
  ablations. ([morel.eval](morel/eval))
- **Reproducibility** — deterministic seeding and a machine-rendered
  fidelity report tying implementations to paper equations.
  ([docs/FIDELITY.md](docs/FIDELITY.md))
- **Inference server** — FastAPI service with token auth, a cached
  model loader, and Docker support. ([docs/DEPLOY.md](docs/DEPLOY.md))

---

## Before you start

You'll need **Python 3.10 or newer** installed on your computer.

If you don't know what Python is or whether you have it:

1. Open a terminal (on macOS: `Cmd + Space`, type "Terminal"; on
   Windows: open "PowerShell"; on Linux: open your usual terminal).
2. Type `python3 --version` and press Enter.
3. If you see a version number starting with `3.10` or newer, you're
   set.
4. If you see "command not found" or an older version, follow the
   [official Python installer guide](https://realpython.com/installing-python/).

You'll also need **git** (a tool for downloading code). Same drill:
type `git --version` in your terminal.

---

## Installation

Pick whichever option fits your setup.

### Option 1 — From source (recommended for development)

A "virtual environment" is an isolated Python sandbox that keeps this
package's dependencies from interfering with your other Python
projects.

```bash
# 1. Download the code
git clone https://github.com/sachncs/morel.git
cd morel

# 2. Make a sandbox for it
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows (PowerShell)

# 3. Install morel and its dev tools
pip install -e '.[dev]'
```

> 💡 **The dot in `.[dev]` is intentional.** It means "install this
> package and also the dev extras." The square brackets are part of
> the command, not punctuation.

After this, your terminal prompt will probably have `(.venv)` at the
front — that tells you the sandbox is active. To leave the sandbox
later, type `deactivate`.

### Option 2 — With the inference server

```bash
pip install -e '.[dev,serve]'
```

### Option 3 — Docker (no Python install needed)

```bash
docker compose up --build
```

This builds the `morel:latest` image and starts the inference service
on port `8080` (health-checked at `/health`). See
[Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml).

---

## Your first run — the command line

The fastest way to see morel work. No dataset required:

```bash
python examples/end_to_end_demo.py
```

The script builds a small synthetic user–item graph, runs the
completion stage, and evaluates the recommender. You'll see output
like:

```
Reconstructed visual shape: (50, 16)
Routing weights shape: (50, 100)
Score matrix shape: (20, 50)
  recall@10: 0.6145
  ndcg@10: 0.6364
```

Once installed, the `morel` command drives the rest of the pipeline:

```bash
python -m morel              # prints the command tree
python -m morel data --help  # data lifecycle: download, extract, build, mask, verify
python -m morel serve        # start the inference server
```

The top-level command tree is:

```
usage: morel [-h] {data,train,eval,bench,reproduce,serve,render-fidelity} ...
```

For a walkthrough of every line of the demo, see
[examples/README.md](examples/README.md).

---

## Your first run — Python

Open a Python interpreter and try the same pipeline step by step:

```python
import numpy as np
import torch

from morel.core.config import Config
from morel.data.build import bipartite, item_cooccurrence
from morel.data.mask import bernoulli
from morel.pipeline import Pipeline
from morel.recommend import Light

# Synthetic data: 200 interactions between 20 users and 50 items.
rng = np.random.default_rng(0)
users, items = 20, 50
ui = bipartite(
    rng.integers(0, users, size=200),
    rng.integers(0, items, size=200),
    users,
    items,
)
adjacency = item_cooccurrence(ui)

# Per-item features, with 40% of modalities masked out.
features = {
    "visual": rng.normal(size=(items, 16)).astype(np.float32),
    "text": rng.normal(size=(items, 8)).astype(np.float32),
}
mask = bernoulli(items, 2, 0.4, seed=42).to_numpy()

# Build the GRE-MC pipeline and attach the corpus.
pipeline = Pipeline(Config(), dims={"visual": 16, "text": 8})
pipeline.attach_corpus(features, mask, adjacency)

# Complete the missing modalities and route through the codebook.
output = pipeline(
    {k: torch.from_numpy(v) for k, v in features.items()},
    torch.from_numpy(mask),
    adjacency,
    index=torch.arange(items),
    training=False,
)
print(output.completed["visual"].shape)  # (50, 16)
print(output.routing.shape)  # (50, 100)

# Rank users against items with the LightGCN-style recommender.
scores = Light(users=users, items=items, embed=32, layers=2)(
    torch.arange(users), torch.arange(items), ui
)
print(scores.shape)  # (20, 50)
```

That first run covers the whole GRE-MC flow: build the graph, retrieve
context, complete the missing modality, and rank. The full walkthrough
of the same code lives in [examples/end_to_end_demo.py](examples/end_to_end_demo.py).

---

## Configuration

Want to change something? morel's configuration is a tree of frozen
dataclasses, built with `Config.defaults()` and overridable field by
field:

```python
from morel import Config

config = Config.defaults()
config = Config.from_dict({...})
config = Config.from_yaml("path/to/config.yaml")
```

The defaults look like this (abridged):

```json
{
  "seed": 42,
  "device": "auto",
  "data":    { "raw": "data/raw", "processed": "data/processed", "category": "Beauty", "min": 5 },
  "encoder": { "text": "sentence-transformers/all-MiniLM-L6-v2", "visual": "resnet50", "text_dim": 384, "visual_dim": 2048 },
  "masking": { "kind": "bernoulli", "ratio": 0.4 },
  "retrieve": { "kind": "mage", "anchors": 10, "iters": 10 },
  "encode":   { "kind": "transformer", "hidden": 128, "layers": 2, "heads": 4 },
  "route":    { "kind": "top", "p": 4, "tau": 0.5 },
  "codebook": { "kind": "gumbel", "size": 100 },
  "complete": { "kind": "mlp", "hidden": 128 },
  "recommend": { "kind": "light", "embed": 64, "layers": 3 },
  "completion": { "epochs": 100, "batch": 512, "lr": 0.001, "usage": 1.0, "balance": 1.0 },
  "recommendation": { "epochs": 100, "batch": 1024, "lr": 0.001, "negatives": 1 },
  "eval": { "ks": [10, 20], "robustness": [0.1, ... 0.9], "ablations": ["no_retrieval", "no_pe", "no_codebook"] },
  "serve": { "host": "0.0.0.0", "port": 8080, "workers": 1, "auth": false },
  "log":   { "level": "INFO", "structured": true, "directory": "runs" }
}
```

What each field means:

| Field | Plain English |
|---|---|
| `data.category` | Which dataset category to use (default `Beauty`). |
| `masking.ratio` | How much of each modality to mask during training. |
| `retrieve.kind` | Retrieval strategy (`mage`, `bfs`, `anchor`, `acs`). |
| `route.kind` | Soft-routing strategy for the codebook (`top`, `dense`, `gumbel`). |
| `completion.usage` | Weight on the codebook-usage loss. |
| `completion.balance` | Weight on the load-balancing loss. |

---

## Where to go next

- **[docs/API.md](docs/API.md)** — the public API reference.
- **[docs/METHOD.md](docs/METHOD.md)** — the math behind the pipeline, step by step.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — module layout and data flow.
- **[docs/REPRODUCE.md](docs/REPRODUCE.md)** — full reproduction workflow on real data.
- **[docs/FIDELITY.md](docs/FIDELITY.md)** — does the code match the paper?
- **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** — known scope boundaries.
- **[examples/](examples)** — runnable demos.
- **[PLAN.md](PLAN.md)** — implementation history and phase notes.

For maintainers:

- **[docs/INSTALL.md](docs/INSTALL.md)** — environment setup.
- **[docs/DEPLOY.md](docs/DEPLOY.md)** — run the inference server.
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** — benchmark results.
- **[docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md)** — production assessment.
- **[Makefile](Makefile)** — `make lint`, `make typecheck`, `make test`, `make bench`, `make serve`, and more.

---

## Contributing

Want to improve morel? See [CONTRIBUTING.md](CONTRIBUTING.md) for how
to set up a development environment and submit changes.

## Code of Conduct

We expect everyone to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Found a security issue? See [SECURITY.md](SECURITY.md) — please don't
open a public GitHub issue for security problems.

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, ship it.