# morel — Architecture

This document describes the package architecture and module layering.

## Layering

The package follows a one-way dependency layout (documented in the
dependency-graph diagram below). These rules are machine-checked on every test
run by `tests/unit/test_architecture.py`, which parses the import graph with
`ast` and asserts that each package imports only from strictly lower layers,
that the package graph is acyclic, and that `core/` imports nothing else from
the project. (An earlier `import-linter` configuration enforced the same rules
before that dependency was dropped; the stdlib test replaces it and needs no
extra tooling.)

Adding a new subpackage requires listing it in the `LAYERS` table in that test
— an unlisted package fails the suite, so the layering cannot drift silently.

```
cli/        (argparse dispatch; thin shim to app/)
  ↓
app/        (orchestration services: experiment, benchmark, reproduce)
  ↓
train/  eval/    (training & evaluation)
  ↓
model/  pipeline/  (algorithms; pipeline composes the 5 stages)
  ↓
data/    graph/  retrieve/  encode/  route/  codebook/  complete/  recommend/
  ↓
core/    (config, seed, device, logging, errors, types, paths, fidelity)
```

`core/` imports nothing from the project. `data/` imports only `core/`. The model subpackages import only `data/` and `core/`. `train/` and `eval/` import `model/`, `data/`, and `core/`. `app/` orchestrates `train/` and `eval/`. `cli/` is pure dispatch.

## Module map

| Package | Responsibility |
|---------|---------------|
| `morel.core` | Configuration, seeding, device, logging, errors, types, paths, fidelity registry |
| `morel.data` | Lifecycle: acquire → validate → extract → build → mask → store |
| `morel.graph` | `Graph` Protocol, `Bipartite`, `Item`, `Subgraph`, `Laplace` PE |
| `morel.retrieve` | `Retriever` Protocol, `Anchor`, `ACS`, `MAGE`, `Pipeline` |
| `morel.encode` | `Encoder` Protocol, `Transformer`, attention/mean/token pools, baselines |
| `morel.route` | `Router` Protocol, `Dense`, `Top` (top-k), `Gumbel`, `Fixed` |
| `morel.codebook` | `Codebook` Protocol, `VQ`, `GumbelVQ`, `usage`, `balance` losses |
| `morel.complete` | `Decoder` Protocol, `Decoders` with learned [MASK] token, per-modality heads |
| `morel.recommend` | `Recommender` Protocol, `Light` (LightGCN), `MF`, `Pop`, `BPR` |
| `morel.pipeline` | End-to-end `Pipeline` composing the 5 stages |
| `morel.train` | `Trainer` ABC, `Completion`, `Recommendation`, `Checkpoint`, `Monitor`, losses |
| `morel.eval` | `Metric` Protocol, ranking metrics, completion metrics, robustness/ablation protocols |
| `morel.app` | `Experiment`, `Benchmark`, `Reproduce` services |
| `morel.cli` | Top-level `morel` entry point dispatching to subcommands |
| `morel.serve` | FastAPI inference server with bearer-token auth |

## Data flow

```
raw text + images
  → morel.data.acquire.fetch   (HTTPS, retries, SHA256)
  → morel.data.extract         (Sentence-Transformers, ResNet-50)
  → morel.data.build           (bipartite + item-item cooccurrence; iterative k-core)
  → morel.data.mask            (Bernoulli, Block, Structured)
  → morel.data.store           (npz + manifest sidecar)
  → morel.retrieve             (Anchor + ACS + MAGE)
  → morel.graph.Laplace        (bottom-k eigenvectors)
  → morel.encode.Transformer   (joint graph transformer)
  → morel.route.Top            (top-k sparse routing)
  → morel.codebook.GumbelVQ    (Gumbel-Softmax + lookup)
  → morel.complete.Decoders    (per-modality MLP)
  → morel.recommend.Light      (LightGCN ranking)
  → morel.eval.*               (Recall@K, NDCG@K, robustness sweep)
```

## Configuration

A single `Config` dataclass tree at `morel.core.config.Config` is the source of truth. Precedence is `CLI > env > YAML > default`. Every experiment produces a `Manifest` sidecar carrying dataset, version, code hash, seed, extractor, and config hash. Resuming a run requires the config hash to match.

## Extension points

Each pipeline stage is selected by a `kind` in the config and built through a
registry. The registries are `morel.encode.ENCODERS`, `morel.route.ROUTERS`,
`morel.codebook.CODEBOOKS`, `morel.complete.COMPLETERS` and
`morel.recommend.RECOMMENDERS`; each is a `morel.core.registry.Registry`.

| Config field | Registry | Shipped kinds |
|---|---|---|
| `encode.kind` | `ENCODERS` | `transformer`, `identity` |
| `route.kind` | `ROUTERS` | `top`, `dense`, `gumbel`, `fixed` |
| `codebook.kind` | `CODEBOOKS` | `gumbel`, `vq`, `identity` |
| `complete.kind` | `COMPLETERS` | `mlp` |
| `recommend.kind` | `RECOMMENDERS` | `light`, `mf`, `pop` |

Adding an implementation does not require editing morel. Register a factory
under a new name and select it from config:

```python
from morel.codebook import CODEBOOKS


@CODEBOOKS.register("my-codebook")
def build_mine(*, dim, size, router, seed=None):
    return MyCodebook(dim, size)
```

```yaml
codebook:
  kind: my-codebook
```

A factory takes keyword arguments only, and takes the full set its stage
supplies even if it ignores some of them, so that all implementations of a
stage are interchangeable. An unregistered `kind` raises `ConfigError` listing
the names that are available; registering an existing name raises unless
`replace=True`.

`tests/unit/pipeline/test_component_selection.py` checks that every shipped
combination runs end to end and that a component defined outside the package
is selectable.

## Determinism

`morel.core.seed.seed(value)` sets `torch`, `torch.cuda`, `numpy`, `random`, `PYTHONHASHSEED`, `cudnn.deterministic`, and `cudnn.benchmark` process-wide. `morel.core.seed.deterministic(value)` is the scoped form: it seeds for the duration of a block and restores the caller's RNG state on exit, which is how model constructors are made reproducible without perturbing the surrounding program. `state()` and `restore()` snapshot RNG state for resume.

Component construction is seeded from `config.seed`, so two pipelines built from one config hold identical weights regardless of ambient RNG state. `Pipeline.forward(training=False)` switches the module into eval mode for the call, so an inference pass applies no dropout. `morel.graph.laplacian.pe` pins ARPACK's start vector, sorts eigenvalues, and canonicalises eigenvector signs and the basis of degenerate eigenspaces; graphs up to `DENSE_MAX_NODES` use the direct dense solver, which is bitwise reproducible.

## Observability

`morel.core.log.configure(...)` initializes structured JSON logging. `Monitor` writes JSONL metric lines to `runs/<run_id>/metrics.jsonl`. The `Fidelity` registry renders `FIDELITY.md` and `FIDELITY.json` from component-level entries.
