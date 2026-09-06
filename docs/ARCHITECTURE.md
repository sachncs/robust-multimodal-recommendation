# morel — Architecture

This document describes the package architecture and module layering.

## Layering

The package follows a one-way dependency layout (documented in the
dependency-graph diagram below). A former `import-linter` enforcement of these
rules was removed; the diagram is now descriptive rather than machine-checked.

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

## Determinism

`morel.core.seed.seed(value)` is the single entry point that sets `torch`, `torch.cuda`, `numpy`, `random`, `PYTHONHASHSEED`, `cudnn.deterministic`, and `cudnn.benchmark`. Every CLI calls it at startup. `state()` and `restore()` snapshot RNG state for resume.

## Observability

`morel.core.log.configure(...)` initializes structured JSON logging. `Monitor` writes JSONL metric lines to `runs/<run_id>/metrics.jsonl`. The `Fidelity` registry renders `FIDELITY.md` and `FIDELITY.json` from component-level entries.
