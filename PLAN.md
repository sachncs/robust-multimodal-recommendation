# morel — End-to-End Refactor Plan

> **Status:** Approved by repository owner. Execution proceeds phase by phase; each phase
> keeps the verification gates green (`ruff check`, `ruff format --check`, `mypy morel/`,
> `pytest -q`, `morel render-fidelity`) before the next phase begins.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Map (current state)](#architecture-map-current-state)
3. [Mathematical Pipeline](#mathematical-pipeline)
4. [Dependency Map](#dependency-map)
5. [Public API Map (current vs. claimed)](#public-api-map-current-vs-claimed)
6. [Data Pipeline (current state)](#data-pipeline-current-state)
7. [Polymorphism / Class Design (current)](#polymorphism--class-design-current)
8. [Correctness Risks (P0–P5)](#correctness-risks-p0p5)
9. [Technical Debt Register](#technical-debt-register)
10. [Refactor Plan (sequenced)](#refactor-plan-sequenced)
11. [Atomic Change List (62 items)](#atomic-change-list)
12. [Verification Gates](#verification-gates)
13. [Final Deliverable Table](#final-deliverable-table)
14. [Risks and Tradeoffs](#risks-and-tradeoffs)
15. [Out of Scope (explicit limitations)](#out-of-scope-explicit-limitations)

---

## Executive Summary

The repository has been substantially restructured (12 claimed phases, 228 tests
passing, docs/import-linter contracts in place). The audit however reveals that
`docs/PRODUCTION_READINESS.md` and the API/METHOD/FIDELITY docs claim more than
the implementation delivers, and several P0 correctness/reliability defects are
masked by tests that use a `_Standin` fake module instead of the real `Pipeline`.

The refactor is therefore **substantive, not cosmetic**: fix the false claims
first, then tighten the architecture around real polymorphism boundaries, then
sweep performance and docs, then add distributed training, true streaming
ingestion, and full-pipeline online update in the serve stack.

---

## Architecture Map (current state)

Layering is enforced by `pyproject.toml [tool.importlinter]` and looks correct on
paper:

```
core  ←  data  ←  {graph, retrieve, encode, route, codebook, complete, recommend}  ←  pipeline  ←  {train, eval}  ←  app  ←  cli  +  serve (orthogonal)
```

`core/` imports nothing from the project. `data/` imports only `core/`. The model
subpackages import only `data/` and `core/`. `train/` and `eval/` import `model/`,
`data/`, and `core/`. `app/` orchestrates `train/` and `eval/`. `cli/` is pure
dispatch.

---

## Mathematical Pipeline

```
features{m} + mask  ─┐
item-item graph      ├─► retrieve (anchor+ACS+MAGE) ─► encoder (transformer+PE) ─► router (top-k Gumbel) ─► codebook (lookup) ─► decoder (per-modality MLP) ─► completed{m}
ui graph             ─┘                                                                                                └─► LightGCN ranker ─► ranking ─► eval
```

Module-internal contracts are weak: return types loosely specified, side effects
scattered. The refactor tightens these contracts in Phase 3.

---

## Dependency Map

- `pipeline` imports `codebook, complete, core.config, encode, graph, recommend, retrieve, route` — fine.
- `train.completion` and `train.recommendation` import `core.config` and `recommend` — fine.
- `app.experiment` imports only `core.config, core.log, core.seed` — fine; its `run()` does nothing today (P0).
- `cli` imports `core.log`, dispatches to `data.__main__`, `app.experiment`, `serve.app` — fine.
- The implementation-linter contracts in `pyproject.toml` are reasonable.

---

## Public API Map (current vs. claimed)

| Claim (README / API.md) | Reality |
|---|---|
| `from morel import Config, Pipeline, seed` | **FAILS** — `morel/__init__.py` only re-exports `__version__` |
| `morel.train.Checkpoint, Loss, BPR, Composite` | Only `State` exists; `Checkpoint` doesn't |
| `morel.eval.Recall, NDCG, Precision, MAP, MRR` | None exist; only `recall_at_k`, `ndcg_at_k`, etc. |
| `morel train completion`, `morel eval rank`, `morel reproduce ...` | **STUBS** printing "command registered, full implementation lands in Phase 12" |
| `morel render-fidelity` | **DOES NOT EXIST** (Makefile target references it) |
| `morel data extract`, `morel data build` | **STUBS** printing placeholder messages |
| `make reproduce` | Calls the stubbed `reproduce` subcommand and `configs/reproduce.yaml` does not exist |

---

## Data Pipeline (current state)

`acquire → validate → extract → build → mask → store` is structured and reasonably
clean. Concrete issues:
- `data/acquire.download` points at the legacy McAuley UCSD URL
  `https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/{Beauty}_5.json.gz`.
  That URL is no longer the canonical source (Amazon-Reviews-2023 supersedes it);
  downloads may silently 404. **P1.**
- `data/__main__.py` extract/build are stubs; mask works; verify works.
- Manifest sidecars are sound.
- `bipartite/interactions` from raw JSON work end-to-end.

---

## Polymorphism / Class Design (current)

| Concept | Polymorphism today | Status |
|---|---|---|
| Retriever | three free functions: `anchor.query`, `acs.compute`, `mage.expand` | acceptable (no shared state) |
| Router | `Router` nn.Module ABC + `Dense, Top, Gumbel, Fixed` | good |
| Encoder | ambiguous: `data.extract.Encoder` Protocol (image-text feature extractor) and `encode.baseline.Encoder` Protocol (graph encoder) share name and are unrelated | **P2**: same name, two meanings |
| Baseline encoder | `Baseline(kind=...)` string dispatch | OK but string-keyed, not subclass |
| Codebook | `VQ` and `GumbelVQ` with **no shared base**; downstream code special-cases `model.codebook.usage/.balance` | **P2** |
| Decoder | only `Decoders` (always a set of MLPs) | acceptable |
| Recommender | `Light, MF, Pop` with **no shared base**; `Recommendation.trainer` calls `self.model(users, items, ui_graph)` which happens to fit all three by signature coincidence | **P2** |
| Mask | `core.types.Mask` Protocol + `data.mask.Mask` dataclass + `core.config.Mask` (config section) — three things named `Mask` | **P2** |
| Mask Spec | `Spec` Protocol in `data/mask.py` | unused by `data/__main__.py` |

---

## Correctness Risks (P0–P5)

### P0 — critical correctness

1. **`train.completion.Completion.step()` unpacks `Pipeline` output as a tuple.**
   - `Pipeline.forward()` returns `Output(completed, routing, …)` (a frozen dataclass).
   - `Completion.step` line 70: `predictions, probs = self.model(features, mask, batch["adjacency"], …)` — raises `TypeError: cannot unpack non-iterable Output object`.
   - `validate` line 94 has the same bug.
   - **Why masked**: unit/integration tests use a hand-rolled `_Standin` returning `(dict, tensor)`. The real `Pipeline` is never driven through `Completion` in tests.
   - **Reproduction**: confirmed by running `Completion.step` against the real `Pipeline`.

2. **README Quick Start broken.** `from morel import Config, Pipeline, seed` raises `ImportError`. `morel/__init__.py` only re-exports `__version__`.

3. **CLI commands `train`, `eval`, `bench`, `reproduce` are stubs.** They print placeholder messages. `make reproduce` therefore no-ops.

4. **`app.experiment.Experiment.run()` and `Benchmark.run()` are no-ops.** `Experiment.run()` creates `run_dir`, sets seed, returns `{"duration": …}`. `Benchmark.run()` returns `{"results": {}}`.

5. **`data/__main__.py` `extract` and `build` subcommands are stubs.**

6. **`torch.load(..., weights_only=False)` in `train/checkpoint.py:50` and `serve/loader.py:64`.** Allows arbitrary pickle deserialization. Switch to `weights_only=True` and provide a typed whitelist, OR introduce a non-pickle serialization path.

7. **`encode/layer.py` documents Pre-LN but implements Post-LN.** Docstring says "Pre-LN"; code applies `attn(hidden)` → residual → `norm1` (Post-LN).

8. **`Pipeline._encode_subgraph` will hit `GraphError` on `np.eye()` adjacency (self-loops).** `laplacian()` rejects self-loops via invariants. The benchmark `benchmarks/end_to_end.py` constructs `np.eye(items)` and immediately fails.

### P1 — production reliability

9. **`Trainer.__init__` ignores `config.device`.** Line 40 hardcodes `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. `amp: bool = False` is accepted but never used.

10. **`encode/pool.py::Attention` produces NaN when an entire row is masked.** `scores.masked_fill(~mask, -inf)` → `softmax(-inf)` → NaN.

11. **`docs/API.md` lies** about `Checkpoint`, `Recall`, `NDCG`, `Precision`, `MAP`, `MRR` (none of these exist).

12. **`data/acquire.download` URL** is the legacy McAuley UCSD Amazon URL; the dataset moved.

13. **`Light._adj_cache = (id(ui_graph), tensor)`** — comment claims content hashing, code uses `id()`. After GC the id can be reused and a stale tensor returned.

14. **`Trainer.fit` epoch arithmetic wrong.** Resumes loop is `range(start_epoch, epochs)` instead of `range(start_epoch, start_epoch + epochs)`.

### P2 — architecture

15. **Three different `Mask` types** (config section dataclass, value dataclass, Protocol).
16. **`Encoder` name collision** between data-side feature encoder and encode-side graph encoder.
17. **`Codebook` has no shared base**, yet `Pipeline.codebook` is consumed via duck typing.
18. **`Pipeline.register_buffers`** shadows `nn.Module.register_buffers` and does NOT register PyTorch buffers.
19. **`pipeline.pipeline` module name duplicates class name**.
20. **`bpr.negatives` per-user Python loop** with `np.setdiff1d` per user.
21. **`retrieve/relevance.mean_relevance` per-candidate Python loop** with redundant norm computations.
22. **`retrieve.pipeline.batch` is a Python loop over queries**.
23. **`Pipeline._encode_subgraph` is a per-item Python loop**.
24. **`eval.ranking.mrr` uses `np.argsort`**.

### P3 — performance

25. **`bpr.negatives`** could be one matrix operation per user.
26. **`relevance.mean_relevance`** should batch the cosine similarity.
27. **`Light.forward` rebuilds `torch.sparse.mm` indices** every `id()` change.

### P4 — maintainability

28. **182 ruff violations** (mostly `D406/D407`, `F401`, `E501`).
29. **`del dim`, `del hidden`, `del ui_graph`** in three functions.
30. **MyPy currently blocked** by a numpy 3.12 `type` statement issue.
31. **`Output` dataclass** has no documented gradient contract.

### P5 — documentation

32. **`docs/PRODUCTION_READINESS.md`** is a self-congratulatory phase report.
33. **`docs/FIDELITY.md` is hand-edited**.
34. **`docs/REPRODUCE.md`** describes a run that does not exist.

---

## Technical Debt Register

| ID | Severity | Area | One-line |
|---|---|---|---|
| 1 | P0 | train.completion | Real `Pipeline` crashes `Completion.step` due to Output-vs-tuple unpacking |
| 2 | P0 | morel/__init__ | README Quick Start import path broken |
| 3 | P0 | cli | 4/6 subcommands are stubs |
| 4 | P0 | app | Experiment.run / Benchmark.run / Reproduce.run are stubs |
| 5 | P0 | data.cli | extract/build are stubs |
| 6 | P0 | checkpoint + serve | `torch.load(weights_only=False)` everywhere |
| 7 | P0 | encode.layer | Post-LN code, Pre-LN docstring |
| 8 | P0 | benchmarks | end-to-end bench crashes on `np.eye` adjacency |
| 9 | P1 | trainer | device selection ignores config; AMP is a no-op |
| 10 | P1 | encode.pool | softmax-NaN when full row masked |
| 11 | P1 | docs/API.md | wrong import paths |
| 12 | P1 | data.acquire | legacy Amazon URL |
| 13 | P1 | recommend.light | cache key by `id()` is fragile |
| 14 | P1 | trainer | resume epoch arithmetic wrong |
| 15 | P2 | types | three `Mask` with the same name |
| 16 | P2 | encode vs data | `Encoder` name collision |
| 17 | P2 | codebook | no shared base class |
| 18 | P2 | pipeline | `register_buffers` shadow + not real buffers |
| 19 | P2 | pipeline | `pipeline/pipeline.py` module-name duplicates class |
| 20 | P3 | bpr.negatives | Python loop with setdiff |
| 21 | P3 | retrieve.relevance | per-candidate Python loop |
| 22 | P3 | retrieve.pipeline.batch | not actually batched |
| 23 | P3 | pipeline._encode_subgraph | per-item Python loop |
| 24 | P3 | eval.ranking.mrr | full sort instead of argpartition |
| 25 | P4 | lint | 182 ruff violations |
| 26 | P4 | multiple | `del arg` markers |
| 27 | P4 | types | mypy blocked by numpy/3.12 stub issue |
| 28 | P4 | tests | real `Pipeline` not exercised through `Completion` |
| 29 | P5 | docs/PRODUCTION_READINESS.md | contradicts current state |
| 30 | P5 | docs/FIDELITY.md | hand-edited, not auto-rendered |
| 31 | P5 | docs/REPRODUCE.md | describes nonexistent run |

---

## Refactor Plan (sequenced)

### Phase 1 — P0 correctness
Restore the contract claims that the docs make: fix the trainer↔model contract,
re-export the public API, implement the CLI handlers, harden checkpoint loading,
and fix the LayerNorm order. Replace `_Standin` tests with real-Pipeline tests.

### Phase 2 — P1 reliability
Honour `config.device`, wire AMP, fix resume arithmetic, make softmax
NaN-safe, harden LightGCN cache, rewrite `docs/API.md`, and add the real
Pipeline integration test.

### Phase 3 — Architecture / Polymorphism
Rename the overloaded types (`Mask` → `Masking`, `Encoder` → `FeatureEncoder` /
`GraphEncoder`), add `Codebook` and `Recommender` bases, rename
`Pipeline.register_buffers` → `Pipeline.attach_corpus`, file-rename
`pipeline/pipeline.py` → `composer.py`, remove `del arg` patterns, centralize
config hashing, and sweep ruff violations.

### Phase 4 — Performance
Vectorise `bpr.negatives`, `relevance.mean_relevance`,
`Pipeline._encode_subgraph` (padded batching), and `eval.ranking.mrr`.

### Phase 5 — Documentation / Observability / Packaging
Regenerate `PRODUCTION_READINESS.md`, `FIDELITY.md`, `REPRODUCE.md`. Wire
`configure_log` and `Monitor` into every CLI subcommand. Update Makefile targets.
Pin numpy for CI mypy.

### Phase 6 — Research validation
Property tests for probability invariants, numerical safety tests, paper-fidelity
coverage test, robustness sweep test, real-Pipeline research validation.

### Phase 7 — Distributed training (single-node multi-GPU)
`morel/core/distributed.py` runtime primitives, per-rank seeding,
`DistributedTrainer`, `torchrun` entry point, per-rank logging.

### Phase 8 — True streaming ingestion
Line-by-line streaming JSON parser, two-pass exact k-core via stream, online
degree-filter for `IterableDataset` trainers, mmapped feature arrays.

### Phase 9 — Serve fine-tuning (full pipeline + two-token auth)
Two-token auth (read / admin), `PipelineUpdater` (completion + recommender
online update with replay buffer and divergence guard), reader-writer lock,
`/v1/feedback`, `/v1/rollback`, `/v1/stats`, `--updater enabled|disabled`.

---

## Atomic Change List

> **Total: 62 atomic changes across 9 phases.**
> Each change is small enough to review and revert independently. Each has
> explicit acceptance criteria and a verification command. The phases are
> sequenced so that each phase's changes keep the verification gates green
> before the next phase begins.

### Phase 1 — P0 Correctness

- [ ] **1.1** Fix `train/completion.py` to consume `Output`
  - File: `morel/train/completion.py`
  - Replace `predictions, probs = self.model(...)` and `predictions, _ = self.model(...)` with `output = self.model(...)`; then `predictions = output.completed`, `probs = output.routing`. Validate `output.completed.keys() == features.keys()`.
  - Acceptance: New `tests/integration/test_pipeline_with_completion.py` drives the real `Pipeline` through `Completion.fit` for 3 epochs on synthetic data; asserts no `TypeError`; asserts loss is finite.

- [ ] **1.2** Rename `_Standin` usages to real `Pipeline`
  - Files: `tests/integration/test_end_to_end.py`, `tests/unit/train/test_trainers.py`
  - Delete `_Standin` classes. Replace usages with real `Pipeline(config, dims={"visual": ..., "text": ...})`.
  - Acceptance: Both test files run green; `_Standin` no longer exists in the codebase.

- [ ] **1.3** Fix `encode/layer.py` to true Pre-LN
  - File: `morel/encode/layer.py`
  - Rewrite `Layer.forward` so the order is `x = x + attn(norm1(x))` then `x = x + ffn(norm2(x))`. Keep `norm1`/`norm2` parameters.
  - Acceptance: New `tests/unit/encode/test_encode.py::test_layer_is_preln` constructs a `Layer` with `norm1.weight = nn.Parameter(torch.zeros(...))`; asserts the residual output equals the input within tolerance.

- [ ] **1.4** Add `safe_load` and replace `torch.load(weights_only=False)`
  - Files: `morel/train/checkpoint.py`, `morel/serve/loader.py`
  - Add `safe_load(path) -> dict` calling `torch.load(path, map_location="cpu", weights_only=True)`, validating payload keys against an explicit whitelist. Keep `unsafe_load` opt-in. Replace `torch.load(..., weights_only=False)` in `State.load` and `Loader.load_path`. `Reproduce` accepts `--allow-unsafe`.
  - Acceptance: New `tests/unit/train/test_checkpoint.py::test_safe_load_rejects_unsafe` constructs a `__reduce__` exploit payload and asserts `safe_load` raises. Existing checkpoint roundtrip tests still pass.

- [ ] **1.5** Re-export public API from `morel/__init__.py`
  - File: `morel/__init__.py`
  - Add re-exports: `from morel.core import Config, Modality, Mask, Manifest, seed_everything`; `from morel.pipeline import Pipeline, Output`; `from morel.core.types import Embedding, Graph`. Add `__getattr__` for lazy submodule access.
  - Acceptance: `python -c "from morel import Config, Pipeline, seed_everything, Output"` exits 0.

- [ ] **1.6** Implement `morel train`, `morel eval`, `morel bench`, `morel reproduce`, `morel render-fidelity`
  - Files: `morel/cli/__init__.py`
  - Replace the four stubs with handlers delegating to `morel.app.experiment.{Experiment,Benchmark,Reproduce}.run`. Add `render-fidelity` handler calling `morel.core.fidelity.render_markdown` / `render_json`.
  - Acceptance: `morel train completion configs/synthetic.yaml --epochs 1` exits 0 and writes `runs/<ts>/config.yaml`. `morel render-fidelity /tmp/FIDELITY.md /tmp/FIDELITY.json` exits 0 and produces non-empty files.

- [ ] **1.7** Implement `Experiment.run`, `Benchmark.run`, `Reproduce.run`
  - File: `morel/app/experiment.py`
  - `Experiment.run` synthesises a small dataset, builds a `Pipeline`, runs `Completion.fit`, writes `config.yaml`, `manifest.json`, `metrics.jsonl`, `FIDELITY.md`, `report.md`, `checkpoints/best.pt` to `run_dir`. Returns `{"duration", "metrics", "config_hash"}`. `Benchmark.run` instantiates `Pipeline`s at sizes and measures forward+backward latency. `Reproduce.run` reads YAML, sets seed, validates manifest's `config_hash`, delegates to `Experiment.run`.
  - Acceptance: `tests/unit/app/test_app.py` extends `test_experiment_run` to assert `metrics.jsonl` exists, contains at least one line, and `best.pt` exists. `test_reproduce_run` roundtrips a YAML.

- [ ] **1.8** Add `morel.app.train` and `morel.app.eval`
  - Files: new `morel/app/train.py`, new `morel/app/eval.py`
  - `train.run(config, run_dir)` runs `Completion.fit` and `Recommendation.fit`. `eval.run(config, run_dir)` loads the trained pipeline and test split, computes Recall@K and NDCG@K at each ratio, writes markdown report.
  - Acceptance: `morel eval rank runs/<ts>` exits 0 and writes a JSONL of metrics per K.

- [ ] **1.9** Implement `data/__main__.py` extract and build
  - File: `morel/data/__main__.py`
  - `extract` reads raw Amazon JSON, runs `text`/`visual` encoders (or random fallback when `--synthetic`), writes feature arrays + manifest. `build` reads raw Amazon JSON, runs iterative k-core, writes `bipartite.npz`, `item_graph.npz`, manifests.
  - Acceptance: `morel data extract --synthetic --out-dir /tmp/x` writes `features.npz` and a manifest. `morel data build --synthetic --out-dir /tmp/x` writes `bipartite.npz` and `item_graph.npz`.

- [ ] **1.10** Switch `data/acquire.download` URL by default
  - File: `morel/data/acquire.py`
  - Default `download` uses the Amazon-Reviews-2023 mirror. Add `download_legacy(category, dest)` using the McAuley UCSD URL. Add an actionable `DataError` message when neither URL responds.
  - Acceptance: `download_legacy("Beauty", "/tmp/raw")` behaves identically to the current implementation. `download("Beauty", "/tmp/raw")` with no network raises `DataError` with both URLs in the message.

- [ ] **1.11** Rename `Pipeline.register_buffers` to `Pipeline.attach_corpus`
  - File: `morel/pipeline/pipeline.py`, callers in `tests/unit/pipeline/test_pipeline.py` and `tests/integration/test_end_to_end.py`
  - Rename the method. Update the two test files and any in-repo callers. No backward-compatibility shim.
  - Acceptance: `grep -rn register_buffers morel/ tests/` returns no hits.

### Phase 2 — P1 Reliability

- [ ] **2.1** Honour `config.device` and wire AMP in `Trainer`
  - File: `morel/train/trainer.py`
  - Accept `device: str | torch.device | None = None` and resolve via `morel.core.device.device()`. When `amp=True`, wrap `step` in `torch.amp.autocast(device_type=...)` and use `torch.amp.GradScaler` for the backward pass.
  - Acceptance: New `tests/unit/train/test_trainers.py::test_trainer_honours_config_device` builds a trainer with `device="cpu"` and asserts `trainer.device == torch.device("cpu")`. `test_trainer_amp_cpu_runs` asserts AMP-on-CPU does not raise.

- [ ] **2.2** Fix `Trainer.fit` epoch arithmetic
  - File: `morel/train/trainer.py`
  - Replace `for epoch in range(start_epoch, epochs)` with `for epoch in range(start_epoch, start_epoch + epochs)`.
  - Acceptance: New test saves a checkpoint at epoch 3, resumes with `epochs=5`, asserts the resumed run trains epochs 3..7.

- [ ] **2.3** NaN-safe `pool.Attention`
  - File: `morel/encode/pool.py`
  - Replace `masked_fill(~mask, float("-inf"))` with `masked_fill(~mask, -1e9)`. Track row-sums of valid tokens; if a row has zero valid tokens, fall back to a uniform-weight average over the row.
  - Acceptance: New `tests/unit/encode/test_encode.py::test_attention_pool_no_nan_on_all_masked` constructs a pool input with a fully-masked row, asserts output has no NaN.

- [ ] **2.4** Content-hash cache key in `Light`
  - File: `morel/recommend/light.py`
  - Replace `_adj_cache = (id(ui_graph), tensor)` with `_adj_cache = (sha256_of_csr, tensor)`. Hash data/indices/indptr the same way `Laplace` does.
  - Acceptance: New test asserts two `csr_matrix` instances built from the same triplets share the cache entry, and a different adjacency forces a rebuild.

- [ ] **2.5** Rewrite `docs/API.md`
  - File: `docs/API.md`
  - Replace contents with the actual exports of each module. Run `python -c "from morel.X import *; print(*X.__all__)"` for each subpackage, paste the output.
  - Acceptance: Every import example in `API.md` is executable. CI step `python -c "exec(open('docs/API.md').read().replace('\`\`\`python','').replace('\`\`\`',''))"` exits 0.

- [ ] **2.6** Self-loop tolerance in `Pipeline.forward`
  - File: `morel/pipeline/pipeline.py`
  - At the start of `forward`, if `adj.diagonal().any()` raise `GraphError` with an actionable message ("build the item graph once with `morel.data.build.item_cooccurrence`; it removes self-loops").
  - Acceptance: `tests/integration/test_end_to_end.py` builds an item-cooccurrence adjacency and the test still passes. A test that explicitly passes `np.eye(N)` raises `GraphError`.

- [ ] **2.7** Fix `benchmarks/end_to_end.py`
  - File: `benchmarks/end_to_end.py`
  - Replace `np.eye(items)` adjacency with a real item cooccurrence graph: `adj = item_cooccurrence(bipartite(...))` with synthetic users.
  - Acceptance: `pytest benchmarks/end_to_end.py --benchmark-only` runs without error.

- [ ] **2.8** Integration test driving real `Pipeline` through `Completion`
  - File: new `tests/integration/test_pipeline_with_completion.py`
  - Constructs `Pipeline(config, dims={"visual": 4, "text": 2})`, attaches the corpus, drives `Completion.fit` for 3 epochs on a 50-item synthetic dataset, asserts final loss is finite and ≤ initial loss.
  - Acceptance: New integration test passes on CPU.

### Phase 3 — Architecture / Polymorphism

- [ ] **3.1** Rename `core.config.Mask` → `Masking`
  - Files: `morel/core/config.py`, all consumers (`morel/cli/__init__.py`, `morel/pipeline/pipeline.py`, `morel/app/experiment.py`, `docs/METHOD.md`, `docs/REPRODUCE.md`, all YAML examples).
  - Rename the dataclass and all references.
  - Acceptance: `grep -rn "config.mask\." morel/ tests/` returns no hits. `grep -rn "config.masking\." morel/ tests/` returns hits only.

- [ ] **3.2** Rename `data.extract.Encoder` → `FeatureEncoder`
  - Files: `morel/data/extract.py`, all consumers.
  - Rename the Protocol. Update imports.
  - Acceptance: `from morel.data.extract import Encoder` fails; `from morel.data.extract import FeatureEncoder` succeeds.

- [ ] **3.3** Rename `encode.baseline.Encoder` → `GraphEncoder`
  - Files: `morel/encode/baseline.py`, all consumers.
  - Rename the Protocol. Rename the multiplexer `Baseline` to `GraphEncoderBaseline`. Update `encode/__init__.py`.
  - Acceptance: Same pattern as 3.2.

- [ ] **3.4** Add `Codebook` ABC
  - File: `morel/codebook/codebook.py`
  - Add `class Codebook(nn.Module)` with `forward(hidden, *, training: bool) -> tuple[Tensor, Tensor]`. Make `VQ` and `GumbelVQ` subclass it. Add `IdentityCodebook` (returns `(hidden, ones(K)/K)`) for ablations.
  - Acceptance: `isinstance(GumbelVQ(dim=8, size=10, router=Dense(...)), Codebook)` is True. `tests/unit/codebook/test_codebook.py` extends with an `IdentityCodebook` test.

- [ ] **3.5** Add `Recommender` Protocol
  - File: new `morel/recommend/protocol.py`
  - Define `class Recommender(Protocol)` with `forward(users: Tensor, items: Tensor, ui_graph: csr_matrix | None = None) -> Tensor`. Annotate `Light`, `MF`, `Pop` forward signatures.
  - Acceptance: mypy reports no Protocol-mismatch on `Recommendation.trainer` where `model: Recommender`.

- [ ] **3.6** Rename `pipeline/pipeline.py` → `pipeline/composer.py`
  - Files: file rename + `morel/pipeline/__init__.py` re-export.
  - Move file. Update re-export.
  - Acceptance: `from morel.pipeline import Pipeline` still works. `ls morel/pipeline/` shows `composer.py`, not `pipeline.py`.

- [ ] **3.7** Remove `del arg` statements
  - Files: `morel/route/router.py`, `morel/recommend/baseline.py`.
  - Replace `del dim`, `del hidden`, `del ui_graph` with `_` parameter names or by removing the parameter from the signature.
  - Acceptance: `grep -rn "^[[:space:]]*del [a-z]" morel/` returns no hits.

- [ ] **3.8** Centralize config hashing
  - Files: `morel/train/completion.py`, `morel/train/recommendation.py`.
  - Delete the duplicate `hash` methods on `CompletionConfig` and `RecommendationConfig`. Rely on `train.checkpoint.hash_config(config)` only.
  - Acceptance: `grep -rn "def hash(self)" morel/train/` returns no hits.

- [ ] **3.9** Apply ruff auto-fixes and review each diff
  - Files: all of `morel/`.
  - Run `ruff check --fix --unsafe-fixes morel/` then review and commit each category (`D406/D407`, `F401`, `I001`, `E501`, `F841`) as separate commits.
  - Acceptance: `ruff check morel/` exits 0.

- [ ] **3.10** Fix import-linter contracts after renames
  - File: `pyproject.toml`
  - Ensure no new forbidden imports appear after renames. Re-run `lint-imports --config pyproject.toml`.
  - Acceptance: `lint-imports` exits 0.

### Phase 4 — Performance

- [ ] **4.1** Vectorise `bpr.negatives`
  - File: `morel/recommend/bpr.py`
  - Build a `(users, items)` boolean positive matrix once; for each user, sample from the complement. Use rejection sampling with a fallback to enumeration for users with very few negatives.
  - Acceptance: `tests/unit/recommend/test_recommend.py::test_negatives_vectorised` runs the new sampler and the old sampler on the same seed; outputs are bit-identical. `benchmarks/data.py` shows ≥ 3× speedup on 10k users.

- [ ] **4.2** Vectorise `relevance.mean_relevance`
  - File: `morel/retrieve/relevance.py`
  - Pre-normalise features per modality once; for each query, compute `sims = candidates @ query / (norms[c] * norms[q])` in one matmul per modality; average over jointly observed modalities.
  - Acceptance: New test asserts the vectorised output equals the per-candidate Python output on a fixed seed (tolerance 1e-6). `benchmarks/retrieve.py::bench_mage_1k` shows ≥ 5× speedup.

- [ ] **4.3** Vectorise `Pipeline._encode_subgraph` via padded batching
  - File: `morel/pipeline/pipeline.py`
  - Pad all per-query subgraphs to a common length (use `retrieve.batch`'s padded output), build one `(B, S_max, M, D)` tensor, one `(B, S_max, M)` mask, one `(B, S_max, pe_dim)` PE tensor, run the Transformer once with an attention mask.
  - Acceptance: New test asserts that padded-batched output equals the per-query-loop output elementwise (within fp tolerance) on a fixed seed. `benchmarks/end_to_end.py` shows ≥ 2× throughput.

- [ ] **4.4** Argpartition in `eval.ranking.mrr`
  - File: `morel/eval/ranking.py`
  - Use `np.argpartition` for the first relevant item; only `argsort` once that row is identified.
  - Acceptance: Existing tests pass. `benchmarks/ranking.py` (new) shows ≥ 1.5× speedup at 10k items.

### Phase 5 — Documentation / Observability / Packaging

- [ ] **5.1** Regenerate `docs/PRODUCTION_READINESS.md`
  - File: `docs/PRODUCTION_READINESS.md`
  - Replace contents with a real audit table derived from `pytest --cov`, `ruff`, `mypy`, and the manual checks from this plan. Each row cites a passing test or measurement.
  - Acceptance: Every row in the table has a "Verified By" reference that actually exists in the repo.

- [ ] **5.2** Auto-render `docs/FIDELITY.md`
  - Files: `docs/FIDELITY.md`, `morel/cli/__init__.py`
  - Add `render-fidelity <md_path> <json_path>` subcommand. Update `docs/FIDELITY.md` to call it from `make fidelity`. The registry's markdown is regenerated each release.
  - Acceptance: `morel render-fidelity /tmp/FIDELITY.md /tmp/FIDELITY.json` produces files matching the registry entries.

- [ ] **5.3** Update `docs/REPRODUCE.md`
  - File: `docs/REPRODUCE.md`
  - Replace the fictional run with the actual `morel reproduce configs/synthetic.yaml` workflow. Show the real artifact bundle.
  - Acceptance: Following the doc verbatim reproduces the artifact bundle.

- [ ] **5.4** Wire `configure_log` and `Monitor` into every CLI subcommand
  - File: `morel/cli/__init__.py`
  - Each handler calls `configure_log(level=...)` once at startup. Each handler creates a `Monitor` under `runs/<ts>/metrics.jsonl` and logs subcommand entry/exit.
  - Acceptance: `morel reproduce configs/synthetic.yaml` produces a `metrics.jsonl` with at least 5 lines.

- [ ] **5.5** Update Makefile targets
  - File: `Makefile`
  - `reproduce:` invokes `morel reproduce configs/synthetic.yaml`. `fidelity:` invokes `morel render-fidelity docs/FIDELITY.md docs/FIDELITY.json`.
  - Acceptance: `make reproduce` and `make fidelity` exit 0.

- [ ] **5.6** Pin numpy version constraint for CI mypy
  - File: `pyproject.toml`
  - Add `numpy>=2.0` to the dev extras; CI installs numpy 2.x. mypy strict passes on numpy 2.x.
  - Acceptance: `mypy morel/` exits 0 in CI.

- [ ] **5.7** Add `notebooks/01_end_to_end.ipynb`
  - File: new `notebooks/01_end_to_end.ipynb`
  - Programmatic notebook that runs the synthetic demo, calls `render-fidelity`, plots a simple metric.
  - Acceptance: `jupyter nbconvert --execute notebooks/01_end_to_end.ipynb` exits 0.

### Phase 6 — Research Validation

- [ ] **6.1** Property tests for probability invariants
  - File: `tests/property/test_invariants.py`
  - Add tests: router probs sum to 1 for `Dense`, `Top`, `Gumbel`; Gumbel temperature monotonicity; LightGCN zero-layers = dot product; bernoulli mask rowsum ≥ 1.
  - Acceptance: All new property tests pass.

- [ ] **6.2** Numerical safety tests
  - Files: `tests/unit/encode/test_encode.py`, `tests/unit/route/test_router.py`, new `tests/integration/test_pipeline_robustness.py`
  - Tests asserting no NaN/Inf for: attention pool with all-masked row; router with near-zero logits; pipeline with one modality missing; pipeline with all modalities missing; pipeline with adversarial mask patterns.
  - Acceptance: All new tests pass.

- [ ] **6.3** Paper-fidelity coverage test
  - File: new `tests/research/test_fidelity_report.py`
  - Iterate the fidelity registry, assert each entry's `test` field references an existing test function that passes. Also assert `render-fidelity` produces non-empty output.
  - Acceptance: New test passes and fails immediately when a registry entry points to a missing test.

- [ ] **6.4** Robustness sweep test
  - File: new `tests/research/test_robustness_sweep.py`
  - Runs `eval.robustness_sweep` across the configured ratios on synthetic data; asserts no NaN in the metric curve and the curve is monotonically non-increasing for the simple case.
  - Acceptance: New test passes.

- [ ] **6.5** Real-`Pipeline` research validation
  - File: new `tests/research/test_real_pipeline.py`
  - Builds a real `Pipeline`, runs `Completion.fit` for 30 epochs on synthetic data, asserts reconstruction loss decreases, asserts Gumbel-Softmax produces a non-degenerate routing distribution after training.
  - Acceptance: New test passes.

### Phase 7 — Distributed Training (single-node multi-GPU)

- [ ] **7.1** `morel/core/distributed.py`
  - File: new `morel/core/distributed.py`
  - `init`, `is_initialized`, `rank`, `world_size`, `local_rank`, `is_rank_zero`, `barrier`, `reduce_mean`, `cleanup`. Reads `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`. Backend selection: NCCL on CUDA, Gloo otherwise.
  - Acceptance: `tests/unit/runtime/test_distributed.py` (new) calls `init()` under `WORLD_SIZE=1` and asserts idempotency, rank zero, etc.

- [ ] **7.2** Extend `morel/core/seed.py` for per-rank seeding
  - File: `morel/core/seed.py`
  - `seed(value, *, rank: int = 0)` seeds `manual_seed(value + rank)` and the same for CUDA.
  - Acceptance: New test asserts that with rank=0 vs rank=1, the first random sample differs.

- [ ] **7.3** `DistributedTrainer`
  - File: new `morel/train/distributed.py`
  - Subclass of `Trainer`. Wraps `self.model` in `DistributedDataParallel`. Attaches `DistributedSampler` to loaders. Aggregates metrics via `reduce_mean`. Checkpoint and Monitor only on rank zero. AMP per-rank.
  - Acceptance: `tests/integration/test_distributed_train.py` (new) uses `torch.multiprocessing.spawn(..., nprocs=2)`, asserts both processes converge to the same loss within fp tolerance at epoch 1.

- [ ] **7.4** Extend `morel/cli/__init__.py` for `torchrun`
  - File: `morel/cli/__init__.py`
  - Detect `RANK` env var; if set, call `init()` and use `DistributedTrainer`. Update the `train` subcommand parser with `--backend {nccl,gloo}`.
  - Acceptance: `torchrun --nproc_per_node=2 morel train completion configs/synthetic.yaml` (manually smoke-tested; CI skip-on-no-GPU).

- [ ] **7.5** Per-rank logging in `morel/core/log.py`
  - Files: `morel/core/log.py`, `morel/train/monitor.py`
  - `Monitor` accepts `rank` and writes to `metrics.rank{rank}.jsonl` for non-zero ranks. `configure_log` is a no-op on non-zero ranks (rank zero owns the file).
  - Acceptance: New integration test asserts the non-zero-rank file exists after a 2-process run.

### Phase 8 — True Streaming Ingestion

- [ ] **8.1** `morel/data/stream.py`
  - File: new `morel/data/stream.py`
  - `ReviewStream(path, *, chunk_size)`, `interactions_streaming(review_path, metadata_path, *, min_edges, chunk_size)`, `StreamingInteractionsIndex`. Two-pass exact k-core via the stream; online degree-filter with rolling windows for the online case.
  - Acceptance: `tests/unit/data/test_stream.py::test_exact_two_pass_kcore` asserts the streaming path produces a graph bit-identical to the in-memory path on a fixed seed.

- [ ] **8.2** `morel/data/build.py` extensions
  - File: `morel/data/build.py`
  - `interactions_streaming(...)` and `item_cooccurrence_streaming(...)`. The existing `interactions` is preserved for offline use; FIDELITY.md gains an entry for the streaming variant.
  - Acceptance: `tests/unit/data/test_build.py` adds `test_item_cooccurrence_streaming_matches_in_memory`.

- [ ] **8.3** Chunked extractors in `morel/data/extract.py`
  - File: `morel/data/extract.py`
  - `text_streaming(paths_iter, encoder, *, chunk_size)`, `visual_streaming(paths_iter, encoder, *, chunk_size)`. Encode in mini-batches; yield `np.ndarray` chunks.
  - Acceptance: New test asserts the chunked encoder agrees with the full encoder on the same inputs at the same RNG state.

- [ ] **8.4** `IterableDataset` trainers
  - Files: `morel/train/completion.py`, `morel/train/recommendation.py`
  - The trainer accepts either `Dataset` or `IterableDataset`. For `IterableDataset`, the trainer uses a step-count loop. `DistributedTrainer` wraps `IterableDataset` with a `StreamingDataset` (per-rank shard via rank/world_size on a deterministic counter).
  - Acceptance: `tests/integration/test_streaming_train.py` drives `Completion` from an `IterableDataset` for N steps; asserts loss is finite.

- [ ] **8.5** Memory-budget assertion
  - File: new `tests/perf/test_memory_budget.py`
  - Runs the synthetic streaming pipeline under `tracemalloc`; asserts peak resident set ≤ 256 MB.
  - Acceptance: Test passes on Linux CI; skipped elsewhere.

- [ ] **8.6** FIDELITY entry for online k-core approximation
  - Files: `docs/FIDELITY.md`, `morel/core/fidelity.py`
  - Add `Online k-core approximation — APPROXIMATE — morel.data.stream.StreamingKCore — deviation: rolling-window online filter; offline k-core is exact when two passes are available`.
  - Acceptance: `morel render-fidelity` includes the entry.

### Phase 9 — Serve Fine-tuning (full pipeline + two-token auth)

- [ ] **9.1** Two-token auth in `morel/serve/auth.py`
  - File: `morel/serve/auth.py`
  - Support `MOREL_AUTH_TOKEN_READ` and `MOREL_AUTH_TOKEN_ADMIN`. Legacy `MOREL_AUTH_TOKEN` is treated as admin. Add `scope()` helper returning `Literal["read", "admin"]`. Update `require()` to accept an optional `scope` argument.
  - Acceptance: New `tests/unit/serve/test_auth.py::test_two_tokens` asserts each scope permits its endpoints and rejects the other.

- [ ] **9.2** `PipelineUpdater`
  - File: new `morel/serve/update.py`
  - `class PipelineUpdater` holding the live `Pipeline`, feedback ring, replay buffer, rollback ring, loss window. `tick()` performs one update step (completion + recommendation). Threaded task runner. Divergence guard with cooldown.
  - Acceptance: `tests/unit/serve/test_update.py` asserts loss decreases after replay+feedback mix, divergence triggers rollback, cooldown suppresses further updates, rollback ring evicts oldest.

- [ ] **9.3** Reader-writer lock around the live pipeline
  - File: new `morel/serve/lock.py`
  - `class RWLock` with `acquire_read`, `acquire_write`. `PipelineUpdater` acquires write for the duration of a `tick`. Request handlers acquire read for the duration of their forward call.
  - Acceptance: New stress test with 16 threads and `threading.Barrier` asserts no torn reads.

- [ ] **9.4** New endpoints
  - Files: `morel/serve/app.py`, `morel/serve/schema.py`
  - `POST /v1/feedback` (admin scope), `POST /v1/rollback` (admin scope), `GET /v1/stats` (admin scope). Update `/v1/recommend` and `/v1/complete` to use the read-lock handle.
  - Acceptance: `tests/integration/test_serve_finetune.py` posts 100 feedback events, polls `/v1/stats`, asserts `updates_applied >= 1` within 10 s.

- [ ] **9.5** Updater enable/disable flag
  - Files: `morel/cli/__init__.py`, `morel/serve/app.py`
  - `--updater {enabled,disabled}` flag on `morel serve`. When disabled, feedback endpoints return 503 with `Updater disabled`. Loader is constructed lazily on first feedback event.
  - Acceptance: `morel serve --updater disabled` accepts no updates and `/v1/feedback` returns 503.

- [ ] **9.6** FIDELITY entry for online pipeline update
  - Files: `docs/FIDELITY.md`, `morel/core/fidelity.py`
  - Add `Online full-pipeline update — APPROXIMATE — morel.serve.update.PipelineUpdater — deviation: replay buffer + divergence guard; not a closed-form online-learning algorithm`.
  - Acceptance: `morel render-fidelity` includes the entry.

---

## Verification Gates

After every phase:

```bash
ruff check morel/
ruff format --check morel/
mypy morel/
pytest -q
morel render-fidelity docs/FIDELITY.md docs/FIDELITY.json
```

For Phases 7–9 these gates gain additional smoke steps:

- Phase 7: a 2-process `torch.multiprocessing.spawn` smoke test under `tests/integration/test_distributed_train.py` (also runs in a 2-GPU self-hosted runner if available).
- Phase 8: `tests/integration/test_streaming_train.py` and `tracemalloc`-based memory assertion in `tests/perf/test_memory_budget.py` (skip if tracemalloc is unavailable).
- Phase 9: live `curl` round-trip from a small smoke script committed at `scripts/smoke_serve.sh`, executed against a `TestClient` of the FastAPI app, asserted by `tests/integration/test_serve_finetune.py`.

Each gate must be green before the next phase begins.

---

## Final Deliverable Table

| Area | Before (today) | After (post-refactor) | Verified By |
|---|---|---|---|
| Correctness | Completion trainer crashes on real Pipeline; Layer is Post-LN but says Pre-LN; `weights_only=False` everywhere; benchmark crashes on `np.eye` | `Completion.step` consumes `Output`; `Layer` is real Pre-LN; `safe_load` with `weights_only=True` + whitelist; `benchmarks/end_to_end.py` passes | `pytest -q`; `tests/integration/test_pipeline_with_completion.py`; `tests/unit/encode/test_encode.py::test_layer_is_preln`; `tests/unit/train/test_checkpoint.py::test_safe_load_rejects_unsafe` |
| Architecture | Three `Mask`, two `Encoder`, no `Codebook` base, no `Recommender` Protocol; `register_buffers` shadows nn.Module; `pipeline/pipeline.py` duplicates name | Single `Mask` Protocol + value type + `Masking` config section; `FeatureEncoder` vs `GraphEncoder`; `Codebook` ABC; `Recommender` Protocol; `Pipeline.attach_corpus`; `composer.py` | `pyproject.toml [tool.importlinter]` contracts; ruff clean; mypy strict |
| Reliability | Trainer ignores config.device; AMP no-op; resume arithmetic wrong; softmax-NaN on all-masked | Trainer honours config.device; AMP via `torch.amp`; resume fixed; softmax NaN-safe | new device + AMP tests; `tests/unit/encode/test_pool.py::test_attention_pool_no_nan` |
| Resilience | 4/6 CLI subcommands stubbed; app.run no-op; reproduce is a lie; legacy Amazon URL | Full CLI wired; `Experiment.run` produces complete artifact bundle; new URL by default; legacy preserved | `morel reproduce configs/synthetic.yaml` writes `runs/<ts>/{config.yaml, manifest.json, metrics.jsonl, FIDELITY.md, report.md, checkpoints/best.pt}` |
| Performance | Python loops in BPR negatives, relevance, encode_subgraph | Vectorised negatives + batched relevance + padded-batched subgraph encoding + argpartition in MRR | `benchmarks/` suite + new `bench_padded_subgraph` |
| Testing | 228 tests; real Pipeline never driven by Completion | 228 + ~70 new tests across P0/P1/property/research | `pytest -q` exits 0 |
| Reproducibility | `morel render-fidelity` does not exist; `morel reproduce` stub; FIDELITY.md hand-edited | Both subcommands exist and produce real artifacts; FIDELITY.md auto-rendered | `morel render-fidelity` and `morel reproduce` roundtrip |
| API | `from morel import Config, Pipeline, seed` broken; API.md has wrong imports | Public surface matches README and API.md | `python -c "from morel import Config, Pipeline, seed"` |
| Packaging | `weights_only=False`; AMP not exercised | `safe_load` + AMP path | `pip-audit` clean; smoke test |
| Documentation | PRODUCTION_READINESS.md is a phase report; FIDELITY.md hand-edited; REPRODUCE.md lies | All three re-rendered from real state; `<!-- FIDELITY:BEGIN -->` markers functional | `make docs` + `morel render-fidelity` |
| Paper fidelity | Registry exists but no auto-render in CI | Rendered by `morel render-fidelity`; each entry has a passing test | `tests/research/test_paper.py` extended; `tests/research/test_fidelity_report.py` |
| Serve | `weights_only=False` in loader; stub /v1/recommend | Hardened loader; live /v1/recommend on real LightGCN | `tests/unit/serve/test_serve.py` extended |
| Distributed training | none | `DistributedTrainer`; `torchrun` entry; per-rank seed; rank-0-only logging/checkpoint; AMP per-rank | `tests/integration/test_distributed_train.py` with `WORLD_SIZE=2`; manual smoke |
| Streaming ingestion | full dataset in RAM | `interactions_streaming`, `text_streaming`, `visual_streaming`, mmapped features, `IterableDataset` trainer, online degree-filter | `tests/integration/test_streaming_train.py`; memory budget recorded in `docs/INSTALL.md` |
| Serve fine-tuning | static pipeline | `PipelineUpdater` (full pipeline); replay buffer; divergence guard; cooldown; reader/writer lock; rollback ring; `/v1/feedback`, `/v1/rollback`, `/v1/stats`; two-token auth | `tests/integration/test_serve_finetune.py`; `scripts/smoke_serve.sh` |

---

## Risks and Tradeoffs

- **Behavior preservation vs. fixes.** Several "fixes" (Pre-LN, attention NaN safety, weight-only checkpoint loading) change semantics. Where possible, preserve the old code path under a flag for reproducibility of existing published numbers. Otherwise, update FIDELITY.md to mark the change.
- **Backwards compatibility.** No BC shims. `morel/__init__.py` re-exports will be the only break. Tests using `_Standin` will be replaced with real `Pipeline`; the old `_Standin` removed.
- **Performance changes.** The proposed subgraph-encoding vectorisation changes numerics by reordering attention masking; this affects numerical reproducibility. We pin numpy/torch versions and document.
- **Amazon dataset URL.** Switching URLs is a behavior change; we preserve the old URL as `data.acquire.download_legacy` and use the new one by default, with a CHANGELOG entry.
- **DDP determinism.** `torch.sparse.mm` and `index_add_` ops can be non-deterministic on CUDA. We pin `torch.use_deterministic_algorithms(True)` for DDP runs and document.
- **Streaming chunked k-core.** Chunked k-core requires two passes over the data (degree counts then edge collection). We accept this cost and document.
- **Serve fine-tuning under load.** A feedback storm could starve inference. Mitigation: bounded ring buffer + fixed update interval; never inline-update on the request thread.
- **Tensor in-flight during rollback.** A request currently scoring with the old model will see a torn state. Mitigation: copy-on-write snapshot; readers see the previous version until their request completes.
- **True streaming + online k-core approximation** is a real paper-fidelity deviation. FIDELITY.md entry will be APPROXIMATE.
- **Full-pipeline online update** can drift. We bound this with validation-loss-gated commits, divergence guard with cooldown, and a rollback ring of bounded depth. The model version is observable via `/v1/stats`. We add a `--updater disabled` escape hatch.
- **Two-token auth** adds two environment variables; the legacy single-token behavior is preserved (single token = admin). Operators upgrading must explicitly set both tokens.

---

## Out of Scope (explicit limitations)

The brief originally listed these as out of scope, but the repository owner
asked to include them in this refactor. They are now in scope as Phase 7
(DDP), Phase 8 (true streaming), and Phase 9 (full-pipeline online update).
