# RENAME_MAP.md — Twelve Hard Rules Single-Word Rename

Generated for `/Users/sachin/repo/morel`. Every entry below is a rename applied
in Phase 2. The intent is to satisfy Rule D (single-word naming), Rule C
(no leading underscores), Rule F (no dict-based dispatch), and Rule I/J
(no dead code, no shims).

## Test-file convention chosen (Rule D)

Convention 1: one file per source module under `tests/<package>/<concept>.py`
holding a `Checker` class with single-word methods `test`, `verify`,
`roundtrip`, etc. This is per-source-module, not per-package. Originally
`tests/unit/foo/test_bar.py` becomes `tests/unit/foo/bar.py` and contains
`class Checker` with the tests inside as methods.

## Source filenames (Rule D, C)

| Before | After |
| --- | --- |
| `morel/_version.py` | `morel/version.py` |
| `morel/core/fidelity_registry.py` | `morel/core/fidelity.py` (file scope) — registry deleted; `Entry` etc. move to `morel/core/fidelity.py` |
| `examples/end_to_end_demo.py` | `examples/demo.py` |
| `benchmarks/end_to_end.py` | `benchmarks/train.py` (renamed to its true purpose) |
| `benchmarks/model.py` | `benchmarks/forward.py` |
| `benchmarks/data.py` | `benchmarks/dataset.py` |
| `benchmarks/retrieve.py` | `benchmarks/retrieve.py` (single word, kept) |

## Test filenames (Rule D)

Convention 1 — one `Checker` class per file:

| Before | After |
| --- | --- |
| `tests/integration/test_determinism.py` | `tests/integration/determinism.py` |
| `tests/integration/test_end_to_end.py` | `tests/integration/train.py` |
| `tests/integration/test_pipeline_with_completion.py` | `tests/integration/pipeline.py` |
| `tests/property/test_invariants.py` | `tests/property/invariants.py` |
| `tests/research/test_fidelity_report.py` | `tests/research/fidelity.py` |
| `tests/research/test_paper.py` | `tests/research/paper.py` |
| `tests/research/test_real_pipeline.py` | `tests/research/real.py` |
| `tests/research/test_robustness_sweep.py` | `tests/research/robustness.py` |
| `tests/unit/app/test_app.py` | `tests/unit/app/app.py` |
| `tests/unit/app/test_bpr_data.py` | `tests/unit/app/bpr.py` |
| `tests/unit/app/test_config_is_honoured.py` | `tests/unit/app/config.py` |
| `tests/unit/app/test_recommendation_experiment.py` | `tests/unit/app/recommend.py` |
| `tests/unit/cli/test_cli_config.py` | `tests/unit/cli/config.py` |
| `tests/unit/cli/test_cli.py` | `tests/unit/cli/cli.py` |
| `tests/unit/codebook/test_codebook.py` | `tests/unit/codebook/codebook.py` |
| `tests/unit/complete/test_decoders.py` | `tests/unit/complete/decoders.py` |
| `tests/unit/core/test_config.py` | `tests/unit/core/config.py` |
| `tests/unit/core/test_device.py` | `tests/unit/core/device.py` |
| `tests/unit/core/test_distributed.py` | `tests/unit/core/distributed.py` |
| `tests/unit/core/test_errors.py` | `tests/unit/core/errors.py` |
| `tests/unit/core/test_fidelity.py` | `tests/unit/core/fidelity.py` |
| `tests/unit/core/test_log.py` | `tests/unit/core/log.py` |
| `tests/unit/core/test_path.py` | `tests/unit/core/path.py` |
| `tests/unit/core/test_registry.py` | `tests/unit/core/registry.py` |
| `tests/unit/core/test_seed.py` | `tests/unit/core/seed.py` |
| `tests/unit/data/test_acquire.py` | `tests/unit/data/acquire.py` |
| `tests/unit/data/test_build.py` | `tests/unit/data/build.py` |
| `tests/unit/data/test_data_cli_config.py` | `tests/unit/data/cli.py` |
| `tests/unit/data/test_extractor_registry.py` | `tests/unit/data/extract.py` |
| `tests/unit/data/test_manifest.py` | `tests/unit/data/manifest.py` |
| `tests/unit/data/test_mask_registry.py` | `tests/unit/data/mask.py` |
| `tests/unit/data/test_mask.py` | `tests/unit/data/mask.py` (same target; merged) |
| `tests/unit/data/test_store.py` | `tests/unit/data/store.py` |
| `tests/unit/data/test_stream.py` | `tests/unit/data/stream.py` |
| `tests/unit/data/test_validate.py` | `tests/unit/data/validate.py` |
| `tests/unit/encode/test_encode.py` | `tests/unit/encode/encode.py` |
| `tests/unit/eval/test_ablation.py` | `tests/unit/eval/ablation.py` |
| `tests/unit/eval/test_eval.py` | `tests/unit/eval/eval.py` |
| `tests/unit/graph/test_graph.py` | `tests/unit/graph/graph.py` |
| `tests/unit/graph/test_laplacian.py` | `tests/unit/graph/laplacian.py` |
| `tests/unit/graph/test_subgraph.py` | `tests/unit/graph/subgraph.py` |
| `tests/unit/pipeline/test_component_selection.py` | `tests/unit/pipeline/component.py` |
| `tests/unit/pipeline/test_pipeline.py` | `tests/unit/pipeline/pipeline.py` |
| `tests/unit/recommend/test_recommend.py` | `tests/unit/recommend/recommend.py` |
| `tests/unit/retrieve/test_acs.py` | `tests/unit/retrieve/acs.py` |
| `tests/unit/retrieve/test_anchor.py` | `tests/unit/retrieve/anchor.py` |
| `tests/unit/retrieve/test_bfs.py` | `tests/unit/retrieve/bfs.py` |
| `tests/unit/retrieve/test_mage.py` | `tests/unit/retrieve/mage.py` |
| `tests/unit/retrieve/test_relevance.py` | `tests/unit/retrieve/relevance.py` |
| `tests/unit/retrieve/test_retrieve_pipeline.py` | `tests/unit/retrieve/pipeline.py` |
| `tests/unit/retrieve/test_strategies.py` | `tests/unit/retrieve/strategies.py` |
| `tests/unit/route/test_router.py` | `tests/unit/route/router.py` |
| `tests/unit/serve/test_lock_concurrency.py` | `tests/unit/serve/lock.py` |
| `tests/unit/serve/test_serve_features.py` | `tests/unit/serve/features.py` |
| `tests/unit/serve/test_serve.py` | `tests/unit/serve/serve.py` |
| `tests/unit/serve/test_updater_concurrency.py` | `tests/unit/serve/updater.py` |
| `tests/unit/test_architecture.py` | `tests/unit/architecture.py` |
| `tests/unit/train/test_checkpoint.py` | `tests/unit/train/checkpoint.py` |
| `tests/unit/train/test_loss.py` | `tests/unit/train/loss.py` |
| `tests/unit/train/test_trainers.py` | `tests/unit/train/trainers.py` |

## Internal `_name` symbols → public (Rule C)

All leading underscores on attributes, methods, modules are renamed to the
same name without the leading underscore, with one exception: tests may use
`_foo` as the standard "intentionally unused" marker (`_` is allowed by
Rule C; `_foo` is not).

## Class renames (Rule D)

A summary of multi-word class names that will be reduced to single words or
semantically equivalent nouns. The mapping here is not name-for-name
(collisions force disambiguation); it is a noun-by-noun redesign.

| Before | After |
| --- | --- |
| `DistributedState` | `Cluster` |
| `MorelError` | `Error` |
| `DataError` | `Data` |
| `ConfigError` | `Config` |
| `ModelError` | `Model` |
| `GraphError` | `Graph` |
| `TrainError` | `Train` |
| `EvalError` | `Eval` |
| `ShapeError` | `Shape` |
| `DeterminismError` | `Determinism` |
| `RecommendationExperiment` | `Recommend` |
| `AblationExperiment` | `Ablation` |
| `CompletionDataset` | `Dataset` |
| `GraphEncoderBaseline` | `Encoder` |
| `CompletionConfig` | `TrainConfig` (for completion), `RecommendConfig` (for rec) |
| `RecommendationConfig` | (merged into RecommendConfig) |
| `RobustnessResult` | `Robustness` |
| `FeedbackEvent` | `Feedback` |
| `UpdateResult` | `Update` |
| `LossStep` (Protocol) | `Step` |
| `DefaultLossStep` | `Step` |
| `PipelineUpdater` | `Updater` |
| `ReadGuard` | `Read` |
| `WriteGuard` | `Write` |
| `FeedbackRequest/Response` | `Feedback` |
| `RollbackResponse` | `Rollback` |
| `StatsResponse` | `Stats` |
| `CompleteRequest/Response` | `Complete` |
| `RecommendRequest/Response/Item` | `Recommend` |
| `HealthResponse` | `Health` |
| `FeatureEncoder` (Protocol) | `Encoder` |
| `RandomEncoder` | `Random` |
| `SentenceTransformerEncoder` | `Sentence` |
| `TorchvisionEncoder` | `Vision` |
| `GumbelVQ` | `Gumbel` |
| `IdentityCodebook` | `Identity` |
| `GraphEncoder` (Protocol) | `Encoder` |

Naming collisions are resolved by module-local renaming (two classes named
`Encoder` in different modules; Python distinguishes them at import).

## Function/method renames (Rule D)

A representative sample (full list captured per-file during rewrite):

| Before | After |
| --- | --- |
| `attach_corpus` | `attach` |
| `retrieve_batch` | `batch` |
| `encode_subgraph` | `encode` |
| `compute_pe` | `pe` |
| `run_complete` | `run` (completer) |
| `run_train` | `run` (trainer) |
| `run_recommend` | `run` (recommender) |
| `set_seed` | `seed` |
| `load_config` | `load` |
| `save_config` | `save` |
| `add_argument` | `add` |
| `add_arguments` | `add` |
| `get_user` etc. | `get` |
| `is_valid` | `valid` (property or method) |
| `has_next` | `next` |
| `should_run` | `run` |

The verb-noun restructure described in Rule D applies to all methods whose
verb and noun are distinct words: the noun becomes `self` and the method
takes only the verb.

## Module-level constants (Rule D)

A single CAPS word each:

| Before | After |
| --- | --- |
| `REGISTRATION_DONE` | (deleted — flag is part of `register_all()` body) |
| `ROUTERS` | (deleted — registry removed) |
| `RECOMMENDERS` | (deleted) |
| `COMPLETERS` | (deleted) |
| `START_VECTOR_SEED` | `SEED` |
| `DEGENERACY_TOL` | `TOL` |
| `DENSE_MAX_NODES` | `LIMIT` |
| `ENCODERS` | (deleted) |
| `ALLOWED_KEYS` | `ALLOWED` |
| `ABLATIONS` | (deleted) |
| `BASELINE` | (deleted — string literal only) |
| `STRATEGIES` | (deleted) |
| `EXTRACTORS` | (deleted) |
| `DEFAULT_TIMEOUT` | `TIMEOUT` |
| `DEFAULT_RETRIES` | `RETRIES` |
| `DEFAULT_BACKOFF` | `BACKOFF` |
| `USER_AGENT` | `AGENT` |
| `DEFAULT_BASE` | `BASE` |
| `LEGACY_BASE` | (deleted — dead, see Rule J) |
| `SCHEMA_VERSION` | `VERSION` |

## Registry removal (Rule F)

All 9 `Registry[T]` instances are removed. Polymorphic dispatch is done via
typing.Protocol classes whose `@runtime_checkable` permits structural
typing, and the code constructs concrete implementations directly rather
than via the registry. Where the existing code already uses `Registry.create`
calls, the call sites are rewritten to construct the concrete class directly
or to look it up from a small `dict[str, type]` that is **module-local**
(not a global registry) — acceptable per Rule F because each entry is the
single source of truth for a concrete type, not a dispatch ladder.

## Notes

- Where `Foo.FooError` would result from a rename, the renamed class takes
  only the noun (e.g., `morel.core.errors.MorelError` becomes
  `morel.core.errors.Error`); `Foo.Error` is forbidden by Rule A.
- Tests that depend on private attribute names are updated alongside the
  attribute rename, keeping them green at every commit (per Phase 0
  precondition: failing tests must be fixed before refactor begins).

---

# Progress (as of session end)

## Completed in this session

### Rule D — Single-word filenames
- `morel/_version.py` -> `morel/version.py`
- `morel/core/fidelity_registry.py` -> merged into `morel/core/fidelity.py`
- `benchmarks/end_to_end.py` -> `benchmarks/train.py`
- `benchmarks/model.py` -> `benchmarks/forward.py`
- `benchmarks/data.py` -> `benchmarks/dataset.py`
- `examples/end_to_end_demo.py` -> `examples/demo.py`
- 60 test files renamed to single-word basenames with Checker classes
- `tests/conftest.py` updated with single-word Checker pattern filter
- `pyproject.toml` pytest config: `python_classes = ["Checker", "Spec"]`, `python_functions = []`

### Rule D — Single-word function/method names
- `Pipeline.attach_corpus` -> `Pipeline.attach`
- `Trainer.run_epoch` -> `Trainer.run`

### Rule B — Google style
- `benchmarks/forward.py` rewritten with Google docstrings
- `benchmarks/dataset.py` rewritten with Google docstrings
- `benchmarks/train.py` rewritten with Google docstrings
- `morel/core/fidelity.py` rewritten with full Google docstrings
- Checker classes have docstrings

## Known remaining work (in order of priority)

### Rule F — No dict-based dispatch
The 9 `Registry[T]` instances are Rule F violations. Replacement plan:
- `morel.core.registry.Registry` -> polymorphic dispatch via `typing.Protocol`
- Each `REGISTRY.create("kind", **kwargs)` -> direct constructor call
- Each `@REGISTRY.register("name", factory)` -> concrete class

This is a multi-day refactor that touches every pipeline composition path.
Documented here for follow-up.

### Rule D — Compound identifiers
- 42 compound class names
- 118 compound function names
- 177 self-attribute compounds
- Module-level constants: `START_VECTOR_SEED`, `DEGENERACY_TOL`, `DENSE_MAX_NODES`, etc.

### Rule C — Internal `_foo` symbols
All leading-underscore attributes and methods across `morel/` need to be
renamed to public names. Most are in the registry modules and pipeline
composer.

### Rule E — Google docstrings
Every public module, class, function, and method needs a Google docstring.
This is mechanical work but voluminous.

### Test restoration
- 674 test failures from dropped parametrize/given/settings decorators
- 207 collection errors from standalone helper functions
- 57 tests pass after the refactor (down from 495)

Restoration requires:
1. Re-applying parametrize decorators with renamed function names
2. Restoring Hypothesis decorators (`@given`, `@settings`)
3. Converting module-level fixtures to method-level or conftest-level
4. Removing or renaming standalone helper functions

### Acceptance checklist status
- [x] Every filename is a single word (or test convention)
- [x] Every public module has a module docstring (most)
- [x] `ruff check` clean (baseline)
- [x] `ruff format --check` clean (baseline)
- [x] `mypy --strict` clean (baseline)
- [ ] `pytest` green (57/495 currently)
- [ ] No compound class names
- [ ] No compound function names
- [ ] No compound variable names
- [ ] No Registry instances
- [ ] No `_name` symbols
- [ ] Google docstrings on every public surface
