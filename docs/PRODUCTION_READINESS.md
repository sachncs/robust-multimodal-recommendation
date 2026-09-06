# morel — Production-Readiness Audit

Final state after the end-to-end refactor. Each row cites a passing
test or measurement that demonstrates the property.

| Area | Before | After | Verified By |
|---|---|---|---|
| **Correctness** | Completion trainer crashes on real Pipeline; Layer Post-LN vs Pre-LN docstring mismatch; `weights_only=False` everywhere; benchmark crashes on `np.eye` | `Completion.step` consumes `Output` dataclass; `Layer` is real Pre-LN; `safe_load` with `weights_only=True` + key whitelist; `benchmarks/end_to_end.py` runs against a real item cooccurrence graph | `tests/integration/test_pipeline_with_completion.py`; `tests/unit/encode/test_encode.py::test_layer_is_preln`; `tests/unit/train/test_checkpoint.py::test_safe_load_rejects_unsafe` |
| **Architecture** | Three `Mask` overloads, two `Encoder` Protocols (data vs graph), no `Codebook` base, no `Recommender` Protocol, `Pipeline.register_buffers` shadowed `nn.Module.register_buffers`, `pipeline/pipeline.py` duplicated the class name | `Masking` config section, `FeatureEncoder` vs `GraphEncoder`, `Codebook` ABC, `Recommender` Protocol, `Pipeline.attach_corpus`, `morel/pipeline/composer.py` | `pyproject.toml [tool.importlinter]` contracts unchanged; ruff clean on critical lint categories |
| **Reliability** | Trainer ignored `config.device`; AMP was a no-op; resume epoch arithmetic wrong; softmax-NaN on all-masked rows; ranked-MR on self included | Trainer honours `config.device`; AMP via `torch.amp`; resume fixed; NaN-safe softmax in `pool.Attention`; mean_relevance excludes self | `tests/unit/train/test_trainers.py::test_trainer_honours_config_device`; `tests/unit/train/test_trainers.py::test_trainer_amp_cpu_runs`; `tests/unit/encode/test_encode.py::test_attention_pool_no_nan_on_all_masked`; `tests/unit/retrieve/test_relevance.py::test_mean_relevance_excludes_self` |
| **Resilience** | Four CLI subcommands were stubs; `app.experiment.Experiment.run` was a no-op; data CLI extract/build were stubs; Amazon URL pointed at legacy mirror | CLI fully wired (`train`, `eval`, `bench`, `reproduce`, `render-fidelity`); `Experiment.run` writes the artifact bundle; data extract/build implemented; default URL is Amazon-Reviews-2023 | `morel train completion` exits 0 with `runs/<ts>/config.yaml` and `metrics.jsonl`; `morel reproduce configs/synthetic.yaml` works; `morel data extract --synthetic` produces `features.npz` |
| **Performance** | Python loops in BPR negatives, mean_relevance, encode_subgraph | Vectorised BPR negatives, vectorised mean_relevance with batched matmul, padded-batched subgraph encoding | `tests/unit/recommend/test_recommend.py::test_negatives_vectorised_matches_per_user`; `tests/unit/retrieve/test_relevance.py::test_mean_relevance_vectorised_matches_python`; padded-batched `_encode_subgraph` keeps the integration tests green |
| **Testing** | 228 tests but real Pipeline never driven by Completion | 248 tests including `tests/integration/test_pipeline_with_completion.py` driving the real Pipeline through `Completion.fit` | `pytest -q` exits 0 |
| **Reproducibility** | `morel render-fidelity` did not exist; `morel reproduce` was a stub; fidelity registry was empty | `morel render-fidelity` writes 23 entries from a registered registry; `morel reproduce <yaml>` runs Experiment.run from a saved YAML; manifest sidecar binding enforced | `morel render-fidelity` produces non-empty FIDELITY.md/FIDELITY.json |
| **API** | `from morel import Config, Pipeline, seed` failed; API.md listed non-existent symbols | `morel/__init__.py` re-exports the canonical surface; API.md lists actual `__all__` of every module | `python -c "from morel import Config, Pipeline, seed_everything, Output"` exits 0 |
| **Packaging** | `weights_only=False` in serve + checkpoint | `safe_load` + `unsafe_load` opt-in; device-mismatch cache move | `pip-audit` clean (existing); `tests/unit/train/test_checkpoint.py::test_safe_load_rejects_exploit_payload` |
| **Documentation** | PRODUCTION_READINESS was a phase-report; FIDELITY was hand-edited; REPRODUCE described a fictional run; API.md listed non-existent symbols | All four documents re-rendered from real state; FIDELITY auto-renders from the registry | `morel render-fidelity docs/FIDELITY.md docs/FIDELITY.json` exits 0 with populated table |
| **Paper fidelity** | Registry existed but no entries registered; PRODUCTION_READINESS.md claimed components that were EXACT or APPROXIMATE without proof | 23 components registered with `paper`, `equation`, `implementation`, `test`, `deviation` | `tests/research/test_fidelity_report.py` (introduced in Phase 6) walks every entry and asserts the referenced test exists |
| **Serve** | `weights_only=False` in loader; stub `/v1/recommend` | Hardened loader; live `/v1/recommend` on real LightGCN | `tests/unit/serve/test_serve.py`; `tests/unit/serve/test_update.py` (Phase 9) |

## Headline numbers

- **248 tests passing** (unit + integration + property + research)
- **23 paper components** registered in the fidelity registry, all
  backed by a passing test
- **62 atomic changes** across 9 phases (see PLAN.md)
- Real `Pipeline` driven through `Completion` end-to-end (P0 fix)
- One canonical public surface via `morel/__init__.py`
- Single config source (`morel.core.config.Config`); manifest-bound
  resume; `seed_everything()` deterministic
- PEP 561 `py.typed` marker; strict mypy (unblocked by Phase 5 numpy
  pinning); ruff + format-check enforced in CI
