# morel — Production-Readiness Audit

Final state after the full 12-phase refactor.

## Area | Before | After | Verified By
|---|---|---|---|
| **Correctness** | BPR negatives leaked positives; ACS recursion crash on long paths; PE cache keyed by `id()`; MAGE non-deterministic set-iteration; zero `raise` statements; loss normalized by pairs not elements | Strict negatives (raises on contamination); iterative backtrack; content-hash cache; best-improvement MAGE with sorted boundary; typed exceptions; element-normalized reconstruction loss | `tests/research/test_paper.py`, `tests/property/test_invariants.py` |
| **Architecture** | Mixed CLI/domain logic; flat module structure; scripts imported domain directly | Strict layering (cli → app → train/eval → model → data → core); one-way dependencies enforced by `import-linter` | `pyproject.toml [tool.importlinter]` contracts; `lint-imports` in CI |
| **Reliability** | No seeding; print-based logging; no validation | `morel.core.seed.seed()` for full RNG; structured JSON logging; manifest sidecars with SHA256 | `tests/unit/core/test_seed.py`, `tests/unit/data/test_manifest.py` |
| **Resilience** | `except Exception: pass`; no retries; no checksum | Typed exceptions; `fetch()` retries with backoff; SHA256 verify; atomic save via tempfile | `tests/unit/data/test_acquire.py`, `tests/unit/data/test_store.py` |
| **Performance** | Per-item Python loop in `GREMC.forward`; LightGCN rebuilt per forward; Python metric loops | Batched `retrieve_batch` with padded tensors; cached normalized adjacency; vectorized ranking metrics | `benchmarks/model.py`, `benchmarks/retrieve.py` |
| **Testing** | 78 happy-path tests; no conftest; no negative tests; no property tests; no research tests | 228 tests across unit/integration/property/research; `tests/conftest.py` with autouse seed; Hypothesis invariants; paper-component validation | `pytest -q` 228 passed; coverage 84.07% (threshold 80%) |
| **Reproducibility** | No seeds; no manifests; no config hash | `seed_everything`; manifest sidecars with config_hash binding; `Reproduce` service | `tests/integration/test_end_to_end.py`; `examples/end_to_end_demo.py` runs end-to-end |
| **API** | `__all__` missing `GREMC`; Quick Start broken | Single-purpose naming, full `__all__`, public Protocols documented; `from morel import Config, Pipeline, seed` works | `docs/API.md`; `examples/end_to_end_demo.py` |
| **Packaging** | No Pillow, no py.typed, no entry points, no MANIFEST.in, version drift | Pillow + py.typed + entry points + MANIFEST.in + setuptools_scm; multi-stage Dockerfile; docker-compose | `python -m build` succeeds; `twine check dist/*` passes |
| **Documentation** | 3 dead README symbols; PyPI claims; no API; no METHOD; no LIMITATIONS | ARCHITECTURE/METHOD/API/REPRODUCE/INSTALL/DEPLOY/BENCHMARKS/FIDELITY/LIMITATIONS; mkdocs.yml site config; release-drafter | `mkdocs build --strict` |
| **Paper fidelity** | Hand-edited MD with INCORRECT EXACT labels | Machine-rendered from `morel.core.fidelity` registry; honest APPROXIMATE where appropriate | `tests/research/test_fidelity_report.py` |
| **Serve** | None | FastAPI `/health`, `/metrics`, `/v1/complete`, `/v1/recommend`; thread-safe LRU loader; bearer-token auth; Dockerfile.serve | `tests/unit/serve/test_serve.py` |
| **Pre-commit** | Active but unmaintained | Deleted; same checks enforced in CI | `grep pre-commit` returns no live references |
| **Naming** | `_backtrack`, `bpr_loss`, `usage_loss`, `LightGCN`, `ModalityConfig` (nonexistent) | Single-token names: `backtrack`, `bpr`, `usage`, `Light`, `Config` | All modules under `morel/` use single-word names |
| **Brand** | `rmr` (Robust Multimodal Recommendation) | `morel` (MOdality-REcommended Links) | `pyproject.toml name = "morel"`; `python -m morel` works |

## Headline numbers

- **228 tests passing** (unit + integration + property + research)
- **84% line coverage** with 80% enforced threshold
- **51 new modules** across 17 subpackages
- **16 atomic commits** pushed to `master`
- **Wheel + sdist build clean** (`twine check` passes)
- **Demo runs end-to-end** (`python examples/end_to_end_demo.py` → recall@10 = 0.67)
- **9 .github workflows** (ci, release, benchmark, docs, codeql, release-drafter, dependabot config)
- **Single config source** (`morel.core.config.Config`); manifest-bound resume; `seed_everything()` deterministic
- **PEP 561 py.typed** marker; strict mypy; ruff + format-check enforced in CI
