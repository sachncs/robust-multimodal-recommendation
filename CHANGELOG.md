# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Rebrand: `rmr` → `morel` (MOdality-REcommended Links).
- Single-word naming convention applied throughout (no `_` prefixes, no suffixes).
- Module structure reorganized around explicit domain boundaries (core → data → model → train → eval → app → cli).
- Strict layering enforced by `tests/unit/test_architecture.py` (stdlib `ast` import-graph check).
- Strict mypy mode for `morel/`.

### Added
- `morel.core` runtime primitives: `Config`, `seed`, `Device`, `Log`, `Fidelity` registry.
- `morel.data` lifecycle: `acquire`, `validate`, `extract`, `build`, `mask`, `store`, `Manifest`.
- `morel.graph` polymorphism: `Graph`, `Bipartite`, `Item`, `Subgraph`, `Laplace`.
- `morel.retrieve` correctness fixes: iterative ACS, best-improvement MAGE, sorted boundary.
- `morel.encode`, `morel.route`, `morel.codebook`, `morel.complete`, `morel.recommend`.
- `morel.pipeline` end-to-end orchestration.
- `morel.train`, `morel.eval`, `morel.app`, `morel.cli`, `morel.serve`.

### Removed
- `.pre-commit-config.yaml` (checks now enforced in CI).
- `rmr/` package (legacy stub remains for one release cycle).

## [0.3.0] - 2026-05-18

### Added
- Initial GRE-MC reproduction baseline (under `rmr`).
- Item-graph construction, missing-modality masking, Laplacian PE.
- Graph transformer, sparse-routing codebook, modality decoders.
- LightGCN downstream recommender.
- Training and evaluation scripts.
- Test suite.

[Unreleased]: https://github.com/sachncs/robust-multimodal-recommendation/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/sachncs/robust-multimodal-recommendation/releases/tag/v0.3.0
