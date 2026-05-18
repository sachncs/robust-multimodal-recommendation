# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Rewrote ACS to use reachability bitmasks per Algorithm 1.
- Fixed relevance score to use cosine similarity instead of raw dot product.
- Fixed transformer to accept variable-length subgraph token sequences.
- Fixed codebook load-balancing loss to include codebook-size multiplier `C`.
- Fixed reconstruction loss normalization to divide by missing count.
- Added fallback to dense eigenvalue solver when `eigsh` fails.

### Added
- `anchor_retrieval` function for Top-K cosine nearest-neighbor search.
- `register_retrieval_buffers` API to GREMC for on-the-fly subgraph retrieval.
- Subgraph retrieval integration into GREMC forward pass.
- Additional tests for anchor retrieval, ACS collision root, sequence masking,
  codebook load-loss multiplier, and retrieval-buffer end-to-end behavior.

### Changed
- Updated `pyproject.toml` ruff configuration to `[tool.ruff.lint]` schema.
- Modernized type annotations (`dict`, `list`, `set`, `tuple`) across codebase.

## [0.1.0] - 2026-05-18

### Added
- Initial GRE-MC reproduction baseline.
- Item-graph construction, missing-modality masking, Laplacian PE.
- Graph transformer, sparse-routing codebook, modality decoders.
- LightGCN downstream recommender.
- Training and evaluation scripts.
- Comprehensive test suite.
