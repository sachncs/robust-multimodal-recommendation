# morel — Reproduction

A single command reproduces an experiment deterministically.

## Quick reproduction (synthetic)

```bash
python -m morel train completion
```

This runs the full pipeline on a synthetic dataset and writes a
reproducible artifact bundle under `runs/<timestamp>/`:

```
runs/<timestamp>/
├── config.yaml           # the exact Config used
├── manifest.json         # dataset, version, code hash, seed
├── metrics.jsonl         # per-step metrics
├── FIDELITY.md           # rendered from the fidelity registry
├── FIDELITY.json         # machine-readable fidelity report
├── report.md             # one-paragraph summary of the run
└── checkpoints/
    └── best.pt           # best validation checkpoint
```

## Reproducing from a saved config

```bash
python -m morel reproduce path/to/config.yaml --items 100 --users 30 --epochs 5
```

The `Reproduce` service reads the YAML, sets the seed, validates the
manifest's `config_hash` if present, and delegates to `Experiment.run`
for full reproducibility.

## Determinism guarantees

- `morel.core.seed.seed(value)` configures deterministic seeding for
  torch, torch CUDA, numpy, Python `random`, `PYTHONHASHSEED`, and
  cuDNN. The CLI calls it at startup.
- Resuming a run requires the saved `config_hash` to match the new
  Config; mismatched hashes raise `ConfigError`.
- `Manifest` sidecar binding ensures the artifact cannot be silently
  consumed under a different configuration.

## Programmatic

```python
from morel.app import Reproduce

result = Reproduce(
    config_path="runs/<ts>/config.yaml",
    run_dir="runs/<ts>",
).run()
```

## Limitations

- The default synthetic reproduction uses random features and a small
  synthetic bipartite graph. Real-data reproduction requires either
  `morel data download` (Amazon-Reviews-2023 mirror) or a manually
  curated `data/processed/<dataset>` directory matching the schema.
- See `LIMITATIONS.md` for the published reproducibility scope.
