# morel — Reproduction

A single command reproduces an experiment deterministically.

## Quick reproduction

```bash
make reproduce
```

This runs the full pipeline on synthetic data and writes a reproducible
artifact bundle to `runs/<timestamp>/`:

```
runs/<timestamp>/
├── config.yaml           # the exact Config used
├── manifest.json         # dataset, version, code hash, seed
├── metrics.jsonl         # per-epoch training metrics
├── morel.log             # structured JSON log
├── checkpoints/          # best.pt and last.pt
├── FIDELITY.md           # rendered from the fidelity registry
├── FIDELITY.json         # machine-readable fidelity report
└── report.md             # tables summarizing the run
```

## Configuration

The reproducer reads a YAML config and overrides it from CLI flags or
environment variables. Example `configs/reproduce.yaml`:

```yaml
seed: 42
device: cpu
data:
  category: Beauty
  min: 5
mask:
  kind: bernoulli
  ratio: 0.4
encode:
  hidden: 128
  layers: 2
  heads: 4
codebook:
  size: 100
route:
  p: 4
  tau: 0.5
complete:
  hidden: 128
recommend:
  embed: 64
  layers: 3
train:
  completion:
    epochs: 100
    batch: 512
  recommendation:
    epochs: 100
    batch: 1024
eval:
  ks: [10, 20]
  robustness: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

## Programmatic

```python
from morel.app import Reproduce

result = Reproduce(config_path="configs/reproduce.yaml", run_dir="runs/exp1").run()
```

## Determinism guarantees

- Same `Config` and `seed` → bitwise-identical outputs (modulo CUDA
  non-determinism, which is disabled by default).
- Resume requires matching `config_hash`; mismatched hashes raise
  `ConfigError`.
- `Manifest` sidecar binding ensures the artifact cannot be silently
  consumed under a different configuration.
