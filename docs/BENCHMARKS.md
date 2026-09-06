# morel — Benchmarks

Benchmarks use `pytest-benchmark`. Run with:

```bash
pytest benchmarks/ --benchmark-only --benchmark-min-rounds=20 --benchmark-autosave
```

## Suites

- `benchmarks/model.py` — forward-pass latency for the encoder + codebook +
  decoder at scales `[1k, 10k, 100k]` items.
- `benchmarks/data.py` — full k-core + graph construction end-to-end.
- `benchmarks/retrieve.py` — anchor + ACS + MAGE throughput.
- `benchmarks/end_to_end.py` — completion training epoch on a 50k-item
  synthetic dataset.

## Threshold

CI runs `pytest benchmarks/` weekly and compares against the previous run.
A 20% regression in any benchmark fails the build.
