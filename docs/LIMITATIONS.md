# morel — Limitations

This document enumerates known boundaries of the implementation.

## Out of scope (v0.1.0)

- **Distributed training** — single-process; multi-GPU via DDP is a future
  extension.
- **Streaming ingestion** — the data pipeline assumes the full dataset fits
  in memory for the k-core and graph construction steps.
- **Real-time fine-tuning** — the serve stack is for inference only.
- **Modality auto-detection** — modality names and dimensions are
  configured; there is no inference of modality presence from data.
- **Multilingual sentence encoders** — only `sentence-transformers` is
  supported; custom encoders plug in via the `Encoder` Protocol.

## Approximations vs. paper

See `FIDELITY.md`. Notable:

- **MAGE** — best-improvement hill climbing + sorted boundary (vs. paper's
  ambiguous "greedy").
- **Joint encoding** — learned linear-pool attention vs. paper's vague
  "attention over the entire set".
- **Top-P** — implemented as Top-K (post-softmax topk + renorm); same
  numerics in expectation.

## Numerical edge cases

- Single-node graphs: PE is all zeros; downstream transformer handles
  via mask broadcasting.
- Empty adjacency: warnings logged; downstream receives a zero PE.
- Disconnected anchors: ACS logs a warning and returns either the anchor
  set or an empty set, depending on `Config.retrieve.acs_fallback`.

## Determinism caveats

- CUDA non-determinism in `torch.sparse.mm` requires explicit
  `torch.use_deterministic_algorithms(True)` if exact bitwise reproducibility
  is required on GPU.
- The fixture `cudnn.deterministic = True` may slow down training on GPU.
- Some `torch` versions do not roundtrip `set_rng_state` bytewise; the
  `state()` snapshot includes the RNG state but exact reproduction may
  require re-running with the same `torch` version.

## Roadmap

- v0.2.0: distributed training, sparse-graph acceleration
- v0.3.0: streaming ingestion, conditional modalities
- v1.0.0: production hardening, audit-ready benchmarks
