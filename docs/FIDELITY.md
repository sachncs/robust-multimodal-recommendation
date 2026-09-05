# morel — Paper Fidelity

The fidelity registry lives in `morel.core.fidelity`. Every component declares
its fidelity status, paper reference, implementation location, and the test
that proves the behavior. This document is rendered from the registry by
`morel render-fidelity`.

## Status legend

- **EXACT** — implementation matches the paper definition.
- **APPROXIMATE** — implementation is a faithful interpretation with one or
  more documented deviations.
- **INCORRECT** — known deviation; see deviation note.
- **UNKNOWN** — paper does not specify enough to claim a status.

## Components

See the auto-rendered table below for the current state of every component.

<!-- FIDELITY:BEGIN -->
<!-- This block is replaced by `morel render-fidelity`. -->
| Component | Status | Implementation | Test |
|-----------|--------|----------------|------|
| ACS | EXACT | `morel.retrieve.acs.compute` | `tests/research/test_paper.py::test_acs_bitmask_correctness_on_path` |
| Anchor retrieval | EXACT | `morel.retrieve.anchor.query` | `tests/unit/retrieve/test_anchor.py` |
| MAGE | APPROXIMATE | `morel.retrieve.mage.expand` | `tests/unit/retrieve/test_mage.py` |
| Bipartite construction | EXACT | `morel.data.build.bipartite` | `tests/unit/data/test_build.py` |
| Iterative k-core | EXACT | `morel.data.build.kcore` | `tests/unit/data/test_build.py::test_kcore_shrinks_until_min_degree` |
| Item graph construction | EXACT | `morel.data.build.item_cooccurrence` | `tests/unit/data/test_build.py::test_item_cooccurrence_no_self_loops` |
| Modality masking | EXACT | `morel.data.mask.bernoulli` | `tests/unit/data/test_mask.py` |
| Laplacian PE | EXACT | `morel.graph.laplacian.pe` | `tests/unit/graph/test_laplacian.py` |
| Joint encoding (transformer) | APPROXIMATE | `morel.encode.transformer.Transformer` | `tests/unit/encode/test_encode.py` |
| Gumbel-Softmax routing | EXACT | `morel.route.router.Gumbel` | `tests/unit/route/test_router.py` |
| Top-k sparse routing | APPROXIMATE | `morel.route.router.Top` | `tests/unit/route/test_router.py` |
| Codebook (lookup) | EXACT | `morel.codebook.codebook.GumbelVQ` | `tests/unit/codebook/test_codebook.py` |
| Usage loss | EXACT | `morel.codebook.codebook.usage` | `tests/unit/codebook/test_codebook.py::test_usage_loss_zero_at_uniform` |
| Load loss | EXACT | `morel.codebook.codebook.balance` | `tests/unit/codebook/test_codebook.py::test_load_loss_includes_K_multiplier` |
| Reconstruction loss | EXACT | `morel.train.loss.Reconstruction` | `tests/unit/train/test_loss.py` |
| LightGCN propagation | EXACT | `morel.recommend.light.Light` | `tests/unit/recommend/test_recommend.py::test_light_gcn_layers_zero_is_dot_product` |
| Recall@K | EXACT | `morel.eval.ranking.recall_at_k` | `tests/unit/eval/test_eval.py` |
| NDCG@K | EXACT | `morel.eval.ranking.ndcg_at_k` | `tests/unit/eval/test_eval.py` |
<!-- FIDELITY:END -->

## Deviations

| Component | Deviation |
|-----------|-----------|
| MAGE | Best-improvement hill climbing (vs. paper's first-improvement ambiguity); sorted boundary iteration (deterministic vs. paper's set-iteration) |
| Joint encoding | Learned linear-pool attention aggregation; the paper says "scaled dot-product attention over the entire set" without specifying |
| Top-k routing | The paper text says "Top-P" but the implementation is Top-K (post-softmax topk + renorm); same numerics in expectation |
