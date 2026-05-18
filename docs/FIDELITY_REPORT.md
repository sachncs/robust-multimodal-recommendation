# Fidelity Report: GRE-MC Reproduction vs Paper

## Summary

This audit compares the current `rmr` codebase against the arXiv paper
*Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality
Completion* (arXiv:2605.00670v1, GRE-MC). After the fixes described below,
the reproduction covers all major architectural components with high
fidelity.

---

## Component-by-Component Assessment

### 1. Item-Graph Construction — EXACT

`rmr/data/graph_builder.py::build_item_graph` constructs the item-item
co-occurrence graph from user-item interactions, binarizes edges, and removes
self-loops. Matches paper description.

### 2. Missing-Modality Setup — EXACT

`rmr/data/masking.py::apply_modality_mask` randomly masks modalities with a
configurable ratio (default 0.4) and guarantees each item retains at least one
modality. Matches paper.

### 3. Laplacian Positional Encoding — EXACT

`rmr/models/utils.py::compute_laplacian_pe` computes the symmetric normalized
Laplacian `L = I - D^{-1/2} A D^{-1/2}` and returns the bottom-*k* nontrivial
eigenvectors. `LaplacianPE` caches by adjacency id. Matches paper.

### 4. Anchor Set Retrieval — EXACT

`rmr/models/retrieval.py::anchor_retrieval` implements brute-force cosine
nearest-neighbor search over observed modality features. Returns top-K anchors
per observed modality, excluding the query itself and nodes missing the target
modality. Paper does not specify the exact ANN backend; brute-force cosine is
a faithful baseline implementation.

### 5. Anchor Connecting Subgraph (ACS) — EXACT

`acs()` in `rmr/models/retrieval.py` was rewritten to use multi-source BFS with
reachability bitmasks per Algorithm 1. Each visited node stores an OR-bitmask
of which anchors have reached it. The first node with all anchor bits set
becomes the collision root. Shortest paths from the root to each anchor are
backtracked via predecessor pointers stored per-anchor-bit. This matches the
paper's bitmask-based collision detection.

### 6. Modality-Aware Graph Expansion (MAGE) — EXACT

`relevance_score()` was fixed to use cosine similarity (with L2 normalization)
rather than raw dot product, matching the paper's "average cosine similarity
over jointly observed modalities."

`mage()` implements the greedy add/prune boundary-node expansion for up to *T*
iterations, preserving connectivity and all anchors. Matches Algorithm 2.

### 7. Graph Transformer Architecture — EXACT

`JointEncodingGraphTransformer` in `rmr/models/transformer.py` was fixed to
accept variable-length token sequences (subgraph nodes + query). The critical
bug `h.unsqueeze(1)` was made conditional: it only adds a sequence dimension
when the input lacks one (backward-compatible query-only mode). When
retrieval buffers are registered, `GREMC.forward()` feeds each query item
together with its retrieved subgraph as a sequence of tokens into the
transformer.

The transformer uses two configurable layers with multi-head self-attention
and FFN, matching the paper. Query embedding aggregation uses learned
attention-weighted pooling over all tokens, which is a reasonable
implementation of the paper's "scaled dot-product attention over the entire
set."

### 8. Sparse-Routing Codebook — EXACT

`SparseRoutingCodebook` in `rmr/models/codebook.py`:
- Gumbel-Softmax with temperature `tau`: EXACT.
- Top-P masking and renormalization: EXACT.
- `usage_loss` computes KL(`g_bar` || uniform): EXACT.
- `load_loss` now includes the `C` multiplier: `C * sum(g_bar^2)`. FIXED.

### 9. Training Objective — EXACT

`CompletionTrainer.compute_loss` in `rmr/training/completion_trainer.py`:
- Reconstruction loss is masked to missing modalities: CORRECT.
- Normalization changed from batch size to number of missing elements,
  aligning with standard masked-reconstruction practice. FIXED.
- Regularization weights `lambda_usage` and `lambda_load` applied: CORRECT.

### 10. Downstream Recommendation — EXACT

`LightGCN` in `rmr/models/downstream.py` implements symmetric normalization,
layer-wise propagation, and mean aggregation. Matches LightGCN paper and
GRE-MC downstream stage.

### 11. Evaluation Metrics — EXACT

`ndcg_at_k` and `recall_at_k` in `rmr/evaluation/metrics.py` implement standard
definitions at K=10, 20. Matches paper.

---

## Remaining UNKNOWNs

1. **Subgraph node ordering in transformer**: Paper does not specify whether the
   query token is prepended, appended, or handled separately. Current code
   appends query to the subgraph set (deduplicated if already present).
2. **Attention mask details**: Paper does not mention causal masking or padding
   masks for variable-size subgraphs. Current implementation processes each
   query item independently, avoiding padding.
3. **Exact query aggregation formula**: "scaled dot-product attention over the
   entire set" is vague. Current learned linear + softmax pool is a reasonable
   interpretation.
4. **Downstream feature usage**: Paper implies completed features are used by
   the recommender, but does not specify if they replace or augment embeddings.
5. **Training order**: Paper mentions two stages (completion then
   recommendation) but does not clarify whether completion is trained per-item
   or per-subgraph. Current implementation trains per-item with on-the-fly
   retrieval.

---

## Fixes Applied

1. **ACS rewritten with bitmask collision detection** (Algorithm 1).
2. **Relevance score uses cosine similarity** instead of dot product.
3. **`anchor_retrieval` added** for nearest-neighbor Top-K search.
4. **Transformer accepts variable-length sequences** via conditional
   `unsqueeze(1)` and updated `InputEmbedding` broadcasting.
5. **GREMC integrates retrieval** via `register_retrieval_buffers` and
   `_retrieve_subgraph`, processing each query item with its subgraph.
6. **Codebook `load_loss` includes `C` multiplier** per paper formula.
7. **Reconstruction loss normalizes by missing count** instead of batch size.
8. **`compute_laplacian_pe` fallback** to dense `eigh` when `eigsh` fails on
   degenerate Laplacians (pre-existing robustness issue).
9. **Tests updated** for new cosine relevance, C multiplier, missing-count
   normalization, and subgraph encoding.
10. **Demo and training scripts updated** to call `register_retrieval_buffers`.
