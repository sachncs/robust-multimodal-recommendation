# morel — Method

Mathematical pipeline of the GRE-MC paper as implemented.

## Notation

- `U`: bipartite user-item matrix `(n_users, n_items)`, CSR.
- `I = sign(U^T U)`: item-item co-occurrence (symmetric, no self-loops).
- `M`: modality availability mask `(n_items, M)`, `1 = kept`, `0 = missing`.
- `f^{(m)}_i`: feature of item `i` in modality `m`, L2-normalized.
- `e_i`: feature matrix in modality `m`.
- `L = I - D^{-1/2} A D^{-1/2}`: symmetric normalized Laplacian.

## Stage 1: Anchor retrieval

For each query item `q` and observed modality `m`, retrieve the top-K
cosine-NN over items with `M[·, m] = 1` (excluding `q` itself):

```
sim(q, v) = <f^(m)_q, f^(m)_v> / (||f^(m)_q|| * ||f^(m)_v||)
```

Union of per-modality top-K sets is the anchor set `A_q`.

## Stage 2: Anchor Connecting Subgraph (ACS)

Algorithm 1: multi-source BFS with reachability bitmask. Each visited
node carries an OR-bitmask of which anchors have reached it. The first node
whose bitmask equals `(1 << |A|) - 1` is the collision root. The ACS is the
union of shortest paths from the collision root back to each anchor.

Iterative backtrack (no recursion-limit crash on long paths). Duplicate and
out-of-range anchor rejection. Self-loop guard on the adjacency.

## Stage 3: Modality-Aware Graph Expansion (MAGE)

Algorithm 2: greedy boundary add/remove. At each iteration:

1. **Add**: pick the boundary node whose inclusion maximizes mean relevance
   while preserving connectivity. Best-improvement (canonical Alg 2).
2. **Remove**: if no add improves, try removing a non-anchor node whose
   removal improves mean relevance.

Relevance:

```
r(i, v) = (1/|S(i) ∩ S(v)|) * Σ_{m in S(i) ∩ S(v)} cos(f^(m)_i, f^(m)_v)
```

Sorted boundary iteration → deterministic across runs (set-iteration order
removed).

## Stage 4: Joint encoding

Per query item, gather the subgraph's features, mask, and Laplacian PE.
Encode with an L-layer Pre-LN Transformer with attention pooling:

```
z = Pool(TransformerStack(features, mask, pe))
```

Pre-LN (norm-before-residual) for stable training. Scaled dot-product
attention pooling with `1/sqrt(d)` scale.

## Stage 5: Sparse routing

Top-k sparsification over codebook logits:

```
g = softmax((W z + gumbel) / τ)
g_top = topk(g, p) / sum(topk(g, p))
q = g_top @ codebook
```

Gumbel noise only during training; deterministic at eval. Load-balancing
loss `L_load = C * sum_e bar_g_e^2` (with `C` multiplier per paper).

## Stage 6: Modality completion

Per-modality MLP decoders from latent `q`:

```
f̂^(m)_i = MLP^(m)(q_i)
```

A learned `[MASK]` token replaces zero-multiplication when a modality is
missing, following BERT-style masked reconstruction.

Loss: masked MSE normalized per modality by **missing elements**:

```
L_recon = Σ_m sum_{i: M[i,m]=0} ||f̂^(m)_i - f^(m)_i||^2 / (sum_miss * D_m)
```

## Stage 7: Downstream recommendation

LightGCN propagation on `U`:

```
H^0 = [user_emb; item_emb]
H^{l+1} = Â H^l
H_final = (1/(L+1)) Σ_l H^l
score(u, i) = <H_final[u], H_final[i]>
```

with `Â = D^{-1/2} A D^{-1/2}` cached at construction.

BPR loss for training with **strict** negative sampling (never returns a
positive item; raises if no negatives exist for any user).
