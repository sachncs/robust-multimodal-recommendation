# Design: GRE-MC Reproduction

## Overview
Reproduce the arXiv paper "Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion" (GRE-MC, arXiv:2605.00670v1) in pure Python with PyTorch for the neural model and NumPy/SciPy for graph algorithms, data preprocessing, and evaluation.

## Paper Summary
- **Title**: Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion
- **Authors**: Yuan Li, Jun Hu, Jiaxin Jiang, Bryan Hooi, Bingsheng He (NUS)
- **Problem**: Multimodal recommender systems assume all modalities are present; real-world data has missing modalities (sensor failures, sparse labels, privacy).
- **Solution**: GRE-MC retrieves semantically relevant subgraphs from the full item graph, jointly encodes the query node with retrieved context via a graph transformer, and uses a sparse-routing codebook to compress latent representations for robust modality completion.
- **Key Components**:
  1. Modality-Aware Subgraph Retrieval (ACS + MAGE)
  2. Joint-Encoding Graph Transformer with Laplacian Positional Encodings
  3. Sparse-Routing Codebook (Gumbel-Softmax, Top-P)
  4. Codebook Regularization (KL usage + load balancing)

## Scope
- Full downstream recommender (MIG-GT or minimal LightGCN fallback) and end-to-end evaluation on real Amazon data.
- Full automatic data pipeline from raw Amazon review JSON.

## Project Structure
```
gre-mc/
├── data/
│   ├── download.py          # Fetch Amazon 5-core JSON + metadata
│   ├── features.py          # Extract CNN visual & sentence-transformer text features
│   ├── graph_builder.py     # Build user-item graph → item graph
│   ├── masking.py           # Apply 40% random modality masking
│   └── dataset.py           # PyTorch Dataset for GRE-MC training
├── models/
│   ├── retrieval.py         # ACS + MAGE algorithms
│   ├── positional_encoding.py # Laplacian PE computation
│   ├── transformer.py       # Joint-Encoding Graph Transformer
│   ├── codebook.py          # Sparse-routing codebook (Gumbel-Softmax, Top-P)
│   ├── decoder.py           # Modality-specific MLP decoders
│   └── downstream.py        # MIG-GT (or minimal LightGCN fallback)
├── training/
│   ├── completion_trainer.py
│   └── recommender_trainer.py
├── evaluation/
│   ├── metrics.py           # Recall@K, NDCG@K
│   └── evaluator.py
├── tests/
│   ├── test_retrieval.py
│   ├── test_transformer.py
│   ├── test_codebook.py
│   └── test_shapes.py
├── scripts/
│   ├── train_completion.py
│   ├── train_recommender.py
│   └── evaluate.py
├── demo.py                  # End-to-end CLI
├── README.md
└── requirements.txt
```

## Data Pipeline
1. **Download**: Fetch Amazon review 5-core subsets (Baby, Sports, Clothing) from [Amazon Product Data](https://nijianmo.github.io/amazon/index.html).
2. **Feature Extraction**:
   - Visual: Use a pretrained ResNet-50 (or He & McAuley 2016 CNN if available) to extract 4096-D vectors from product image URLs. If image download is infeasible, fall back to pre-extracted `.pt` files from the MMRec benchmark.
   - Text: Use `sentence-transformers/all-MiniLM-L6-v2` to extract 384-D vectors from product descriptions.
3. **Graph Construction**: Build user-item bipartite graph, then induce item-item graph where edges connect items sharing at least one user.
4. **Masking**: Randomly mask 40% of item modalities such that each item retains at least one modality, producing the indicator matrix **E**.

## Model Architecture
### Retrieval (`retrieval.py`)
- **ACS (Algorithm 1)**: Multi-source BFS from anchor nodes to find a connecting root, then union shortest paths.
- **MAGE (Algorithm 2)**: Greedy add/remove boundary nodes to maximize mean relevance `Phi(i,S)`, while preserving connectivity and all anchors.
- **Relevance Score**:
  ```
  r(i,v) = (sum_{m in M} E_{i,m} E_{v,m} cos(x_i^(m), x_v^(m))) / (sum_{m in M} E_{i,m} E_{v,m})
  ```

### Positional Encoding (`positional_encoding.py`)
- Normalized Laplacian: `L = I - D^{-1/2} A D^{-1/2}`
- Bottom-`k` nontrivial eigenvectors (`k=20`).

### Transformer (`transformer.py`)
- Input embedding: `h_v^(0) = MLP([concat_m tilde(x_v^(m)); p_v])`
  - `tilde(x_v^(m)) = x_v^(m)` if `E_{v,m}=1`, else `0`.
- `L=2` transformer layers, 4 heads, hidden dim `d=128`.
- Query embedding: attention-weighted aggregation over all contextualized tokens.

### Codebook (`codebook.py`)
- Routing weights: `g_i = Softmax((W z_i + epsilon) / tau)`
  - `epsilon = -log(-log(u))`, `u ~ Uniform(0,1)` (Gumbel noise, training only).
  - `tau = 0.5`.
- Top-P selection (`P=4`): `q_i = sum_{e in TopP(g_i)} g_{i,e} c_e`
- Inference: `epsilon = 0`.

### Decoder (`decoder.py`)
- `x_hat_i^(m) = MLP^(m)(q_i)` for each missing modality `m`.

### Loss
- `L_recon = sum_m sum_i ||x_hat_i^(m) - x_i^(m)||^2`
- `L_usage = KL(bar(g) || uniform)` where `bar(g) = (1/B) sum_i g_i`
- `L_load = C * sum_e (bar(hat(g))_e)^2`
- Total: `L = L_recon + lambda_usage * L_usage + lambda_load * L_load`

## Hyperparameters (from paper)
- Hidden dimension: `d = 128`
- Anchors: `K in {5, 10, 20}`
- Codebook size: `C in {10, 50, 100}`
- Expansion iterations: `T = 10`
- Positional encoding dim: `k = 20`
- Transformer layers: `L = 2`, heads = 4
- Top-P: `P = 4`
- Temperature: `tau = 0.5`
- Regularization weights: `lambda_usage, lambda_load in {0.5, 1.0, 2.0}`
- Batch size: `512`
- Dropout: `0.5`
- Optimizer: Adam, learning rate `eta in {0.01, 0.003, 0.001, 0.0003, 0.0001}`
- L2 weight decay: `lambda_L2 in {1e-4, 1e-5, 1e-6}`

## Training & Evaluation
- **Completion Training**: Adam, early stopping on validation loss.
- **Recommendation Training**: Train MIG-GT (or LightGCN fallback) on completed features.
- **Split**: 80% train, 10% validation, 10% test.
- **Metrics**: Recall@10, Recall@20, NDCG@10, NDCG@20. Report mean ± std over 5 random seeds.

## Testing
- Tensor shape invariants for transformer attention maps.
- Forward-pass shape checks for ACS/MAGE output.
- Loss computation correctness (reconstruction + regularizers).
- Gumbel-Softmax differentiability check.
- End-to-end synthetic-data smoke test.

## Extensions (Isolated from Baseline)
- FAISS-based approximate nearest-neighbor retrieval for scalability.
- Gradient checkpointing for memory efficiency.
- Wandb logging hooks.
- Lightweight LightGCN downstream alternative.

## Risks & Gaps
- MIG-GT architecture details may be incomplete in the paper; a LightGCN fallback is prepared.
- Pre-extracted Amazon features (4096-D visual, 384-D text) may need to be fetched from the MMRec benchmark repository if raw image download is infeasible.
- The paper does not specify the exact number of training epochs; early stopping on validation loss will be used.
