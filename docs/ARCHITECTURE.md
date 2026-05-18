# Architecture

This document describes the high-level architecture of the GRE-MC reproduction.

## Data Flow

```
Raw Amazon Data
      |
      v
[Data Pipeline]
  - download.py      : fetch review & metadata JSON
  - features.py      : extract text/visual embeddings
  - graph_builder.py : build user-item & item-item graphs
  - masking.py       : simulate missing modalities
      |
      v
[Training Stage 1: Modality Completion]
  GREMC Model
    |-- retrieval.py   : anchor retrieval + ACS + MAGE
    |-- positional_encoding.py : Laplacian PE
    |-- transformer.py : joint-encoding graph transformer
    |-- codebook.py    : sparse-routing codebook
    |-- decoder.py     : per-modality MLP decoders
    |-- gre_mc.py      : end-to-end composition
  CompletionTrainer (completion_trainer.py)
      |
      v
Completed Features
      |
      v
[Training Stage 2: Downstream Recommendation]
  LightGCN (downstream.py)
      |
      v
[Evaluation]
  Evaluator (evaluator.py) -> Recall@K, NDCG@K
```

## Module Responsibilities

| Module | Purpose |
|--------|---------|
| `rmr.data.download` | Fetch Amazon 5-core review & metadata. |
| `rmr.data.features` | Extract SentenceTransformer & ResNet embeddings. |
| `rmr.data.graph_builder` | Build sparse user-item & item-item adjacency. |
| `rmr.data.masking` | Random modality masking with guaranteed retention. |
| `rmr.data.dataset` | PyTorch Dataset yielding per-item batches. |
| `rmr.models.retrieval` | Anchor retrieval, ACS (Algorithm 1), MAGE (Algorithm 2). |
| `rmr.models.positional_encoding` | Cached Laplacian eigenvector PE. |
| `rmr.models.transformer` | Input embedding + transformer layers + query pooling. |
| `rmr.models.codebook` | Gumbel-Softmax sparse-routing codebook with regularizers. |
| `rmr.models.decoder` | Per-modality MLP decoders. |
| `rmr.models.downstream` | LightGCN for final recommendation. |
| `rmr.models.gre_mc` | End-to-end GRE-MC model wiring retrieval to decoders. |
| `rmr.training.completion_trainer` | Training loop with masked reconstruction loss. |
| `rmr.evaluation.metrics` | Recall@K & NDCG@K implementations. |
| `rmr.evaluation.evaluator` | Batch metric computation across cutoffs. |
