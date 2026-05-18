# Usage

## Quick Demo

Run a complete end-to-end pipeline on synthetic data:

```bash
python demo.py
```

## Stage 1: Train Modality Completion

```bash
python -m rmr.scripts.train_completion \
  --data-dir data/processed \
  --hidden-dim 128 \
  --num-layers 2 \
  --num-heads 4 \
  --codebook-size 100 \
  --top-p 4 \
  --tau 0.5 \
  --pe-dim 20 \
  --dropout 0.5 \
  --lr 0.001 \
  --batch-size 512 \
  --epochs 100 \
  --device cpu
```

## Stage 2: Train Downstream Recommender

```bash
python -m rmr.scripts.train_recommender \
  --data-dir data/processed \
  --embedding-dim 64 \
  --num-layers 3 \
  --lr 0.001 \
  --batch-size 1024 \
  --epochs 100 \
  --device cpu
```

## Evaluation

```bash
python -m rmr.scripts.evaluate \
  --predictions results/scores.npy \
  --labels results/labels.npy
```

## Running Tests

```bash
pytest tests/ -v --cov=rmr --cov-report=term-missing
```

## Running the Linter

```bash
ruff check .
```

## Inference / Modality Completion

```python
import numpy as np
import scipy.sparse as sp
import torch
from rmr.models.gre_mc import GREMC

# Load your data
features = {
    "visual": np.load("visual_features.npy"),
    "text": np.load("text_features.npy"),
}
mask = np.load("mask.npy")
adj = sp.load_npz("item_graph.npz")

model = GREMC(
    input_dims={"visual": 4096, "text": 384},
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    codebook_size=100,
    top_p=4,
    tau=0.5,
    pe_dim=20,
)
model.register_retrieval_buffers(features, mask)
model.load_state_dict(torch.load("checkpoints/completion_best.pt"))
model.eval()

# Batch inference
feats_torch = {k: torch.from_numpy(v) for k, v in features.items()}
mask_torch = torch.from_numpy(mask)
with torch.no_grad():
    completed, routing = model(feats_torch, mask_torch, adj, training=False)
```
