# morel — API

Auto-rendered via `mkdocstrings`. Run `make docs` to build the static site.

## morel

```python
import morel
morel.__version__
```

## morel.core

```python
from morel.core import Config, seed_everything, configure_log, get_logger, log_metrics
```

## morel.data

```python
from morel.data import (
    Manifest, save_manifest, load_manifest,
    download, fetch, checksum,
    bipartite, item_cooccurrence, kcore, interactions,
    Mask, bernoulli, block, structured, stack,
    save_npz, load_npz, save_graph, load_graph,
)
```

## morel.graph

```python
from morel.graph import Bipartite, Item, Subgraph, Laplace, laplacian, pe, connected
```

## morel.retrieve

```python
from morel.retrieve import (
    acs, mage, anchor, relevance, mean_relevance,
    bfs, path, retrieve, retrieve_batch, Result,
)
```

## morel.encode

```python
from morel.encode import Transformer, Attention, Mean, Token, Layer, Input, Baseline
```

## morel.route

```python
from morel.route import Top, Dense, Gumbel, Fixed, Router, Weights, build
```

## morel.codebook

```python
from morel.codebook import VQ, GumbelVQ, usage, balance
```

## morel.complete

```python
from morel.complete import Decoders
```

## morel.recommend

```python
from morel.recommend import Light, MF, Pop, bpr, negatives
```

## morel.pipeline

```python
from morel.pipeline import Pipeline, Output
```

## morel.train

```python
from morel.train import (
    Trainer, Completion, Recommendation,
    Checkpoint, Monitor, Loss, Reconstruction, BPR, Composite,
)
```

## morel.eval

```python
from morel.eval import (
    Recall, NDCG, Precision, MAP, MRR,
    Completion, Robustness, Ablation,
    robustness_sweep, ablation_results,
)
```

## morel.serve

```python
from morel.serve import create, Loader
app = create()
```

## morel.cli

```bash
morel data download --category Beauty
morel data extract
morel data build
morel data mask --items 100 --modalities 2 --out mask.npy
morel train completion
morel train recommendation
morel eval rank
morel eval robustness
morel bench
morel reproduce configs/reproduce.yaml
morel serve --host 0.0.0.0 --port 8080
```
