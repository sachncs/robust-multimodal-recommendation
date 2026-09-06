# morel — API

The morel API is the union of each package's `__all__`. This page is
machine-generated from the actual `__all__` of every module so it cannot
drift.

## morel

```python
import morel
morel.__version__
from morel import Config, Pipeline, Output, seed_everything
```

`__all__`: `Config`, `Embedding`, `Graph`, `Manifest`, `Mask`, `Modality`,
`Output`, `Pipeline`, `configure_log`, `get_logger`, `seed_everything`.

## morel.core

```python
from morel.core import (
    Config,
    Device, device, to,
    Modality, Mask, Graph, Embedding,
    MorelError, DataError, ConfigError, ModelError, GraphError,
    TrainError, EvalError, ShapeError, DeterminismError,
    SeedState, seed_everything, seed_state, seed_restore,
    configure_log, get_logger, log_metrics,
    root, raw, processed, features, graphs, checkpoints, runs, manifest,
    FidelityEntry, FidelityStatus,
    fidelity_register, fidelity_all, fidelity_clear,
    fidelity_render_markdown, fidelity_render_json,
)
```

## morel.data

```python
from morel.data import (
    Manifest, save_manifest, load_manifest, manifest_path, checksum,
    download, fetch, download_legacy,
    Encoder, fingerprint, random, text, visual,
    bipartite, item_cooccurrence, kcore, interactions,
    Mask, bernoulli, block, structured, stack,
    save_npz, load_npz, save_graph, load_graph,
    features, graph, validate_interactions, validate_mask,
)
```

## morel.graph

```python
from morel.graph import Bipartite, Item, Subgraph, Laplace, laplacian, pe, connected
```

## morel.retrieve

```python
from morel.retrieve import (
    Result,
    acs, acs_batch,
    anchor, anchor_batch,
    mage, mage_batch,
    relevance, mean_relevance,
    bfs, path, neighbor_iter, neighbor_array,
    retrieve, retrieve_batch, as_tensor,
)
```

## morel.encode

```python
from morel.encode import (
    Encoder, Baseline, Identity, Sum,
    Input, Layer, Transformer,
    Attention, Mean, Token, CLS,
)
```

## morel.route

```python
from morel.route import Router, Weights, Dense, Top, Gumbel, Fixed, build
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
    Trainer, Completion, CompletionConfig,
    Recommendation, RecommendationConfig,
    State, hash_config, safe_load, unsafe_load, Monitor,
    Loss, Reconstruction, BPR, Composite, ce,
)
```

## morel.eval

```python
from morel.eval import (
    recall_at_k, ndcg_at_k, precision_at_k, map_at_k, mrr,
    mse, per_modality_mse, explained_variance,
    RobustnessResult, robustness_sweep, ablation_results,
)
```

## morel.app

```python
from morel.app import Experiment, Benchmark, Reproduce
```

## morel.serve

```python
from morel.serve import (
    create, Loader,
    CompleteRequest, CompleteResponse,
    RecommendRequest, RecommendResponse, RecommendItem,
    HealthResponse, serialize_completed,
    auth_enabled, auth_assert_configured,
)
```

## morel.cli

```bash
morel data download [--category CATEGORY] [--dest DIR] [--legacy]
morel data extract [--data-dir DIR] [--out-dir DIR] [--config CFG] [--synthetic]
morel data build [--data-dir DIR] [--out-dir DIR] [--config CFG] [--synthetic]
morel data mask --items N --modalities M [--ratio R] --out FILE
morel data verify [--data-dir DIR]

morel train completion [--config CFG]
morel train recommendation [--config CFG]

morel eval rank
morel eval robustness

morel bench [--sizes "8,16,32"] [--epochs N]

morel reproduce CONFIG.yaml [--items N] [--users N] [--epochs N]

morel render-fidelity OUT.md [OUT.json]

morel serve [--host HOST] [--port PORT]
```
