# Frequently Asked Questions (FAQ)

## General

### What is RMR?

RMR (Robust Multimodal Recommendation) is a reproduction of the paper "Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion" (GRE-MC) by Li et al. (NUS, 2026). It addresses missing modalities in multimodal recommender systems.

### What problem does RMR solve?

RMR solves the problem of missing modalities in multimodal recommender systems. In real-world scenarios, items often have incomplete feature sets (e.g., missing images or text descriptions), which can significantly degrade recommendation quality.

### How does RMR work?

RMR uses a graph retrieval-enhanced approach:
1. **Modality-Aware Subgraph Retrieval** - Retrieves relevant subgraphs from the item-item graph
2. **Graph Transformer** - Encodes query nodes with retrieved context
3. **Sparse-Routing Codebook** - Compresses latent representations
4. **Modality Completion** - Reconstructs missing modalities
5. **Downstream Recommendation** - LightGCN trained on completed features

## Installation & Setup

### What Python version do I need?

RMR requires Python 3.10 or higher. We recommend using Python 3.11 or 3.12 for best compatibility.

### Can I use RMR on Windows?

Yes, RMR is compatible with Windows. However, some features like CUDA acceleration may require additional setup.

### Do I need a GPU?

No, RMR can run on CPU. However, GPU acceleration (CUDA) is recommended for training on large datasets.

### How do I install dependencies?

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode with development dependencies.

## Data

### What datasets does RMR support?

RMR is designed for the Amazon review dataset (5-core). The data pipeline includes:
- Downloading Amazon review and metadata
- Extracting text and visual features
- Building user-item and item-item graphs
- Applying modality masking

### How do I prepare my data?

Follow the data preparation steps in [docs/SETUP.md](SETUP.md):

```bash
python -m rmr.data.download
python -m rmr.data.features
python -m rmr.data.graph_builder
python -m rmr.data.masking
```

### Can I use my own dataset?

Yes, you can adapt RMR for custom datasets by modifying the data pipeline modules in `rmr/data/`. The core model components are dataset-agnostic.

## Training

### How long does training take?

Training time depends on:
- Dataset size
- Hardware (CPU vs GPU)
- Model configuration

On a modern CPU, training may take several hours. GPU acceleration can significantly reduce training time.

### What are the two training stages?

1. **Stage 1: Modality Completion** - Trains the GRE-MC model to reconstruct missing modalities
2. **Stage 2: Downstream Recommendation** - Trains LightGCN on the completed features

### How do I monitor training progress?

Training scripts output progress bars and metrics. You can also use tensorboard or similar tools by modifying the training scripts.

## Evaluation

### What metrics does RMR use?

RMR uses standard recommendation metrics:
- **Recall@K** - Proportion of relevant items in top-K recommendations
- **NDCG@K** - Normalized Discounted Cumulative Gain at K

### How do I evaluate my model?

```bash
python -m rmr.scripts.evaluate \
  --predictions results/scores.npy \
  --labels results/labels.npy
```

## Technical

### What is the Sparse-Routing Codebook?

The Sparse-Routing Codebook is a key component that compresses latent representations using Gumbel-Softmax and Top-P selection. It learns to route input features to a sparse set of codebook entries.

### What is Laplacian Positional Encoding?

Laplacian Positional Encoding (PE) is a graph-based positional encoding computed from the eigenvectors of the graph Laplacian. It provides structural information about node positions in the graph.

### How does the graph retrieval work?

The retrieval system uses:
1. **Anchor Retrieval** - Finds top-K nearest neighbors using cosine similarity
2. **Anchor Connecting Subgraph (ACS)** - Connects anchors via shortest paths
3. **Modality-Aware Graph Expansion (MAGE)** - Expands the subgraph based on modality relevance

## Troubleshooting

### I get a "CUDA out of memory" error

Try reducing batch size or using CPU:
```bash
--batch-size 256 --device cpu
```

### Tests fail with import errors

Ensure you've installed the package in editable mode:
```bash
pip install -e ".[dev]"
```

### Ruff reports linting errors

Run the formatter:
```bash
ruff format .
ruff check --fix .
```

### Training loss doesn't decrease

Common causes:
- Learning rate too high/low
- Batch size too small
- Insufficient epochs
- Data preprocessing issues

## Contributing

### How do I contribute?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

### Where do I report bugs?

Use the [bug report template](https://github.com/sachin/rmr/issues/new?template=bug_report.md) on GitHub.

### How do I request features?

Use the [feature request template](https://github.com/sachin/rmr/issues/new?template=feature_request.md) on GitHub.

## License

### Is RMR open source?

Yes, RMR is released under the MIT License. See [LICENSE](../LICENSE) for details.

### Can I use RMR in commercial projects?

Yes, the MIT License permits commercial use with attribution.

## Getting Help

### Where can I get help?

- Check the documentation in `docs/`
- Search [existing issues](https://github.com/sachin/rmr/issues)
- Open a new issue with the appropriate template
- Contact the maintainers

### How do I cite RMR?

If you use RMR in your research, please cite the original paper:

```bibtex
@article{li2026gremc,
  title={Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion},
  author={Li, Yuan and Hu, Jun and Jiang, Jiaxin and Hooi, Bryan and He, Bingsheng},
  journal={arXiv preprint arXiv:2605.00670},
  year={2026}
}
```
