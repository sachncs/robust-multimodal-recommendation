"""Performance benchmarks for RMR components."""

import numpy as np
import scipy.sparse as sp
import torch

from rmr.models.codebook import SparseRoutingCodebook
from rmr.models.decoder import ModalityDecoder
from rmr.models.gre_mc import GREMC
from rmr.models.positional_encoding import LaplacianPE
from rmr.models.retrieval import acs, anchor_retrieval, mage
from rmr.models.transformer import JointEncodingGraphTransformer


def create_random_graph(
    n_nodes: int = 100, density: float = 0.1
) -> sp.csr_matrix:
    """Create a random sparse graph."""
    rows = []
    cols = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < density:
                rows.extend([i, j])
                cols.extend([j, i])
    if not rows:
        rows = [0, 1]
        cols = [1, 0]
    data = np.ones(len(rows))
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return adj


def create_random_features(
    n_nodes: int = 100, n_features: int = 128
) -> np.ndarray:
    """Create random features."""
    return np.random.randn(n_nodes, n_features).astype(np.float32)


def test_anchor_retrieval_baseline(benchmark):
    """Benchmark anchor retrieval."""
    n_nodes = 200
    features = create_random_features(n_nodes, 128)
    mask = np.zeros(n_nodes, dtype=bool)
    mask[10:20] = True

    def run():
        return anchor_retrieval(features, mask, k=10)

    result = benchmark(run)
    assert result.shape == (n_nodes, 10)


def test_acs_baseline(benchmark):
    """Benchmark Anchor Connecting Subgraph."""
    n_nodes = 100
    adj = create_random_graph(n_nodes, 0.15)
    anchors = [0, 5, 10, 15, 20]

    def run():
        return acs(adj, anchors, root=0)

    result = benchmark(run)
    assert len(result) > 0


def test_mage_baseline(benchmark):
    """Benchmark Modality-Aware Graph Expansion."""
    n_nodes = 100
    adj = create_random_graph(n_nodes, 0.15)
    features = create_random_features(n_nodes, 128)
    anchors = [0, 5, 10]
    query_idx = 0
    modality_mask = np.ones(n_nodes, dtype=bool)

    def run():
        return mage(adj, features, anchors, query_idx, modality_mask, t=3)

    result = benchmark(run)
    assert len(result) >= len(anchors)


def test_laplacian_pe_baseline(benchmark):
    """Benchmark Laplacian Positional Encoding."""
    n_nodes = 100
    adj = create_random_graph(n_nodes, 0.15)
    pe_dim = 20
    pe = LaplacianPE(pe_dim)

    def run():
        return pe(adj)

    result = benchmark(run)
    assert result.shape == (n_nodes, pe_dim)


def test_transformer_baseline(benchmark):
    """Benchmark Graph Transformer."""
    hidden_dim = 128
    num_heads = 4
    num_layers = 2
    transformer = JointEncodingGraphTransformer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    x = torch.randn(10, hidden_dim)
    pe = torch.randn(10, 20)

    def run():
        return transformer(x, pe)

    result = benchmark(run)
    assert result.shape == (10, hidden_dim)


def test_codebook_baseline(benchmark):
    """Benchmark Sparse-Routing Codebook."""
    hidden_dim = 128
    codebook_size = 100
    top_p = 4
    codebook = SparseRoutingCodebook(
        hidden_dim=hidden_dim,
        codebook_size=codebook_size,
        top_p=top_p,
    )
    x = torch.randn(32, hidden_dim)

    def run():
        return codebook(x, training=True)

    result = benchmark(run)
    assert result[0].shape == (32, hidden_dim)


def test_decoder_baseline(benchmark):
    """Benchmark Modality Decoder."""
    hidden_dim = 128
    output_dim = 256
    decoder = ModalityDecoder(hidden_dim, output_dim)
    x = torch.randn(32, hidden_dim)

    def run():
        return decoder(x)

    result = benchmark(run)
    assert result.shape == (32, output_dim)


def test_gre_mc_forward_baseline(benchmark):
    """Benchmark GRE-MC forward pass."""
    input_dims = {"visual": 4096, "text": 384}
    hidden_dim = 128
    model = GREMC(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        codebook_size=50,
        top_p=4,
        tau=0.5,
        pe_dim=20,
    )

    n_items = 50
    features = {
        "visual": torch.randn(n_items, 4096),
        "text": torch.randn(n_items, 384),
    }
    mask = torch.zeros(n_items, dtype=torch.bool)
    adj = sp.eye(n_items, format="csr")

    def run():
        return model(features, mask, adj, training=False)

    result = benchmark(run)
    assert "visual" in result
    assert "text" in result
