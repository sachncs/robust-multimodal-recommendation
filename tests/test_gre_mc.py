import numpy as np
import scipy.sparse as sp
import torch

from rmr.models.gre_mc import GREMC


def test_gre_mc_forward():
    # Use 5-node graph so Laplacian PE (k=4) has enough eigenvectors
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.float32))
    model = GREMC(
        input_dims={"visual": 10, "text": 8},
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        codebook_size=5,
        top_p=2,
        tau=0.5,
        pe_dim=4,
    )
    features = {
        "visual": torch.randn(5, 10),
        "text": torch.randn(5, 8),
    }
    mask = torch.ones(5, 2)
    out, g = model(features, mask, adj)
    assert out["visual"].shape == (5, 10)
    assert out["text"].shape == (5, 8)


def test_gre_mc_training_false():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.float32))
    model = GREMC(
        input_dims={"visual": 10, "text": 8},
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        codebook_size=5,
        top_p=2,
        tau=0.5,
        pe_dim=4,
        dropout=0.0,
    )
    model.eval()
    features = {
        "visual": torch.randn(5, 10),
        "text": torch.randn(5, 8),
    }
    mask = torch.ones(5, 2)
    out1, g1 = model(features, mask, adj, training=False)
    out2, g2 = model(features, mask, adj, training=False)
    # Without Gumbel noise and with dropout disabled, outputs should be deterministic
    for key in out1:
        torch.testing.assert_close(out1[key], out2[key], atol=1e-6, rtol=0)
    torch.testing.assert_close(g1, g2, atol=1e-6, rtol=0)


def test_gre_mc_index_parameter():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.float32))
    model = GREMC(
        input_dims={"visual": 10, "text": 8},
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        codebook_size=5,
        top_p=2,
        tau=0.5,
        pe_dim=4,
    )
    features = {
        "visual": torch.randn(3, 10),
        "text": torch.randn(3, 8),
    }
    mask = torch.ones(3, 2)
    index = torch.tensor([0, 2, 4])
    out, g = model(features, mask, adj, index=index, training=True)
    assert out["visual"].shape == (3, 10)
    assert out["text"].shape == (3, 8)
    assert g.shape == (3, 5)


def test_gre_mc_with_retrieval_buffers():
    """GREMC should retrieve subgraphs and encode them when buffers are set."""
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=np.float32))

    # Full features for 5 items
    all_features = {
        "visual": np.random.randn(5, 10).astype(np.float32),
        "text": np.random.randn(5, 8).astype(np.float32),
    }
    all_mask = np.ones((5, 2), dtype=np.float32)

    model = GREMC(
        input_dims={"visual": 10, "text": 8},
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        codebook_size=5,
        top_p=2,
        tau=0.5,
        pe_dim=4,
        dropout=0.0,
        top_k_anchors=2,
        mage_iters=2,
    )
    model.register_retrieval_buffers(all_features, all_mask)
    model.eval()

    # Batch of 2 query items
    features = {
        "visual": torch.from_numpy(all_features["visual"][[0, 4]]),
        "text": torch.from_numpy(all_features["text"][[0, 4]]),
    }
    mask = torch.from_numpy(all_mask[[0, 4]])
    index = torch.tensor([0, 4])
    out, g = model(features, mask, adj, index=index, training=False)
    assert out["visual"].shape == (2, 10)
    assert out["text"].shape == (2, 8)
    assert g.shape == (2, 5)
    # Deterministic without Gumbel noise and dropout disabled
    out2, g2 = model(features, mask, adj, index=index, training=False)
    torch.testing.assert_close(out["visual"], out2["visual"], atol=1e-5, rtol=0)
    torch.testing.assert_close(g, g2, atol=1e-5, rtol=0)
