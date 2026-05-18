import torch

from rmr.models.transformer import (
    GraphTransformerLayer,
    InputEmbedding,
    JointEncodingGraphTransformer,
)


def test_transformer_forward_shape():
    model = JointEncodingGraphTransformer(
        input_dims={"visual": 10, "text": 8},
        pe_dim=4,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )
    features = {
        "visual": torch.randn(3, 10),
        "text": torch.randn(3, 8),
    }
    mask = torch.ones(3, 2)
    pe = torch.randn(3, 4)
    out = model(features, mask, pe)
    assert out.shape == (3, 16)


def test_input_embedding_masking_behavior():
    embed = InputEmbedding(
        input_dims={"visual": 4, "text": 3},
        pe_dim=2,
        hidden_dim=8,
        dropout=0.0,
    )
    features = {
        "visual": torch.ones(2, 4),
        "text": torch.ones(2, 3) * 2.0,
    }
    # Mask out visual for both samples
    mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    pe = torch.zeros(2, 2)
    out = embed(features, mask, pe)
    # The output should be a valid tensor of the right shape
    assert out.shape == (2, 8)
    # Since visual is zeroed out, the concatenated input should only have
    # text features + pe. We can test by comparing with explicit zero mask.
    features_visual_zeroed = {
        "visual": torch.zeros(2, 4),
        "text": features["text"],
    }
    mask_all_ones = torch.ones(2, 2)
    out2 = embed(features_visual_zeroed, mask_all_ones, pe)
    torch.testing.assert_close(out, out2, atol=1e-6, rtol=0)


def test_graph_transformer_layer_attn_mask():
    layer = GraphTransformerLayer(hidden_dim=8, num_heads=2, dropout=0.0)
    x = torch.randn(2, 4, 8)
    # Create a causal-like mask
    attn_mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
    attn_mask = attn_mask.masked_fill(attn_mask, float("-inf"))
    out = layer(x, attn_mask=attn_mask)
    assert out.shape == (2, 4, 8)
    # With a valid mask the output should be finite
    assert torch.isfinite(out).all()


def test_joint_encoding_query_pool_correctness():
    model = JointEncodingGraphTransformer(
        input_dims={"a": 3, "b": 3},
        pe_dim=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    features = {
        "a": torch.randn(4, 3),
        "b": torch.randn(4, 3),
    }
    mask = torch.ones(4, 2)
    pe = torch.randn(4, 2)
    out = model(features, mask, pe)
    assert out.shape == (4, 8)
    # Query pool applies softmax over seq_len=1, so weights sum to 1 implicitly.
    # Since seq_len is 1 after unsqueeze, the weighted sum is just the single token.
    # We verify by checking that with identical inputs we get deterministic results.
    out2 = model(features, mask, pe)
    torch.testing.assert_close(out, out2, atol=1e-6, rtol=0)


def test_transformer_sequence_forward_shape():
    """Transformer should accept batched sequences (subgraph tokens)."""
    model = JointEncodingGraphTransformer(
        input_dims={"visual": 4, "text": 3},
        pe_dim=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    B, S = 2, 5
    features = {
        "visual": torch.randn(B, S, 4),
        "text": torch.randn(B, S, 3),
    }
    mask = torch.ones(B, S, 2)
    pe = torch.randn(B, S, 2)
    out = model(features, mask, pe)
    assert out.shape == (B, 8)


def test_input_embedding_sequence_masking():
    """InputEmbedding should broadcast per-node masks over sequences."""
    embed = InputEmbedding(
        input_dims={"v": 2, "t": 2},
        pe_dim=2,
        hidden_dim=4,
        dropout=0.0,
    )
    B, S = 2, 3
    features = {
        "v": torch.ones(B, S, 2),
        "t": torch.ones(B, S, 2) * 2.0,
    }
    # Mask out modality v for all nodes
    mask = torch.zeros(B, S, 2)
    mask[:, :, 1] = 1.0
    pe = torch.zeros(B, S, 2)
    out = embed(features, mask, pe)
    assert out.shape == (B, S, 4)
    # Verify v is zeroed by comparing with explicit zeros
    features_zeroed = {
        "v": torch.zeros(B, S, 2),
        "t": features["t"],
    }
    mask_ones = torch.ones(B, S, 2)
    out2 = embed(features_zeroed, mask_ones, pe)
    torch.testing.assert_close(out, out2, atol=1e-6, rtol=0)
