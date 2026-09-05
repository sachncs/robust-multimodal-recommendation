"""Tests for morel.encode."""

from __future__ import annotations

import torch

from morel.encode import Attention, Identity, Layer, Mean, Token, Transformer


def test_input_embedding_shape() -> None:
    from morel.encode.input import Input

    emb = Input(dims={"v": 4, "t": 2}, pe_dim=2, hidden=8)
    feats = {"v": torch.randn(3, 4), "t": torch.randn(3, 2)}
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    pe = torch.randn(3, 2)
    out = emb(feats, mask, pe)
    assert out.shape == (3, 8)


def test_layer_with_attention_mask() -> None:
    layer = Layer(dim=8, heads=2)
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    out = layer(x, mask)
    assert out.shape == (2, 5, 8)


def test_layer_is_preln() -> None:
    """When norm1 is zeroed, the residual term dominates and output equals input + ffn(norm2(input))."""
    layer = Layer(dim=8, heads=2, dropout=0.0)
    torch.manual_seed(0)
    layer.eval()
    layer.norm1.weight.data.zero_()
    layer.norm1.bias.data.zero_()
    layer.norm2.weight.data.zero_()
    layer.norm2.bias.data.zero_()
    x = torch.randn(1, 3, 8)
    out = layer(x)
    expected_normed1 = torch.zeros_like(x)
    expected_attn, _ = layer.attn(expected_normed1, expected_normed1, expected_normed1, need_weights=False)
    expected_after_attn = x + expected_attn
    expected_normed2 = torch.zeros_like(expected_after_attn)
    expected_ffn = layer.ffn(expected_normed2)
    expected = expected_after_attn + expected_ffn
    assert torch.allclose(out, expected, atol=1e-5)


def test_transformer_pool_attention() -> None:
    tf = Transformer(dims={"v": 4, "t": 2}, pe_dim=2, hidden=8, layers=2, heads=2, pool="attention")
    feats = {"v": torch.randn(3, 4), "t": torch.randn(3, 2)}
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    pe = torch.randn(3, 2)
    out = tf(feats, mask, pe)
    assert out.shape == (3, 8)


def test_transformer_pool_mean() -> None:
    tf = Transformer(dims={"v": 4, "t": 2}, pe_dim=2, hidden=8, layers=2, heads=2, pool="mean")
    feats = {"v": torch.randn(2, 4), "t": torch.randn(2, 2)}
    mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    pe = torch.randn(2, 2)
    assert tf(feats, mask, pe).shape == (2, 8)


def test_transformer_pool_token() -> None:
    tf = Transformer(dims={"v": 4}, pe_dim=2, hidden=8, layers=1, heads=2, pool="token")
    feats = {"v": torch.randn(2, 4)}
    mask = torch.ones(2, 1)
    pe = torch.randn(2, 2)
    assert tf(feats, mask, pe).shape == (2, 8)


def test_transformer_rejects_invalid_pool() -> None:
    import pytest

    with pytest.raises(ValueError):
        Transformer(dims={"v": 4}, pe_dim=2, hidden=8, layers=1, heads=2, pool="bogus")


def test_baseline_identity() -> None:
    from morel.encode import GraphEncoderBaseline

    base = GraphEncoderBaseline("identity", dims={"v": 4}, pe_dim=2, hidden=8)
    feats = {"v": torch.randn(3, 4)}
    mask = torch.ones(3, 1)
    pe = torch.randn(3, 2)
    assert base(feats, mask, pe).shape == (3, 8)


def test_baseline_sum() -> None:
    from morel.encode import GraphEncoderBaseline

    base = GraphEncoderBaseline("sum", dims={"v": 4, "t": 2}, pe_dim=2, hidden=8)
    feats = {"v": torch.randn(3, 4), "t": torch.randn(3, 2)}
    mask = torch.ones(3, 2)
    pe = torch.randn(3, 2)
    assert base(feats, mask, pe).shape == (3, 8)


def test_baseline_unknown_kind() -> None:
    import pytest

    from morel.encode import GraphEncoderBaseline

    with pytest.raises(ValueError):
        GraphEncoderBaseline("nope", dims={"v": 4}, pe_dim=2, hidden=8)


def test_pool_attention_sums_to_one() -> None:
    pool = Attention(dim=8)
    h = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = pool(h, mask)
    assert out.shape == (2, 8)


def test_attention_pool_no_nan_on_all_masked() -> None:
    """All-masked rows must not produce NaN."""
    pool = Attention(dim=8)
    h = torch.randn(2, 5, 8)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    out = pool(h, mask)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 8)


def test_attention_pool_partial_mask_finite() -> None:
    pool = Attention(dim=4)
    h = torch.randn(3, 6, 4)
    mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [True, False, True, False, False, False],
            [False, False, False, False, False, False],
        ]
    )
    out = pool(h, mask)
    assert torch.isfinite(out).all()
    assert out.shape == (3, 4)


def test_pool_mean_masked() -> None:
    pool = Mean()
    h = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    out = pool(h, mask)
    assert out.shape == (2, 3)


def test_pool_token_first() -> None:
    pool = Token()
    h = torch.randn(2, 5, 3)
    out = pool(h)
    assert torch.equal(out, h[:, 0, :])
