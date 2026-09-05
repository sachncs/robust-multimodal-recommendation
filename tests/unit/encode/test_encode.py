"""Tests for morel.encode."""

from __future__ import annotations

import torch

from morel.encode import Attention, Baseline, Identity, Layer, Mean, Token, Transformer


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
    base = Baseline("identity", dims={"v": 4}, pe_dim=2, hidden=8)
    feats = {"v": torch.randn(3, 4)}
    mask = torch.ones(3, 1)
    pe = torch.randn(3, 2)
    assert base(feats, mask, pe).shape == (3, 8)


def test_baseline_sum() -> None:
    base = Baseline("sum", dims={"v": 4, "t": 2}, pe_dim=2, hidden=8)
    feats = {"v": torch.randn(3, 4), "t": torch.randn(3, 2)}
    mask = torch.ones(3, 2)
    pe = torch.randn(3, 2)
    assert base(feats, mask, pe).shape == (3, 8)


def test_baseline_unknown_kind() -> None:
    import pytest

    with pytest.raises(ValueError):
        Baseline("nope", dims={"v": 4}, pe_dim=2, hidden=8)


def test_pool_attention_sums_to_one() -> None:
    pool = Attention(dim=8)
    h = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = pool(h, mask)
    assert out.shape == (2, 8)


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
