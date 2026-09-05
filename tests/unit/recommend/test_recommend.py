"""Tests for morel.recommend."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from morel.core.errors import DataError
from morel.recommend import Light, MF, Pop, bpr, negatives


def test_light_l0_equals_dot() -> None:
    users, items = 3, 4
    light = Light(users=users, items=items, embed=8, layers=0)
    # Need to call forward with ui_graph to populate cached adjacency
    ui = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.float32))
    light(torch.arange(users), torch.arange(items), ui)
    out = light(torch.arange(users), torch.arange(items))
    expected = light.user_emb.weight @ light.item_emb.weight.t()
    assert torch.allclose(out, expected)


def test_light_invalid_dims() -> None:
    with pytest.raises(ValueError):
        Light(users=0, items=5)


def test_light_invalid_layers() -> None:
    with pytest.raises(ValueError):
        Light(users=5, items=5, layers=-1)


def test_light_out_of_range_user() -> None:
    light = Light(users=3, items=4)
    ui = sp.csr_matrix(np.eye(3, 4, dtype=np.float32))
    light(torch.arange(3), torch.arange(4), ui)
    with pytest.raises(IndexError):
        light(torch.tensor([0, 99]), torch.arange(4))


def test_bpr_loss_positive() -> None:
    loss = bpr(torch.tensor([0.5]), torch.tensor([0.1]))
    assert float(loss) > 0


def test_negatives_strict() -> None:
    ui = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.float32))
    out = negatives(ui, count=1, seed=0)
    assert out.shape == (2, 1)
    for u in range(2):
        positives = set(ui.indices[ui.indptr[u] : ui.indptr[u + 1]].tolist())
        assert int(out[u, 0]) not in positives


def test_negatives_invalid_count() -> None:
    ui = sp.csr_matrix(np.eye(3, dtype=np.float32))
    with pytest.raises(DataError):
        negatives(ui, count=0, seed=0)


def test_negatives_insufficient_items() -> None:
    ui = sp.csr_matrix(np.array([[1, 1], [1, 1]], dtype=np.float32))
    with pytest.raises(DataError):
        negatives(ui, count=1, seed=0)


def test_mf_forward() -> None:
    mf = MF(users=3, items=4, embed=8)
    out = mf(torch.arange(3), torch.arange(4))
    assert out.shape == (3, 4)


def test_pop_forward() -> None:
    pop = Pop(users=3, items=4)
    ui = sp.csr_matrix(np.eye(3, 4, dtype=np.float32))
    pop(torch.arange(3), torch.arange(4), ui)
    out = pop(torch.arange(3), torch.arange(4))
    assert out.shape == (3, 4)
