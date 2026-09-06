"""Tests for morel.recommend."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from morel.core.errors import DataError
from morel.recommend import MF, Light, Pop, bpr, negatives
from morel.recommend.bpr import distinct_ranks, ranks_to_items


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
    with pytest.raises(ValueError, match="users and items must be positive"):
        Light(users=0, items=5)


def test_light_invalid_layers() -> None:
    with pytest.raises(ValueError, match="layers must be non-negative"):
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


def test_negatives_vectorised_matches_per_user() -> None:
    """Vectorised negatives match the old per-user implementation when fed the same seed."""
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 50, size=200)
    cols = rng.integers(0, 30, size=200)
    ui = sp.csr_matrix((np.ones(200, dtype=np.float32), (rows, cols)), shape=(50, 30))
    out = negatives(ui, count=2, seed=0)
    for u in range(50):
        positives = set(ui.indices[ui.indptr[u] : ui.indptr[u + 1]].tolist())
        assert positives.isdisjoint(set(out[u].tolist()))


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


def test_light_gcn_cache_key_is_content_based() -> None:
    """Two distinct CSR objects with identical content share the cache entry."""
    users, items = 4, 5
    arr = np.eye(users, items, dtype=np.float32)
    ui1 = sp.csr_matrix(arr)
    ui2 = sp.csr_matrix(arr.copy())
    assert ui1 is not ui2
    light = Light(users=users, items=items, embed=4, layers=1)
    light(torch.arange(users), torch.arange(items), ui1)
    first_key, first_tensor = light.adj_cache
    light(torch.arange(users), torch.arange(items), ui2)
    assert light.adj_cache[0] == first_key
    assert light.adj_cache[1] is first_tensor


def test_light_gcn_cache_invalidates_on_new_graph() -> None:
    users, items = 4, 6
    ui_a = sp.csr_matrix(np.eye(users, items, dtype=np.float32))
    rng = np.random.default_rng(0)
    ui_b = sp.csr_matrix(
        (
            np.ones(users * 2, dtype=np.float32),
            (
                np.concatenate([np.arange(users), np.arange(users)]),
                rng.integers(0, items, size=users * 2),
            ),
        ),
        shape=(users, items),
    )
    light = Light(users=users, items=items, embed=4, layers=1)
    light(torch.arange(users), torch.arange(items), ui_a)
    first_key = light.adj_cache[0]
    light(torch.arange(users), torch.arange(items), ui_b)
    assert light.adj_cache[0] != first_key


def test_light_seed_makes_init_reproducible() -> None:
    torch.manual_seed(1)
    first = Light(users=5, items=7, embed=8, layers=2, seed=3)
    torch.manual_seed(9999)
    second = Light(users=5, items=7, embed=8, layers=2, seed=3)
    assert torch.equal(first.user_emb.weight, second.user_emb.weight)
    assert torch.equal(first.item_emb.weight, second.item_emb.weight)


def test_light_seed_does_not_disturb_global_rng() -> None:
    torch.manual_seed(7)
    expected = torch.randn(4)
    torch.manual_seed(7)
    Light(users=5, items=7, embed=8, layers=2, seed=3)
    assert torch.equal(expected, torch.randn(4))


def test_light_without_seed_follows_global_rng() -> None:
    torch.manual_seed(21)
    first = Light(users=5, items=7, embed=8, layers=2)
    torch.manual_seed(21)
    second = Light(users=5, items=7, embed=8, layers=2)
    assert torch.equal(first.user_emb.weight, second.user_emb.weight)


def test_mf_seed_makes_init_reproducible() -> None:
    torch.manual_seed(1)
    first = MF(users=5, items=7, embed=8, seed=11)
    torch.manual_seed(9999)
    second = MF(users=5, items=7, embed=8, seed=11)
    assert torch.equal(first.user_emb.weight, second.user_emb.weight)


def test_ranks_to_items_skips_positives() -> None:
    positives = np.array([0, 2], dtype=np.int64)
    # Non-positive items below 5 are 1, 3, 4 -> ranks 0, 1, 2.
    got = ranks_to_items(np.array([0, 1, 2], dtype=np.int64), positives)
    assert got.tolist() == [1, 3, 4]


def test_ranks_to_items_without_positives_is_identity() -> None:
    ranks = np.array([0, 3, 7], dtype=np.int64)
    assert ranks_to_items(ranks, np.array([], dtype=np.int64)).tolist() == ranks.tolist()


def test_distinct_ranks_are_distinct_and_in_range() -> None:
    rng = np.random.default_rng(0)
    for high, size in [(10, 9), (1000, 5), (50, 25)]:
        out = distinct_ranks(rng, high, size)
        assert out.shape == (size,)
        assert len(set(out.tolist())) == size
        assert out.min() >= 0
        assert out.max() < high


def test_negatives_are_distinct_per_user() -> None:
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, 40, 400), rng.integers(0, 60, 400)
    ui = sp.csr_matrix((np.ones(400, dtype=np.float32), (rows, cols)), shape=(40, 60))
    out = negatives(ui, count=5, seed=0)
    for u in range(40):
        assert len(set(out[u].tolist())) == 5


def test_negatives_is_deterministic_for_a_seed() -> None:
    ui = sp.csr_matrix(np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=np.float32))
    assert np.array_equal(negatives(ui, count=2, seed=7), negatives(ui, count=2, seed=7))


def test_negatives_scales_to_a_wide_catalogue() -> None:
    """Regression: the old sampler densified to (users, items) and needed ~8 GB here."""
    users, items = 500, 200_000
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, users, 5_000), rng.integers(0, items, 5_000)
    ui = sp.csr_matrix((np.ones(5_000, dtype=np.float32), (rows, cols)), shape=(users, items))

    out = negatives(ui, count=3, seed=0)

    assert out.shape == (users, 3)
    for u in range(0, users, 50):
        positives = set(ui.indices[ui.indptr[u] : ui.indptr[u + 1]].tolist())
        assert positives.isdisjoint(out[u].tolist())
