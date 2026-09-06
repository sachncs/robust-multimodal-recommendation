"""BPR data helpers for the recommendation stage."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.app.data import BPRDataset, build_recommendation_loader
from morel.core.errors import DataError


def interactions(users: int = 20, items: int = 50, count: int = 200) -> sp.csr_matrix:
    """Return a random sparse interaction matrix."""
    rng = np.random.default_rng(0)
    return sp.csr_matrix(
        (
            np.ones(count, dtype=np.float32),
            (rng.integers(0, users, count), rng.integers(0, items, count)),
        ),
        shape=(users, items),
    )


def positives_of(ui: sp.csr_matrix, user: int) -> set[int]:
    """Return the item ids ``user`` interacted with."""
    return set(ui.indices[ui.indptr[user] : ui.indptr[user + 1]].tolist())


def test_triples_are_strict() -> None:
    """The positive must be a real interaction and the negative must not be."""
    ui = interactions()
    dataset = BPRDataset(ui, length=400, seed=0)
    for i in range(len(dataset)):
        sample = dataset[i]
        pos = positives_of(ui, sample["users"])
        assert sample["positive"] in pos
        assert sample["negative"] not in pos


def test_sampling_is_deterministic() -> None:
    ui = interactions()
    first = BPRDataset(ui, length=50, seed=0)
    second = BPRDataset(ui, length=50, seed=0)
    assert [first[i] for i in range(50)] == [second[i] for i in range(50)]


def test_a_different_seed_changes_the_samples() -> None:
    ui = interactions()
    first = BPRDataset(ui, length=50, seed=0)
    second = BPRDataset(ui, length=50, seed=1)
    assert [first[i] for i in range(50)] != [second[i] for i in range(50)]


def test_users_without_interactions_are_skipped() -> None:
    arr = np.zeros((4, 5), dtype=np.float32)
    arr[1, 2] = arr[3, 0] = 1.0
    dataset = BPRDataset(sp.csr_matrix(arr), length=40, seed=0)
    assert dataset.users == [1, 3]
    assert {dataset[i]["users"] for i in range(40)} <= {1, 3}


def test_graph_with_no_interactions_is_rejected() -> None:
    with pytest.raises(DataError, match="no user has any interaction"):
        BPRDataset(sp.csr_matrix((3, 4), dtype=np.float32), length=5)


def test_user_interacting_with_everything_is_rejected() -> None:
    with pytest.raises(DataError, match="no negative can be sampled"):
        BPRDataset(sp.csr_matrix(np.ones((2, 3), dtype=np.float32)), length=5)


def test_loader_yields_the_keys_the_trainer_reads() -> None:
    loader = build_recommendation_loader(interactions(), batch_size=32)
    batch = next(iter(loader))
    assert set(batch) == {"users", "positive", "negative"}
    for value in batch.values():
        assert value.shape == (32,)


def test_epoch_length_defaults_to_the_interaction_count() -> None:
    ui = interactions()
    loader = build_recommendation_loader(ui)
    assert len(loader.dataset) == ui.nnz
