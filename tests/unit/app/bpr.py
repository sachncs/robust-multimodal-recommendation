"""BPR data helpers for the recommendation stage."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from morel.app.data import (
    BPR,
    recommend_loader,
    recommend_loaders,
    split,
)
from morel.core.errors import Datum


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


def positives(ui: sp.csr_matrix, user: int) -> set[int]:
    """Return the item ids ``user`` interacted with."""
    return set(ui.indices[ui.indptr[user] : ui.indptr[user + 1]].tolist())


class Checker:
    """Aggregated test methods for this module."""

    def triples(self) -> None:
        """The positive must be a real interaction and the negative must not be."""
        ui = interactions()
        dataset = BPR(ui, length=400, seed=0)
        for i in range(len(dataset)):
            sample = dataset[i]
            pos = positives(ui, sample["users"])
            assert sample["positive"] in pos
            assert sample["negative"] not in pos

    def sampling(self) -> None:
        ui = interactions()
        first = BPR(ui, length=50, seed=0)
        second = BPR(ui, length=50, seed=0)
        assert [first[i] for i in range(50)] == [second[i] for i in range(50)]

    def a(self) -> None:
        ui = interactions()
        first = BPR(ui, length=50, seed=0)
        second = BPR(ui, length=50, seed=1)
        assert [first[i] for i in range(50)] != [second[i] for i in range(50)]

    def users(self) -> None:
        arr = np.zeros((4, 5), dtype=np.float32)
        arr[1, 2] = arr[3, 0] = 1.0
        dataset = BPR(sp.csr_matrix(arr), length=40, seed=0)
        assert dataset.users == [1, 3]
        assert {dataset[i]["users"] for i in range(40)} <= {1, 3}

    def graph(self) -> None:
        with pytest.raises(Datum, match="no user has any interaction"):
            BPR(sp.csr_matrix((3, 4), dtype=np.float32), length=5)

    def user(self) -> None:
        with pytest.raises(Datum, match="no negative can be sampled"):
            BPR(sp.csr_matrix(np.ones((2, 3), dtype=np.float32)), length=5)

    def loader(self) -> None:
        loader = recommend_loader(interactions(), batch_size=32)
        batch = next(iter(loader))
        assert set(batch) == {"users", "positive", "negative"}
        for value in batch.values():
            assert value.shape == (32,)

    def epoch(self) -> None:
        ui = interactions()
        loader = recommend_loader(ui)
        assert len(loader.dataset) == ui.nnz

    def split(self) -> None:
        dataset = BPR(interactions(), length=100, seed=0)
        train, val = split(dataset, val_fraction=0.2, seed=0)
        assert val is not None
        assert len(val) == 20
        assert len(train) == 80

    def disjoint(self) -> None:
        dataset = BPR(interactions(), length=100, seed=0)
        train, val = split(dataset, val_fraction=0.25, seed=0)
        assert val is not None
        train_idx, val_idx = set(train.indices), set(val.indices)
        assert not (train_idx & val_idx), "a sample must not be in both splits"
        assert train_idx | val_idx == set(range(100))

    def reproducible(self) -> None:
        dataset = BPR(interactions(), length=100, seed=0)
        first, _ = split(dataset, val_fraction=0.3, seed=5)
        second, _ = split(dataset, val_fraction=0.3, seed=5)
        assert first.indices == second.indices

    def zero(self) -> None:
        dataset = BPR(interactions(), length=100, seed=0)
        train, val = split(dataset, val_fraction=0.0, seed=0)
        assert val is None
        assert train is dataset

    def fraction(self) -> None:
        dataset = BPR(interactions(), length=4, seed=0)
        _, val = split(dataset, val_fraction=0.01, seed=0)
        assert val is None, "rather than build an empty validation loader"

    def invalid(self) -> None:
        dataset = BPR(interactions(), length=10, seed=0)
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(Datum, match=r"validation fraction must be in \[0, 1\)"):
                split(dataset, val_fraction=bad, seed=0)

    def recommendation(self) -> None:
        ui = interactions()
        train, val = recommend_loaders(ui, batch_size=8, val_fraction=0.25, seed=0)
        assert val is not None
        assert len(train.dataset) + len(val.dataset) == ui.nnz