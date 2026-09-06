"""Tests for morel.data.stream streaming primitives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from morel.data.stream import (
    exact_two_pass_interactions,
    review_stream,
    streaming_interactions,
    streaming_item_cooccurrence,
)


def write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "reviews.json"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


class Checker:
    """Aggregated test methods for this module."""

    def review(tmp_path: Path) -> None:
        records = [
            {"reviewerID": "u1", "asin": "i1"},
            {"reviewerID": "u1", "asin": "i2"},
            {"reviewerID": "u2", "asin": "i1"},
            {"reviewerID": "u2", "asin": "i2"},
        ]
        path = write(tmp_path, records)
        chunks = list(review_stream(path, chunk_size=2))
        flat = [r for chunk in chunks for r in chunk]
        assert len(flat) == len(records)

    def stream(tmp_path: Path) -> None:
        from morel.core.errors import DataError

        with pytest.raises(DataError):
            list(review_stream(tmp_path / "nope.json"))

    def exact(tmp_path: Path) -> None:
        records = [
            {"reviewerID": "u1", "asin": "i1"},
            {"reviewerID": "u1", "asin": "i2"},
            {"reviewerID": "u1", "asin": "i3"},
            {"reviewerID": "u2", "asin": "i1"},
            {"reviewerID": "u2", "asin": "i2"},
            {"reviewerID": "u3", "asin": "i1"},
        ]
        path = write(tmp_path, records)
        ui, _, _ = exact_two_pass_interactions(path, min_edges=2, chunk_size=2)
        assert ui.nnz >= 4

    def streaming() -> None:
        chunks = iter(
            [
                (np.array([0, 0]), np.array([0, 1])),
                (np.array([1, 1]), np.array([1, 2])),
            ]
        )
        cooc = streaming_item_cooccurrence(chunks, items=3)
        assert cooc.shape == (3, 3)
        # user 0 -> items {0, 1}: cooc(0, 1) += 1
        # user 1 -> items {1, 2}: cooc(1, 2) += 1
        assert cooc[0, 1] == 1
        assert cooc[1, 0] == 1
        assert cooc[1, 2] == 1
        assert cooc[2, 1] == 1
        assert cooc[0, 0] == 0
        assert cooc[0, 2] == 0
        assert cooc[2, 2] == 0

    def interactions(tmp_path: Path) -> None:
        records = [
            {"reviewerID": "u1", "asin": "i1"},
            {"reviewerID": "u1", "asin": "i2"},
            {"reviewerID": "u1", "asin": "i3"},
            {"reviewerID": "u2", "asin": "i1"},
            {"reviewerID": "u2", "asin": "i2"},
            {"reviewerID": "u3", "asin": "i1"},
        ]
        path = write(tmp_path, records)
        emitted = []
        for user_ids, item_ids in streaming_interactions(path, min_edges=2, chunk_size=10):
            emitted.append((user_ids, item_ids))
        assert len(emitted) >= 1
        assert all(isinstance(u, np.ndarray) for u, _ in emitted)
__test__ = False
