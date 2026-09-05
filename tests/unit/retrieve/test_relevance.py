"""Tests for morel.retrieve.relevance."""

from __future__ import annotations

import numpy as np

from morel.retrieve.relevance import mean_relevance, relevance


def test_relevance_self_one() -> None:
    f = {"v": np.eye(3, dtype=np.float32)}
    m = np.ones((3, 1), dtype=np.float32)
    assert relevance(0, 0, f, m) == 1.0


def test_relevance_cosine_orthogonal_zero() -> None:
    f = {"v": np.eye(2, dtype=np.float32)}
    m = np.ones((2, 1), dtype=np.float32)
    assert relevance(0, 1, f, m) == 0.0


def test_relevance_missing_modality_zero() -> None:
    f = {"v": np.eye(2, dtype=np.float32)}
    m = np.zeros((2, 1), dtype=np.float32)
    assert relevance(0, 1, f, m) == 0.0


def test_mean_relevance_excludes_self() -> None:
    f = {"v": np.eye(2, dtype=np.float32)}
    m = np.ones((2, 1), dtype=np.float32)
    val = mean_relevance(0, np.array([0, 1]), f, m)
    assert val == 0.0  # orthogonal to self (excluded)


def test_mean_relevance_empty_returns_zero() -> None:
    f = {"v": np.eye(2, dtype=np.float32)}
    m = np.ones((2, 1), dtype=np.float32)
    assert mean_relevance(0, np.array([], dtype=np.int64), f, m) == 0.0
