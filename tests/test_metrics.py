import numpy as np

from rmr.evaluation.metrics import dcg_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k():
    scores = np.array([[0.1, 0.3, 0.5]], dtype=np.float32)
    labels = np.array([[0, 0, 1]], dtype=np.int32)
    r = recall_at_k(scores, labels, k=1)
    assert abs(r - 1.0) < 1e-6


def test_recall_at_k_multiple_users():
    scores = np.array([
        [0.1, 0.3, 0.5],
        [0.5, 0.1, 0.3],
    ], dtype=np.float32)
    labels = np.array([
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.int32)
    r = recall_at_k(scores, labels, k=1)
    expected = (1.0 + 1.0) / 2
    assert abs(r - expected) < 1e-6


def test_recall_at_k_zero_relevant():
    scores = np.array([
        [0.1, 0.3, 0.5],
        [0.5, 0.1, 0.3],
    ], dtype=np.float32)
    labels = np.array([
        [0, 0, 0],
        [1, 0, 0],
    ], dtype=np.int32)
    r = recall_at_k(scores, labels, k=2)
    assert abs(r - 1.0) < 1e-6


def test_dcg_at_k():
    relevances = np.array([3, 2, 1, 0])
    dcg = dcg_at_k(relevances, k=3)
    expected = 3.0 / np.log2(2) + 2.0 / np.log2(3) + 1.0 / np.log2(4)
    assert abs(dcg - expected) < 1e-6


def test_ndcg_at_k_exact():
    scores = np.array([[0.1, 0.3, 0.5, 0.2]], dtype=np.float32)
    labels = np.array([[0, 1, 1, 0]], dtype=np.int32)
    n = ndcg_at_k(scores, labels, k=3)
    top_k = [2, 1, 3]  # indices sorted by scores descending
    relevances = labels[0][top_k]  # [1, 1, 0]
    ideal = np.sort(labels[0])[::-1][:3]  # [1, 1, 0]
    dcg = dcg_at_k(relevances, 3)
    idcg = dcg_at_k(ideal, 3)
    expected = dcg / idcg
    assert abs(n - expected) < 1e-6


def test_ndcg_at_k_k_greater_than_n_items():
    scores = np.array([[0.1, 0.3, 0.5]], dtype=np.float32)
    labels = np.array([[0, 0, 1]], dtype=np.int32)
    n = ndcg_at_k(scores, labels, k=10)
    assert n > 0.0
