import numpy as np

from rmr.evaluation.evaluator import Evaluator


def test_evaluator_compute():
    scores = np.array([
        [0.1, 0.3, 0.5],
        [0.5, 0.1, 0.3],
    ], dtype=np.float32)
    labels = np.array([
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.int32)
    ev = Evaluator()
    res = ev.evaluate(scores, labels)
    assert "recall@10" in res
    assert "ndcg@10" in res


def test_evaluator_custom_ks():
    scores = np.array([
        [0.1, 0.3, 0.5],
        [0.5, 0.1, 0.3],
    ], dtype=np.float32)
    labels = np.array([
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.int32)
    ev = Evaluator(ks=[1, 2, 5])
    res = ev.evaluate(scores, labels)
    assert "recall@1" in res
    assert "ndcg@1" in res
    assert "recall@2" in res
    assert "ndcg@2" in res
    assert "recall@5" in res
    assert "ndcg@5" in res
    assert "recall@10" not in res
    assert "ndcg@10" not in res


def test_evaluator_exact_values():
    scores = np.array([
        [0.1, 0.3, 0.5],
        [0.5, 0.1, 0.3],
    ], dtype=np.float32)
    labels = np.array([
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.int32)
    ev = Evaluator(ks=[1])
    res = ev.evaluate(scores, labels)
    assert abs(res["recall@1"] - 1.0) < 1e-6
    # For NDCG@1: each user has ideal DCG of 1.0 / log2(2) = 1.0,
    # and each gets the top-1 item correct, so DCG@1 = 1.0 -> NDCG = 1.0.
    assert abs(res["ndcg@1"] - 1.0) < 1e-6
