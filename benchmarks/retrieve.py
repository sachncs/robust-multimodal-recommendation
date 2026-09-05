"""Benchmarks for retrieval."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from morel.retrieve import acs, mage


def _make_graph(n: int) -> sp.csr_matrix:
    arr = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        arr[i, i + 1] = 1
        arr[i + 1, i] = 1
    return sp.csr_matrix(arr)


def bench_acs_1k(benchmark) -> None:
    g = _make_graph(1000)

    def run():
        for _ in range(10):
            acs.compute(g, [0, 999])

    benchmark(run)


def bench_mage_1k(benchmark) -> None:
    g = _make_graph(1000)
    features = {"v": np.eye(1000, dtype=np.float32)}
    mask = np.ones((1000, 1), dtype=np.float32)

    def run():
        for _ in range(5):
            mage.expand(g, [0, 999], 0, features, mask, iters=3)

    benchmark(run)
