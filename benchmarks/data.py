"""Benchmarks for data lifecycle."""

from __future__ import annotations

import numpy as np

from morel.data.build import bipartite, item_cooccurrence
from morel.data.mask import bernoulli


def bench_kcore_10k(benchmark) -> None:
    rng = np.random.default_rng(0)
    users = 2000
    items = 10000
    uids = rng.integers(0, users, size=20000)
    iids = rng.integers(0, items, size=20000)
    ui = bipartite(uids, iids, users, items)

    def run():
        item_cooccurrence(ui)

    benchmark(run)


def bench_mask_100k(benchmark) -> None:
    def run():
        bernoulli(100000, 4, 0.4, seed=0)

    benchmark(run)
