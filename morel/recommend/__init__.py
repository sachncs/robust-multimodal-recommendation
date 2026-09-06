"""Public API for the morel.recommend package."""

import torch.nn as nn

from morel.core.registry import Registry
from morel.recommend.baseline import MF, Pop
from morel.recommend.bpr import bpr, distinct_ranks, negatives, ranks_to_items
from morel.recommend.light import Light
from morel.recommend.protocol import Recommender

#: Selectable downstream rankers, keyed by ``config.recommend.kind``.
RECOMMENDERS: Registry[nn.Module] = Registry("recommender")


@RECOMMENDERS.register("light")
def build_light(
    *, users: int, items: int, embed: int, layers: int, seed: int | None = None
) -> nn.Module:
    """Build the LightGCN ranker used by the full model."""
    return Light(users=users, items=items, embed=embed, layers=layers, seed=seed)


@RECOMMENDERS.register("mf")
def build_mf(
    *, users: int, items: int, embed: int, layers: int, seed: int | None = None
) -> nn.Module:
    """Build a matrix-factorization ranker; it has no propagation layers."""
    del layers
    return MF(users=users, items=items, embed=embed, seed=seed)


@RECOMMENDERS.register("pop")
def build_pop(
    *, users: int, items: int, embed: int, layers: int, seed: int | None = None
) -> nn.Module:
    """Build the popularity baseline, which has no learned parameters."""
    del embed, layers, seed
    return Pop(users=users, items=items)


__all__ = [
    "MF",
    "RECOMMENDERS",
    "Light",
    "Pop",
    "Recommender",
    "bpr",
    "build_light",
    "build_mf",
    "build_pop",
    "distinct_ranks",
    "negatives",
    "ranks_to_items",
]
