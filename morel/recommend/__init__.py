"""Public API for the morel.recommend package."""

import torch.nn as nn

from morel.core.errors import ConfigError
from morel.recommend.baseline import MF, Pop
from morel.recommend.bpr import bpr, distinct_ranks, negatives, ranks_to_items
from morel.recommend.light import Light
from morel.recommend.protocol import Recommender


def build(
    kind: str,
    *,
    users: int,
    items: int,
    embed: int,
    layers: int,
    feature_dim: int | None = None,
    seed: int | None = None,
) -> nn.Module:
    """Build the downstream ranker selected by ``config.recommend.kind``.

    Args:
        kind: Ranker name. One of ``"light"``, ``"mf"``, ``"pop"``.
        users: Number of users.
        items: Number of items.
        embed: Embedding dimension.
        layers: Number of propagation layers (ignored for non-graph rankers).
        feature_dim: Optional modality feature dimension for ``"light"``.
        seed: Optional RNG seed.

    Returns
    -------
        The constructed ranker module.

    Raises
    ------
        ValueError: If ``kind`` is not a known ranker name.
    """
    if kind == "light":
        return Light(
            users=users,
            items=items,
            embed=embed,
            layers=layers,
            feature_dim=feature_dim,
            seed=seed,
        )
    if kind == "mf":
        return MF(users=users, items=items, embed=embed, seed=seed)
    if kind == "pop":
        return Pop(users=users, items=items)
    raise ConfigError(f"unknown recommender kind {kind!r}; available: light, mf, pop")


#: Map from config name to ranker class for introspection.
KIND: dict[str, type[nn.Module]] = {
    "light": Light,
    "mf": MF,
    "pop": Pop,
}


__all__ = [
    "KIND",
    "MF",
    "Light",
    "Pop",
    "Recommender",
    "bpr",
    "build",
    "distinct_ranks",
    "negatives",
    "ranks_to_items",
]
