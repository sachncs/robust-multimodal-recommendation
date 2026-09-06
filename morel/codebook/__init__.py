"""Public API for the morel.codebook package."""

from morel.codebook.codebook import (
    VQ,
    Codebook,
    GumbelVQ,
    IdentityCodebook,
    balance,
    usage,
)
from morel.core.registry import Registry
from morel.route import Router

#: Selectable codebooks, keyed by ``config.codebook.kind``.
CODEBOOKS: Registry[Codebook] = Registry("codebook")


@CODEBOOKS.register("gumbel")
def build_gumbel(*, dim: int, size: int, router: Router, seed: int | None = None) -> Codebook:
    """Build the router-driven codebook used by the full model."""
    return GumbelVQ(dim=dim, size=size, router=router, seed=seed)


@CODEBOOKS.register("vq")
def build_vq(*, dim: int, size: int, router: Router, seed: int | None = None) -> Codebook:
    """Build a nearest-neighbour vector-quantizing codebook.

    The router is accepted for a uniform factory signature; plain VQ selects
    codes by distance and does not consult it.
    """
    del router
    return VQ(dim=dim, size=size, seed=seed)


@CODEBOOKS.register("identity")
def build_identity(*, dim: int, size: int, router: Router, seed: int | None = None) -> Codebook:
    """Build the pass-through codebook used for the no-codebook ablation."""
    del router, seed
    return IdentityCodebook(dim=dim, size=size)


__all__ = [
    "CODEBOOKS",
    "VQ",
    "Codebook",
    "GumbelVQ",
    "IdentityCodebook",
    "balance",
    "build_gumbel",
    "build_identity",
    "build_vq",
    "usage",
]
