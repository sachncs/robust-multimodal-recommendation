"""Public API for the morel.codebook package."""

from morel.codebook.codebook import (
    VQ,
    Codebook,
    GumbelVQ,
    IdentityCodebook,
    balance,
    usage,
)
from morel.route import Router


def build(
    kind: str,
    *,
    dim: int,
    size: int,
    router: Router,
    seed: int | None = None,
) -> Codebook:
    """Build the codebook selected by ``config.codebook.kind``.

    Args:
        kind: Codebook name. One of ``"gumbel"``, ``"vq"``, ``"identity"``.
        dim: Embedding dimension.
        size: Number of codebook entries.
        router: Router used by Gumbel-VQ to produce assignment weights.
        seed: Optional RNG seed for reproducibility.

    Returns
    -------
        The constructed codebook.

    Raises
    ------
        ValueError: If ``kind`` is not a known codebook name.
    """
    if kind == "gumbel":
        return GumbelVQ(dim=dim, size=size, router=router, seed=seed)
    if kind == "vq":
        return VQ(dim=dim, size=size, seed=seed)
    if kind == "identity":
        return IdentityCodebook(dim=dim, size=size)
    raise ValueError(f"unknown codebook kind {kind!r}; available: gumbel, vq, identity")


#: Map from config name to codebook class for introspection.
KIND: dict[str, type[Codebook]] = {
    "gumbel": GumbelVQ,
    "vq": VQ,
    "identity": IdentityCodebook,
}


__all__ = [
    "KIND",
    "VQ",
    "Codebook",
    "GumbelVQ",
    "IdentityCodebook",
    "balance",
    "build",
    "usage",
]
