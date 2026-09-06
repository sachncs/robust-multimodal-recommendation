"""Public API for the morel.complete package."""

import torch.nn as nn

from morel.complete.decoders import Decoders
from morel.core.registry import Registry

#: Selectable modality completers, keyed by ``config.complete.kind``.
COMPLETERS: Registry[nn.Module] = Registry("completer")


@COMPLETERS.register("mlp")
def build_mlp(*, latent_dim: int, dims: dict[str, int], hidden: int) -> nn.Module:
    """Build the per-modality MLP decoder bank used by the full model."""
    return Decoders(latent_dim=latent_dim, dims=dims, hidden=hidden)


__all__ = ["COMPLETERS", "Decoders", "build_mlp"]
