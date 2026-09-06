"""Public API for the morel.encode package."""

import torch.nn as nn

from morel.core.errors import ConfigError
from morel.encode.baseline import Baseline, Enc, Identity, Sum
from morel.encode.input import Input
from morel.encode.layer import Layer
from morel.encode.pool import CLS, Attention, Mean, Token
from morel.encode.transformer import Transformer


def build(
    kind: str,
    *,
    dims: dict[str, int],
    pe_dim: int,
    hidden: int,
    layers: int,
    heads: int,
    dropout: float,
) -> nn.Module:
    """Build the joint encoder selected by ``config.encode.kind``.

    Args:
        kind: Encoder name. One of ``"transformer"`` or ``"identity"``.
        dims: Mapping from modality name to its feature dimension.
        pe_dim: Positional-encoding dimension.
        hidden: Hidden width of the encoder output.
        layers: Number of transformer layers (ignored for ``identity``).
        heads: Number of attention heads (ignored for ``identity``).
        dropout: Dropout rate (ignored for ``identity``).

    Returns
    -------
        The constructed encoder module.

    Raises
    ------
        ValueError: If ``kind`` is not a known encoder name.
    """
    if kind == "transformer":
        return Transformer(
            dims=dims,
            pe_dim=pe_dim,
            hidden=hidden,
            layers=layers,
            heads=heads,
            dropout=dropout,
        )
    if kind == "identity":
        return Identity(dims, pe_dim, hidden)
    raise ConfigError(f"unknown encoder '{kind}'; available: transformer, identity")


#: Map from config name to encoder class for introspection.
KIND: dict[str, type[nn.Module]] = {
    "transformer": Transformer,
    "identity": Identity,
}


__all__ = [
    "CLS",
    "KIND",
    "Attention",
    "Baseline",
    "Enc",
    "Identity",
    "Input",
    "Layer",
    "Mean",
    "Sum",
    "Token",
    "Transformer",
    "build",
]
