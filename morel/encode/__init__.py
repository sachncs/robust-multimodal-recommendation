"""Public API for the morel.encode package."""

import torch.nn as nn

from morel.core.registry import Registry
from morel.encode.baseline import GraphEncoder, GraphEncoderBaseline, Identity, Sum
from morel.encode.input import Input
from morel.encode.layer import Layer
from morel.encode.pool import CLS, Attention, Mean, Token
from morel.encode.transformer import Transformer

#: Selectable joint encoders, keyed by ``config.encode.kind``.
#:
#: A registered encoder must accept ``(features, mask, pe)`` plus the optional
#: ``attention_mask`` and ``sequence`` arguments, and return one
#: ``hidden``-width embedding per batch item. ``Sum`` is deliberately absent:
#: it concatenates without projecting, so its output width is not ``hidden``
#: and the router downstream cannot consume it.
ENCODERS: Registry[nn.Module] = Registry("encoder")


@ENCODERS.register("transformer")
def build_transformer(
    *,
    dims: dict[str, int],
    pe_dim: int,
    hidden: int,
    layers: int,
    heads: int,
    dropout: float,
) -> nn.Module:
    """Build the masked transformer encoder used by the full model."""
    return Transformer(
        dims=dims,
        pe_dim=pe_dim,
        hidden=hidden,
        layers=layers,
        heads=heads,
        dropout=dropout,
    )


@ENCODERS.register("identity")
def build_identity_encoder(
    *,
    dims: dict[str, int],
    pe_dim: int,
    hidden: int,
    layers: int,
    heads: int,
    dropout: float,
) -> nn.Module:
    """Build the linear no-transformer ablation encoder."""
    del layers, heads, dropout
    return Identity(dims, pe_dim, hidden)


__all__ = [
    "CLS",
    "ENCODERS",
    "Attention",
    "GraphEncoder",
    "GraphEncoderBaseline",
    "Identity",
    "Input",
    "Layer",
    "Mean",
    "Sum",
    "Token",
    "Transformer",
    "build_identity_encoder",
    "build_transformer",
]
