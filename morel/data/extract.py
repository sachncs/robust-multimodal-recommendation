"""Modality-agnostic feature extractors.

A single ``FeatureEncoder`` Protocol covers text and visual encoders.
Implementations are responsible for producing L2-normalized ``float32``
arrays of shape ``(items, dim)``.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
import torch

from morel.core.errors import DataError
from morel.core.log import get as get_logger

log = get_logger("data.extract")


class FeatureEncoder(Protocol):
    """One feature extractor for raw modality inputs."""

    name: str
    dim: int

    def encode(self, inputs: list[str], *, device: str | torch.device | None = None) -> np.ndarray:
        """Encode a batch of inputs to ``(len(inputs), self.dim)`` float32."""
        ...


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    """L2-normalize each row; replace zero-norm rows with the zero vector."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return (array / safe).astype(np.float32, copy=False)


def text(
    inputs: list[str],
    encoder: FeatureEncoder,
    *,
    batch: int = 64,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Encode text inputs through any FeatureEncoder implementation.

    Args:
        inputs: List of strings.
        encoder: A text or multimodal encoder.
        batch: Batch size.
        device: Device override.

    Returns
    -------
        Array of shape ``(len(inputs), encoder.dim)``, L2-normalized, float32.
    """
    if not inputs:
        raise DataError("text encoder received empty input list")
    return encoder.encode(inputs, device=device)


def visual(
    paths: list[str],
    encoder: FeatureEncoder,
    *,
    batch: int = 32,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Encode image paths through any FeatureEncoder implementation.

    Args:
        paths: List of filesystem paths to images.
        encoder: A visual encoder.
        batch: Batch size.
        device: Device override.

    Returns
    -------
        Tuple of ``(features, kept_indices)`` where ``features`` is
        ``(len(kept), encoder.dim)`` L2-normalized float32 and ``kept_indices``
        are positions in the original ``paths`` that succeeded.
    """
    if not paths:
        raise DataError("visual encoder received empty input list")
    return encoder.encode(paths, device=device), list(range(len(paths)))


def random(items: int, dim: int, *, seed: int, name: str = "random") -> np.ndarray:
    """Deterministic random L2-normalized features.

    Used in tests, demos, and as a fallback when real encoders are unavailable.
    """
    if items <= 0:
        raise DataError(f"items must be positive, got {items}")
    if dim <= 0:
        raise DataError(f"dim must be positive, got {dim}")
    rng = np.random.default_rng(seed)
    array = rng.normal(size=(items, dim)).astype(np.float32)
    return _l2_normalize(array)


def fingerprint(payload: np.ndarray) -> str:
    """Stable SHA256 of an array's bytes (used for manifest binding)."""
    return hashlib.sha256(np.ascontiguousarray(payload).tobytes()).hexdigest()


__all__ = [
    "FeatureEncoder",
    "text",
    "visual",
    "random",
    "fingerprint",
    "_l2_normalize",
]
