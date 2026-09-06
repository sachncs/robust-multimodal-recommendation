"""Retrieval pipeline: anchor retrieval + ACS + MAGE in one call.

The single-query path is exposed as ``retrieve`` and is the workhorse called
from the model. The batch path returns padded index tensors plus an attention
mask for downstream sequence models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from morel.retrieve.anchor import query as anchor_query
from morel.retrieve.mage import expand as mage_expand


@dataclass(frozen=True)
class Result:
    """Retrieval output for one query or a batch."""

    nodes: np.ndarray  # (B, max_S) int64
    sizes: np.ndarray  # (B,) int64
    mask: np.ndarray  # (B, max_S) bool, True = real node

    @property
    def batch(self) -> int:
        """Return the batch size."""
        return int(self.sizes.shape[0])

    @property
    def max_size(self) -> int:
        """Return the maximum subgraph size."""
        return int(self.nodes.shape[1])


def retrieve(
    query: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adj: sp.csr_matrix,
    *,
    anchors: int = 10,
    iters: int = 10,
    fallback: str = "anchors",
) -> set[int]:
    """Retrieve and expand a subgraph for one query item."""
    observed = [name for idx, name in enumerate(features.keys()) if mask[query, idx] > 0]
    if not observed:
        return {int(query)}
    anchor_set: set[int] = set()
    for name in observed:
        anchor_set.update(int(a) for a in anchor_query(query, name, features, mask, top=anchors))
    if not anchor_set:
        return {int(query)}
    subgraph = mage_expand(
        adj,
        list(anchor_set),
        query,
        features,
        mask,
        iters=iters,
        fallback=fallback,
    )
    subgraph.add(int(query))
    return subgraph


def batch(
    queries: list[int],
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adj: sp.csr_matrix,
    *,
    anchors: int = 10,
    iters: int = 10,
    fallback: str = "anchors",
) -> Result:
    """Batched retrieval that returns padded tensors for downstream models."""
    subgraphs = [
        retrieve(q, features, mask, adj, anchors=anchors, iters=iters, fallback=fallback)
        for q in queries
    ]
    sizes = np.array([len(s) for s in subgraphs], dtype=np.int64)
    max_size = int(sizes.max()) if sizes.size else 0
    nodes = np.zeros((len(queries), max_size), dtype=np.int64)
    valid = np.zeros((len(queries), max_size), dtype=bool)
    for i, sg in enumerate(subgraphs):
        arr = np.array(sorted(sg), dtype=np.int64)
        nodes[i, : arr.size] = arr
        valid[i, : arr.size] = True
    return Result(nodes=nodes, sizes=sizes, mask=valid)


def as_tensor(
    result: Result, *, device: str | torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a batched Result into (nodes, mask, sizes) torch tensors."""
    nodes = torch.from_numpy(result.nodes).long()
    mask_t = torch.from_numpy(result.mask).bool()
    sizes = torch.from_numpy(result.sizes).long()
    if device is not None:
        nodes = nodes.to(device)
        mask_t = mask_t.to(device)
        sizes = sizes.to(device)
    return nodes, mask_t, sizes


__all__ = ["Result", "as_tensor", "batch", "retrieve"]
