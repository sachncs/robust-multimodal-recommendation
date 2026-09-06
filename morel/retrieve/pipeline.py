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

from morel.core.errors import Cfg
from morel.retrieve.acs import compute as acs_compute
from morel.retrieve.anchor import query as anchor_query
from morel.retrieve.bfs import bfs as bfs_distances
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
    def peak(self) -> int:
        """Return the maximum subgraph size."""
        return int(self.nodes.shape[1])


def for_query(
    query: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    anchors: int,
) -> set[int]:
    """Return the cosine-nearest anchors of ``query`` across observed modalities."""
    observed = [name for idx, name in enumerate(features.keys()) if mask[query, idx] > 0]
    found: set[int] = set()
    for name in observed:
        found.update(int(a) for a in anchor_query(query, name, features, mask, top=anchors))
    return found


def mage(
    query: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adj: sp.csr_matrix,
    *,
    anchors: int,
    iters: int,
    fallback: str,
) -> set[int]:
    """Anchor retrieval followed by MAGE expansion; the method's default."""
    anchor_set = for_query(query, features, mask, anchors=anchors)
    if not anchor_set:
        return {int(query)}
    return mage_expand(adj, list(anchor_set), query, features, mask, iters=iters, fallback=fallback)


def acs(
    query: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adj: sp.csr_matrix,
    *,
    anchors: int,
    iters: int,
    fallback: str,
) -> set[int]:
    """Anchor retrieval followed by the Anchor Connecting Subgraph, without MAGE."""
    del iters
    anchor_set = for_query(query, features, mask, anchors=anchors)
    if not anchor_set:
        return {int(query)}
    return acs_compute(adj, sorted(anchor_set), fallback=fallback)


def anchor(
    query: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adj: sp.csr_matrix,
    *,
    anchors: int,
    iters: int,
    fallback: str,
) -> set[int]:
    """Anchors only, with no graph expansion; the no-expansion ablation."""
    del adj, iters, fallback
    return for_query(query, features, mask, anchors=anchors)


def bfs(
    query: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adj: sp.csr_matrix,
    *,
    anchors: int,
    iters: int,
    fallback: str,
) -> set[int]:
    """Graph neighbourhood of the query within ``iters`` hops, ignoring features.

    Uses only the graph, so subgraph selection does not depend on the modality
    features at all.
    """
    del features, mask, anchors, fallback
    distances = bfs_distances(adj, [int(query)])
    return {int(node) for node, hops in distances.items() if hops <= max(int(iters), 1)}


def none(
    query: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adj: sp.csr_matrix,
    *,
    anchors: int,
    iters: int,
    fallback: str,
) -> set[int]:
    """Return the query alone: the no-retrieval ablation.

    Nothing is retrieved, so the encoder sees only the item being completed
    and no graph context. This is the condition ``eval.ablations`` calls
    ``noretry``.
    """
    del features, mask, adj, anchors, iters, fallback
    return {int(query)}


#: Map from config name to strategy function. Module-local; no global registry.
KIND: dict[str, object] = {
    "mage": mage,
    "acs": acs,
    "anchor": anchor,
    "bfs": bfs,
    "none": none,
}


def retrieve(
    query: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    adj: sp.csr_matrix,
    *,
    anchors: int = 10,
    iters: int = 10,
    fallback: str = "anchors",
    kind: str = "mage",
) -> set[int]:
    """Retrieve and expand a subgraph for one query item.

    Args:
        query: Item id to build a subgraph around.
        features: Per-modality feature arrays.
        mask: ``(items, modalities)`` availability.
        adj: Symmetric item-item adjacency.
        anchors: Number of cosine neighbours per observed modality.
        iters: Expansion budget, interpreted by the strategy.
        fallback: Behaviour when expansion finds no collision root.
        kind: Registered retrieval strategy; see ``morel.retrieve.STRATEGIES``.

    Returns
    -------
        Node ids of the retrieved subgraph, always including ``query``.

    Raises
    ------
        Cfg: If ``kind`` is not a registered strategy.
    """
    if kind not in KIND:
        raise Cfg(
            f"unknown retrieval strategy {kind!r}; available: {', '.join(sorted(KIND)) or '(none)'}"
        )
    strategy = KIND[kind]
    observed = [name for idx, name in enumerate(features.keys()) if mask[query, idx] > 0]
    if not observed and kind != "bfs":
        return {int(query)}
    subgraph = strategy(  # type: ignore[operator]
        query,
        features,
        mask,
        adj,
        anchors=anchors,
        iters=iters,
        fallback=fallback,
    )
    subgraph = set(subgraph)
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
    kind: str = "mage",
) -> Result:
    """Batched retrieval that returns padded tensors for downstream models."""
    subgraphs = [
        retrieve(q, features, mask, adj, anchors=anchors, iters=iters, fallback=fallback, kind=kind)
        for q in queries
    ]
    sizes = np.array([len(s) for s in subgraphs], dtype=np.int64)
    peak = int(sizes.max()) if sizes.size else 0
    nodes = np.zeros((len(queries), peak), dtype=np.int64)
    valid = np.zeros((len(queries), peak), dtype=bool)
    for i, sg in enumerate(subgraphs):
        arr = np.array(sorted(sg), dtype=np.int64)
        nodes[i, : arr.size] = arr
        valid[i, : arr.size] = True
    return Result(nodes=nodes, sizes=sizes, mask=valid)


def cast(
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


__all__ = ["Cfg", "Result", "batch", "cast", "retrieve"]
