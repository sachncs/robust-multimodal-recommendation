"""Modality-Aware Graph Expansion (MAGE) — Algorithm 2 from the paper.

Greedy boundary add/remove that maximizes mean relevance while preserving
connectivity and all anchors.

Correctness fixes over the legacy implementation:
- best-improvement hill climbing (canonical Algorithm 2)
- sorted boundary iteration (deterministic across runs)
- cached neighbor arrays (no O(N) neighbor rebuild per iteration)
- iterative bounds checking
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from morel.core.errors import Net
from morel.core.log import get as get_logger
from morel.graph.subgraph import connected
from morel.retrieve.acs import compute as acs_compute
from morel.retrieve.bfs import neighbor_array
from morel.retrieve.relevance import rel

log = get_logger("retrieve.mage")


def boundary_nodes(subgraph: set[int], neighbors: dict[int, np.ndarray]) -> list[int]:
    """Return the sorted list of nodes adjacent to ``subgraph`` but not in it."""
    boundary: set[int] = set()
    for node in subgraph:
        for neighbor in neighbors.get(node, ()):
            if neighbor not in subgraph:
                boundary.add(int(neighbor))
    return sorted(boundary)


def neighbors(
    subgraph_nodes: np.ndarray, neighbors: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Build a neighbor lookup restricted to ``subgraph_nodes``."""
    set_nodes = set(subgraph_nodes.tolist())
    return {
        int(node): np.array(
            [int(n) for n in neighbors.get(int(node), ()) if int(n) in set_nodes],
            dtype=np.int64,
        )
        for node in subgraph_nodes
    }


def expand(
    adj: sp.csr_matrix,
    anchors: list[int],
    query_item: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    iters: int = 10,
    fallback: str = "anchors",
) -> set[int]:
    """Run MAGE greedy expansion.

    Args:
        adj: Symmetric item-item adjacency.
        anchors: Anchor set from anchor retrieval.
        query_item: Query node id.
        features: Per-modality feature arrays.
        mask: Modality availability mask.
        iters: Maximum greedy iterations.
        fallback: ACS fallback strategy.

    Returns
    -------
        Set of node ids in the expanded subgraph.
    """
    if iters <= 0:
        raise Net(f"iters must be positive, got {iters}")
    anchors_int = [int(a) for a in anchors]
    if query_item not in anchors_int:
        anchors_int = sorted({*anchors_int, int(query_item)})

    subgraph_set = acs_compute(adj, anchors_int, fallback=fallback)
    if not subgraph_set:
        subgraph_set = set(anchors_int)
    subgraph_set.add(int(query_item))
    neighbors = neighbor_array(adj)
    best_score = rel(query_item, np.array(sorted(subgraph_set), dtype=np.int64), features, mask)

    for _ in range(iters):
        changed = False
        boundary = boundary_nodes(subgraph_set, neighbors)
        best_candidate: int | None = None
        best_candidate_score = best_score
        for node in boundary:
            trial = subgraph_set | {node}
            if not connected(np.array(sorted(trial), dtype=np.int64), neighbors):
                continue
            score = rel(
                query_item,
                np.array(sorted(trial), dtype=np.int64),
                features,
                mask,
            )
            if score > best_candidate_score + 1e-12:
                best_candidate = node
                best_candidate_score = score
        if best_candidate is not None:
            subgraph_set.add(best_candidate)
            best_score = best_candidate_score
            changed = True
        else:
            best_removal: int | None = None
            best_removal_score = best_score
            for node in sorted(subgraph_set):
                if node in anchors_int:
                    continue
                trial = subgraph_set - {node}
                if not trial:
                    continue
                if not connected(np.array(sorted(trial), dtype=np.int64), neighbors):
                    continue
                score = rel(
                    query_item,
                    np.array(sorted(trial), dtype=np.int64),
                    features,
                    mask,
                )
                if score > best_removal_score + 1e-12:
                    best_removal = node
                    best_removal_score = score
            if best_removal is not None:
                subgraph_set.discard(best_removal)
                best_score = best_removal_score
                changed = True
        if not changed:
            break
    return subgraph_set


def batch(
    adj: sp.csr_matrix,
    anchor_sets: list[list[int]],
    queries: list[int],
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    iters: int = 10,
    fallback: str = "anchors",
) -> list[set[int]]:
    """Run MAGE for a batch of queries and anchor sets."""
    return [
        expand(adj, anchors, query, features, mask, iters=iters, fallback=fallback)
        for anchors, query in zip(anchor_sets, queries, strict=True)
    ]


__all__ = ["batch", "expand"]
