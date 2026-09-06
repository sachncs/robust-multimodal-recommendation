"""Anchor Connecting Subgraph (ACS) — Algorithm 1 from the paper.

Multi-source BFS with reachability bitmasks. The first node whose bitmask
equals the all-anchors mask is the collision root. The ACS is the union of
shortest paths from the collision root back to each anchor.

Correctness fixes over the legacy implementation:
- iterative backtrack (no RecursionError on long paths)
- explicit self-loop guard
- duplicate anchor rejection
- deterministic fallback when anchors are not mutually reachable
"""

from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp

from morel.core.errors import GraphError
from morel.core.log import get as get_logger
from morel.graph import invariants

log = get_logger("retrieve.acs")


def validate_acs_inputs(adj: sp.csr_matrix, anchors: list[int]) -> None:
    if anchors is None:
        raise GraphError("anchors must not be None")
    if not anchors:
        return
    nodes = adj.shape[0]
    seen: set[int] = set()
    for anchor in anchors:
        if not isinstance(anchor, (int, np.integer)):
            raise GraphError(f"anchor must be int, got {type(anchor).__name__}")
        idx = int(anchor)
        if idx < 0 or idx >= nodes:
            raise GraphError(f"anchor {idx} out of range [0, {nodes})")
        if idx in seen:
            raise GraphError(f"duplicate anchor {idx}")
        seen.add(idx)


def compute(adj: sp.csr_matrix, anchors: list[int], *, fallback: str = "anchors") -> set[int]:
    """Compute the Anchor Connecting Subgraph.

    Args:
        adj: Symmetric adjacency.
        anchors: Distinct anchor node ids.
        fallback: Behaviour when no collision root is found. ``"anchors"``
            returns the anchor set (default); ``"empty"`` returns an empty
            set.

    Returns
    -------
        Set of node ids in the ACS.

    Raises
    ------
        GraphError: On bad input.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise GraphError(f"adj must be square, got {adj.shape}")
    validate_acs_inputs(adj, anchors)
    invariants.no_self_loops(adj)
    if not anchors:
        return set()
    if len(anchors) == 1:
        return {int(anchors[0])}

    anchors_int = [int(a) for a in anchors]
    k = len(anchors_int)
    all_mask = (1 << k) - 1
    bitmask: dict[int, int] = {}
    previous: dict[int, dict[int, int]] = {}
    queue: deque[int] = deque()
    for idx, anchor in enumerate(anchors_int):
        queue.append(anchor)
        bitmask[anchor] = 1 << idx
        previous[anchor] = {}

    collision: int | None = None
    while queue and collision is None:
        node = queue.popleft()
        bits_node = bitmask[node]
        row_start = adj.indptr[node]
        row_end = adj.indptr[node + 1]
        for neighbor in adj.indices[row_start:row_end]:
            if neighbor == node:
                continue
            if neighbor not in bitmask:
                bitmask[neighbor] = 0
                previous[neighbor] = {}
            new_bits = bits_node & ~bitmask[neighbor]
            if new_bits == 0:
                continue
            for bit in range(k):
                if (new_bits >> bit) & 1:
                    previous[neighbor][bit] = node
            bitmask[neighbor] |= new_bits
            queue.append(int(neighbor))
            if bitmask[neighbor] == all_mask:
                collision = int(neighbor)
                break

    if collision is None:
        if fallback == "empty":
            log.warning(
                "no collision root found; returning empty set",
                extra={"anchors": anchors_int},
            )
            return set()
        if fallback == "anchors":
            log.warning(
                "no collision root found; returning anchor set",
                extra={"anchors": anchors_int},
            )
            return set(anchors_int)
        raise GraphError(f"unknown fallback strategy: {fallback!r}")

    subgraph: set[int] = {int(collision)}
    for anchor_idx, target in enumerate(anchors_int):
        if target == int(collision):
            continue
        visited: set[int] = set()
        stack: list[int] = [int(collision)]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node == target:
                continue
            nxt = previous.get(node, {}).get(anchor_idx)
            if nxt is None:
                raise GraphError(f"backtrack failed at node {node} for anchor {target}")
            stack.append(int(nxt))
        subgraph.update(visited)
    return subgraph


def batch(
    adj: sp.csr_matrix, anchor_sets: list[list[int]], *, fallback: str = "anchors"
) -> list[set[int]]:
    """Compute ACS for a batch of anchor sets."""
    return [compute(adj, anchors, fallback=fallback) for anchors in anchor_sets]


__all__ = ["batch", "compute"]
