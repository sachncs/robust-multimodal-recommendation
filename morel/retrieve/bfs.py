"""BFS and shortest-path utilities."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp

from morel.core.errors import Net


def bfs(adj: sp.csr_matrix, sources: list[int]) -> dict[int, int]:
    """Multi-source BFS.

    Args:
        adj: Symmetric adjacency matrix.
        sources: Source node ids.

    Returns
    -------
        Dict mapping visited node id to its shortest distance from the
        nearest source.
    """
    if not sources:
        return {}
    distances = dict.fromkeys(sources, 0)
    queue: deque[int] = deque(sources)
    while queue:
        node = queue.popleft()
        row_start = adj.indptr[node]
        row_end = adj.indptr[node + 1]
        for neighbor in adj.indices[row_start:row_end]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def path(adj: sp.csr_matrix, start: int, end: int) -> list[int]:
    """Return a shortest path from ``start`` to ``end``.

    Args:
        adj: Symmetric adjacency matrix.
        start: Source node id.
        end: Target node id.

    Returns
    -------
        List of node ids on a shortest path. Empty if unreachable.
        Single-element list ``[start]`` if ``start == end``.
    """
    if start == end:
        return [start]
    if start < 0 or end < 0:
        raise Net("negative node id")
    if start >= adj.shape[0] or end >= adj.shape[0]:
        raise Net("node id out of range")
    previous: dict[int, int | None] = {start: None}
    distances: dict[int, int] = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            result = [node]
            parent = previous[node]
            while parent is not None:
                result.append(parent)
                parent = previous[parent]
            return result[::-1]
        row_start = adj.indptr[node]
        row_end = adj.indptr[node + 1]
        for neighbor in adj.indices[row_start:row_end]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                previous[neighbor] = node
                queue.append(int(neighbor))
    return []


def iter_neighbors(adj: sp.csr_matrix, node: int) -> Iterator[int]:
    """Iterate over the neighbors of ``node``."""
    row_start = adj.indptr[node]
    row_end = adj.indptr[node + 1]
    return iter(int(n) for n in adj.indices[row_start:row_end])


def neighbor_array(adj: sp.csr_matrix) -> dict[int, np.ndarray]:
    """Precompute neighbor arrays for every node."""
    nodes = adj.shape[0]
    out: dict[int, np.ndarray] = {}
    for node in range(nodes):
        row_start = adj.indptr[node]
        row_end = adj.indptr[node + 1]
        out[node] = np.sort(adj.indices[row_start:row_end].astype(np.int64, copy=False))
    return out


__all__ = ["bfs", "iter_neighbors", "neighbor_array", "path"]
