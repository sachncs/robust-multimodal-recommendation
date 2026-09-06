"""Subgraph view over an Item graph.

A Subgraph is a sorted set of node indices into a parent Item graph. It
exposes neighbor queries and connectivity checks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from morel.core.errors import GraphError


@dataclass(frozen=True)
class Subgraph:
    """A node-subset view of an Item graph."""

    nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __post_init__(self) -> None:
        """Validate that nodes are a 1-D array of non-negative indices."""
        if self.nodes.ndim != 1:
            raise GraphError("subgraph nodes must be 1-D")
        if self.nodes.size > 0 and self.nodes.min() < 0:
            raise GraphError("subgraph nodes must be non-negative")

    @property
    def size(self) -> int:
        """Return the number of nodes in the subgraph."""
        return int(self.nodes.size)

    def __len__(self) -> int:
        """Return the number of nodes in the subgraph."""
        return self.size

    def __iter__(self) -> Iterator[int]:
        """Iterate over the node indices."""
        return iter(self.nodes.tolist())

    def __contains__(self, node: int) -> bool:
        """Return whether ``node`` is in the subgraph."""
        return int(node) in self.nodes

    def to_set(self) -> set[int]:
        """Return the subgraph as a Python set."""
        return set(self.nodes.tolist())

    @classmethod
    def from_idx(cls, indices: list[int] | np.ndarray) -> Subgraph:
        """Construct from a list of indices, deduplicating and sorting."""
        return cls(nodes=np.unique(np.asarray(indices, dtype=np.int64)))


def connected(subgraph_nodes: np.ndarray, parent_neighbors: dict[int, np.ndarray]) -> bool:
    """Return True if the subgraph nodes form a connected subgraph.

    Args:
        subgraph_nodes: Sorted 1-D array of node indices.
        parent_neighbors: Mapping from node id to sorted array of neighbor ids.
    """
    if subgraph_nodes.size <= 1:
        return True
    set_nodes = set(subgraph_nodes.tolist())
    start = int(subgraph_nodes[0])
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in parent_neighbors.get(node, ()):
            n_int = int(neighbor)
            if n_int in set_nodes and n_int not in visited:
                visited.add(n_int)
                queue.append(n_int)
    return len(visited) == subgraph_nodes.size


__all__ = ["Subgraph", "connected"]
