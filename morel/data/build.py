"""Graph construction: bipartite user-item and item-item co-occurrence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from morel.core.errors import DataError
from morel.core.log import get as get_logger

log = get_logger("data.build")


def bipartite(user: np.ndarray, item: np.ndarray, users: int, items: int) -> sp.csr_matrix:
    """Build a user-item bipartite adjacency matrix.

    Args:
        user: 1-D array of user indices.
        item: 1-D array of item indices, same length as ``user``.
        users: Total number of users.
        items: Total number of items.

    Returns
    -------
        Sparse CSR matrix of shape ``(users, items)`` with float32 ones.
    """
    if user.shape != item.shape:
        raise DataError(f"shape mismatch: user {user.shape} vs item {item.shape}")
    data = np.ones(user.shape[0], dtype=np.float32)
    matrix = sp.csr_matrix((data, (user, item)), shape=(users, items))
    matrix.sum_duplicates()
    return matrix


def item_cooccurrence(graph: sp.csr_matrix) -> sp.csr_matrix:
    """Build an item-item co-occurrence graph from a user-item bipartite.

    The result is binarized, symmetrized, and self-loops are removed.

    Args:
        graph: User-item adjacency of shape ``(users, items)``.

    Returns
    -------
        Symmetric item-item CSR matrix with no self-loops.
    """
    if graph.ndim != 2:
        raise DataError(f"bipartite must be 2-D, got {graph.ndim}-D")
    cooc = graph.T @ graph
    cooc = cooc.sign()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    return cooc.tocsr()


def kcore(graph: sp.csr_matrix, min_edges: int) -> sp.csr_matrix:
    """Peel nodes below ``min_edges`` until stable.

    Implements strict k-core: every node in the returned graph has at least
    ``min_edges`` surviving neighbors.

    Args:
        graph: Symmetric item-item CSR.
        min_edges: Minimum number of neighbors per node.

    Returns
    -------
        The largest subgraph satisfying the k-core invariant.
    """
    if min_edges <= 0:
        return graph
    current = graph.copy()
    while True:
        degrees = np.asarray(current.sum(axis=1)).flatten()
        keep = degrees >= min_edges
        if keep.all():
            return current
        keep_indices = np.where(keep)[0]
        # Remap kept indices to a contiguous range.
        mapping = -np.ones(current.shape[0], dtype=np.int64)
        mapping[keep_indices] = np.arange(keep_indices.shape[0])
        sub = current[keep_indices][:, keep_indices]
        # Rebuild as CSR with the new contiguous ids.
        sub = sub.tocsr()
        sub.sum_duplicates()
        current = sub
        if current.shape[0] == 0:
            return current


def interactions(
    review: Path | str,
    metadata: Path | str,
    *,
    min_edges: int = 5,
) -> tuple[sp.csr_matrix, dict[int, dict[str, Any]], int, int]:
    """Load user-item interactions and item metadata from Amazon JSON files.

    Performs iterative k-core filtering so every returned user has at least
    ``min_edges`` interactions.

    Args:
        review: Path to the Amazon reviews ``.json`` file.
        metadata: Path to the Amazon metadata ``.json`` file.
        min_edges: K-core threshold.

    Returns
    -------
        Tuple of ``(ui_graph, item_meta, users, items)``.
    """
    review_path = Path(review)
    metadata_path = Path(metadata)
    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    user_item_pairs: list[tuple[str, str]] = []
    skipped = 0
    with review_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            user = record.get("reviewerID")
            item = record.get("asin")
            if user is None or item is None:
                skipped += 1
                continue
            user_item_pairs.append((user, item))
            user_counts[user] += 1
            item_counts[item] += 1
    if skipped:
        log.warning("skipped malformed review lines", extra={"count": skipped})

    def filter_pairs(keep_users: set[str], keep_items: set[str]) -> list[tuple[str, str]]:
        return [(u, i) for u, i in user_item_pairs if u in keep_users and i in keep_items]

    filtered = filter_pairs(
        {u for u, c in user_counts.items() if c >= min_edges},
        {i for i, c in item_counts.items() if c >= min_edges},
    )
    # Iterative k-core.
    while True:
        u_counts = Counter(u for u, _ in filtered)
        i_counts = Counter(i for _, i in filtered)
        new_filtered = [
            (u, i) for u, i in filtered if u_counts[u] >= min_edges and i_counts[i] >= min_edges
        ]
        if len(new_filtered) == len(filtered):
            break
        filtered = new_filtered
    if not filtered:
        raise DataError("k-core filtering produced empty interaction set")

    user2id = {u: idx for idx, u in enumerate(sorted({u for u, _ in filtered}))}
    item2id = {i: idx for idx, i in enumerate(sorted({i for _, i in filtered}))}
    user_arr = np.fromiter((user2id[u] for u, _ in filtered), dtype=np.int64, count=len(filtered))
    item_arr = np.fromiter((item2id[i] for _, i in filtered), dtype=np.int64, count=len(filtered))
    graph_matrix = bipartite(user_arr, item_arr, len(user2id), len(item2id))

    item_meta: dict[int, dict[str, Any]] = {}
    skipped_meta = 0
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped_meta += 1
                    continue
                asin = record.get("asin")
                if asin in item2id:
                    item_meta[item2id[asin]] = record
    if skipped_meta:
        log.warning("skipped malformed metadata lines", extra={"count": skipped_meta})
    return graph_matrix, item_meta, len(user2id), len(item2id)


__all__ = ["bipartite", "interactions", "item_cooccurrence", "kcore"]
