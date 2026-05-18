"""Graph construction utilities for user-item and item-item adjacency."""

import json
from collections import Counter

import numpy as np
import scipy.sparse as sp


def build_user_item_graph(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    n_users: int,
    n_items: int,
) -> sp.csr_matrix:
    """Build a sparse user-item bipartite adjacency matrix.

    Args:
        user_ids: 1-D array of user indices.
        item_ids: 1-D array of item indices aligned with ``user_ids``.
        n_users: Total number of users.
        n_items: Total number of items.

    Returns:
        A sparse matrix of shape (n_users, n_items) with float32 entries.
    """
    ratings = np.ones(len(user_ids), dtype=np.float32)
    ui = sp.csr_matrix(
        (ratings, (user_ids, item_ids)), shape=(n_users, n_items)
    )
    return ui


def build_item_graph(ui_graph: sp.csr_matrix) -> sp.csr_matrix:
    """Build an item-item graph where edges connect items sharing a user.

    The graph is binarized (no edge weights) and self-loops are removed.

    Args:
        ui_graph: Sparse user-item bipartite matrix.

    Returns:
        A sparse item-item co-occurrence matrix.
    """
    ii = ui_graph.T @ ui_graph
    ii = ii.sign()
    ii.setdiag(0)
    ii.eliminate_zeros()
    return ii


def load_interactions_from_json(
    review_path: str,
    meta_path: str,
    min_interactions: int = 5,
) -> tuple[np.ndarray, np.ndarray, int, int, dict]:
    """Load user-item interactions and item metadata from Amazon JSON.

    Applies k-core filtering so that every remaining user and item has at
    least ``min_interactions`` records.

    Args:
        review_path: Path to the Amazon review ``.json`` file.
        meta_path: Path to the Amazon metadata ``.json`` file.
        min_interactions: Minimum number of interactions for k-core filtering.

    Returns:
        A 5-tuple of ``(user_ids, item_ids, n_users, n_items, item_meta)``.
        ``user_ids`` and ``item_ids`` are aligned 1-D arrays of contiguous
        integer IDs. ``item_meta`` maps item IDs to metadata dictionaries.
    """
    user_counts = Counter()
    item_counts = Counter()
    user_item_pairs = []
    item_meta = {}
    with open(review_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user = obj["reviewerID"]
            item = obj["asin"]
            user_item_pairs.append((user, item))
            user_counts[user] += 1
            item_counts[item] += 1
    valid_users = {u for u, c in user_counts.items() if c >= min_interactions}
    valid_items = {i for i, c in item_counts.items() if c >= min_interactions}
    filtered_pairs = []
    for u, i in user_item_pairs:
        if u in valid_users and i in valid_items:
            filtered_pairs.append((u, i))
    user2id = {u: idx for idx, u in enumerate(sorted(valid_users))}
    item2id = {i: idx for idx, i in enumerate(sorted(valid_items))}
    uids = np.array(
        [user2id[u] for u, i in filtered_pairs], dtype=np.int64
    )
    iids = np.array(
        [item2id[i] for u, i in filtered_pairs], dtype=np.int64
    )
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            asin = obj.get("asin")
            if asin in item2id:
                item_meta[item2id[asin]] = obj
    return uids, iids, len(user2id), len(item2id), item_meta
