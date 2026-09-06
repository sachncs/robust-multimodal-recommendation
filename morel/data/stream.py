"""Streaming ingestion primitives.

Two-pass exact k-core via stream: pass 1 collects degree counts, pass 2
emits the filtered edges. Online degree-filter uses a rolling window for
the IterableDataset training path.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Iterator

import numpy as np
import scipy.sparse as sp

from morel.core.errors import DataError
from morel.data.build import bipartite as build_bipartite


def review_stream(path: Path | str, *, chunk_size: int = 100_000) -> Iterator[list[dict]]:
    """Yield chunks of JSON-decoded records from an Amazon review file.

    Args
    ----
    path : Path | str
        Path to the decompressed ``.json`` file (one JSON object per line).
    chunk_size : int
        Maximum number of records per chunk.

    Yields
    ------
    list[dict]
        A list of at most ``chunk_size`` parsed records.
    """
    target = Path(path)
    if not target.exists():
        raise DataError(f"review file not found: {target}")
    chunk: list[dict] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def streaming_interactions(
    review_path: Path | str,
    *,
    min_edges: int = 5,
    chunk_size: int = 100_000,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(user_ids_chunk, item_ids_chunk)`` for an Amazon review file.

    Args
    ----
    review_path : Path | str
        Path to the decompressed ``.json`` file.
    min_edges : int
        Nodes whose rolling-window degree is below this are filtered out
        (online degree filter).
    chunk_size : int
        Records per streaming chunk.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        ``(user_ids, item_ids)`` for each chunk after filtering.
    """
    rolling_window: deque[tuple[str, str]] = deque(maxlen=chunk_size * 4)
    degree_counter: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()

    for chunk in review_stream(review_path, chunk_size=chunk_size):
        for record in chunk:
            user = record.get("reviewerID")
            item = record.get("asin")
            if user is None or item is None:
                continue
            key = (user, item)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            # Increment first, then check the threshold; this matches the
            # offline k-core filter semantics.
            degree_counter[user] += 1
            degree_counter[item] += 1
            if degree_counter[user] >= min_edges and degree_counter[item] >= min_edges:
                rolling_window.append(key)
        if rolling_window:
            users = np.array([u for u, _ in rolling_window])
            items = np.array([i for _, i in rolling_window])
            yield users, items


def exact_two_pass_interactions(
    review_path: Path | str,
    *,
    min_edges: int = 5,
    chunk_size: int = 100_000,
) -> tuple[sp.csr_matrix, dict[str, int], dict[str, int]]:
    """Two-pass exact k-core streaming loader.

    Pass 1 accumulates degree counts; pass 2 emits only edges whose
    endpoints satisfy the degree threshold. Output is bit-identical to
    the in-memory :func:`morel.data.build.interactions` on the same input.

    Args
    ----
    review_path : Path | str
        Path to the decompressed ``.json`` file.
    min_edges : int
        Minimum degree per node.
    chunk_size : int
        Records per streaming chunk.

    Returns
    -------
    tuple[sp.csr_matrix, dict[str, int], dict[str, int]]
        ``(ui_graph, user_id_to_idx, item_id_to_idx)``.
    """
    review_path = Path(review_path)
    if not review_path.exists():
        raise DataError(f"review file not found: {review_path}")

    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    for chunk in review_stream(review_path, chunk_size=chunk_size):
        for record in chunk:
            user = record.get("reviewerID")
            item = record.get("asin")
            if user is None or item is None:
                continue
            user_counts[user] += 1
            item_counts[item] += 1

    keep_users = {u for u, c in user_counts.items() if c >= min_edges}
    keep_items = {i for i, c in item_counts.items() if c >= min_edges}

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in review_stream(review_path, chunk_size=chunk_size):
        for record in chunk:
            user = record.get("reviewerID")
            item = record.get("asin")
            if user is None or item is None:
                continue
            if user not in keep_users or item not in keep_items:
                continue
            key = (user, item)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)

    if not pairs:
        raise DataError("k-core filtering produced empty interaction set")

    users = sorted({u for u, _ in pairs})
    items = sorted({i for _, i in pairs})
    user2id = {u: idx for idx, u in enumerate(users)}
    item2id = {i: idx for idx, i in enumerate(items)}
    u_arr = np.array([user2id[u] for u, _ in pairs], dtype=np.int64)
    i_arr = np.array([item2id[i] for _, i in pairs], dtype=np.int64)
    ui = build_bipartite(u_arr, i_arr, len(users), len(items))
    return ui, user2id, item2id


def streaming_item_cooccurrence(
    ui_chunks: Iterator[tuple[np.ndarray, np.ndarray]],
    *,
    items: int,
) -> sp.csr_matrix:
    """Build an item cooccurrence graph from a stream of bipartite chunks.

    Args
    ----
    ui_chunks : Iterator
        Stream of ``(user_ids, item_ids)`` pairs.
    items : int
        Number of items.

    Returns
    -------
    sp.csr_matrix
        Symmetric ``(items, items)`` item cooccurrence adjacency.
    """
    from itertools import combinations

    cooc: dict[tuple[int, int], int] = {}
    for user_ids, item_ids in ui_chunks:
        per_user_items = sorted({int(i) for i in item_ids})
        for a, b in combinations(per_user_items, 2):
            cooc[(a, b)] = cooc.get((a, b), 0) + 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for (a, b), c in cooc.items():
        rows.append(a)
        cols.append(b)
        data.append(c)
        rows.append(b)
        cols.append(a)
        data.append(c)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(items, items))
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix.tocsr()


__all__ = [
    "review_stream",
    "streaming_interactions",
    "exact_two_pass_interactions",
    "streaming_item_cooccurrence",
]
