"""Graph retrieval functions for anchor connecting subgraphs and expansion.

Implements Algorithm 1 (ACS) and Algorithm 2 (MAGE) from the GRE-MC paper,
along with brute-force cosine-based anchor retrieval.
"""

from collections import deque

import numpy as np
import scipy.sparse as sp


def bfs_multi_source(
    adj: sp.csr_matrix, sources: list[int]
) -> dict[int, int]:
    """Run multi-source BFS to compute shortest distances.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        sources: List of source node indices to start BFS from.

    Returns:
        Dictionary mapping visited node indices to their shortest distance
        from the nearest source.
    """
    dist = dict.fromkeys(sources, 0)
    q = deque(sources)
    while q:
        u = q.popleft()
        row_start = adj.indptr[u]
        row_end = adj.indptr[u + 1]
        for v in adj.indices[row_start:row_end]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def shortest_path_nodes(
    adj: sp.csr_matrix, start: int, end: int
) -> list[int]:
    """Return the list of nodes on a shortest path from start to end.

    Uses BFS to find a shortest path. If start equals end, returns a
    single-element list containing start.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        start: Starting node index.
        end: Target node index.

    Returns:
        List of node indices representing a shortest path from start to
        end. Returns an empty list if no path exists.
    """
    if start == end:
        return [start]
    dist = {start: 0}
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        row_start = adj.indptr[u]
        row_end = adj.indptr[u + 1]
        for v in adj.indices[row_start:row_end]:
            if v not in dist:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
                if v == end:
                    path = [v]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
    return []


def acs(adj: sp.csr_matrix, anchors: list[int]) -> set[int]:
    """Compute the Anchor Connecting Subgraph (ACS).

    Implements Algorithm 1 from the paper using multi-source BFS with
    reachability bitmasks. All anchors start simultaneously; as the search
    proceeds each visited node stores an OR-bitmask of which anchors have
    reached it.  When a node has all anchor bits set it becomes the
    *collision root*.  The ACS is the union of shortest paths from the
    collision root back to each anchor.

    If anchors are not mutually reachable, falls back to returning the
    anchor set itself.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        anchors: List of anchor node indices.

    Returns:
        Set of node indices in the anchor connecting subgraph.
    """
    if not anchors:
        return set()
    if len(anchors) == 1:
        return {anchors[0]}

    k = len(anchors)
    all_mask = (1 << k) - 1
    q: deque = deque()
    # bitmask[node] = OR of anchor bits that have reached node
    bitmask: dict[int, int] = {}
    # prev[node][anchor_idx] = predecessor of node on a shortest path
    # from anchor anchor_idx to node.
    prev: dict[int, dict[int, int]] = {}

    for idx, a in enumerate(anchors):
        q.append(a)
        bitmask[a] = 1 << idx
        prev[a] = {}

    collision_root: int | None = None
    while q and collision_root is None:
        u = q.popleft()
        bits_u = bitmask[u]
        row_start = adj.indptr[u]
        row_end = adj.indptr[u + 1]
        for v in adj.indices[row_start:row_end]:
            if v not in bitmask:
                bitmask[v] = 0
                prev[v] = {}
            new_bits = bits_u & ~bitmask[v]
            if new_bits:
                for idx in range(k):
                    if (new_bits >> idx) & 1:
                        prev[v][idx] = u
                bitmask[v] |= new_bits
                q.append(v)
                if bitmask[v] == all_mask:
                    collision_root = v
                    break

    if collision_root is None:
        return set(anchors)

    # Backtrack shortest paths from collision_root to each anchor
    subgraph: set[int] = set()

    def _backtrack(node: int, anchor_idx: int) -> None:
        subgraph.add(node)
        if node == anchors[anchor_idx]:
            return
        nxt = prev.get(node, {}).get(anchor_idx)
        if nxt is not None:
            _backtrack(nxt, anchor_idx)

    for idx in range(k):
        _backtrack(collision_root, idx)

    return subgraph


def relevance_score(
    i: int,
    v: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    """Compute the relevance score r(i, v) between two nodes.

    Uses average *cosine similarity* over all jointly observed modalities,
    matching the paper definition.

    Args:
        i: Index of the query node.
        v: Index of the candidate node.
        features: Dictionary mapping modality names to feature arrays.
            Each array should be indexable by node index.
        mask: Binary mask of shape (num_nodes, num_modalities)
            indicating which modalities are available for each node.

    Returns:
        The relevance score as a float. Returns 0.0 if no shared
        modalities have positive masks for both nodes.
    """
    mods = list(features.keys())
    num = 0.0
    den = 0.0
    for m_idx, m in enumerate(mods):
        e_i = mask[i, m_idx]
        e_v = mask[v, m_idx]
        if e_i > 0 and e_v > 0:
            vec_i = features[m][i]
            vec_v = features[m][v]
            norm_i = np.linalg.norm(vec_i)
            norm_v = np.linalg.norm(vec_v)
            if norm_i > 0.0 and norm_v > 0.0:
                num += np.dot(vec_i, vec_v) / (norm_i * norm_v)
            den += 1.0
    return num / den if den > 0 else 0.0


def mean_relevance(
    i: int,
    node_set: set[int],
    features: dict[str, np.ndarray],
    mask: np.ndarray,
) -> float:
    """Compute the mean relevance of node i over a set of nodes.

    Args:
        i: Index of the query node.
        node_set: Set of node indices over which to average relevance.
        features: Dictionary mapping modality names to feature arrays.
        mask: Binary mask indicating available modalities per node.

    Returns:
        Mean relevance score. Returns 0.0 if node_set is empty or only
        contains i.
    """
    vals = [
        relevance_score(i, v, features, mask)
        for v in node_set
        if v != i
    ]
    return float(np.mean(vals)) if vals else 0.0


def mage(
    adj: sp.csr_matrix,
    anchors: list[int],
    query_item: int,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    max_iters: int = 10,
) -> set[int]:
    """Run Modality-Aware Graph Expansion (MAGE).

    Implements Algorithm 2 from the paper. Greedily adds or removes
    boundary nodes to maximize mean relevance while preserving
    connectivity and all anchors.

    Args:
        adj: Sparse adjacency matrix in CSR format.
        anchors: List of anchor node indices that must be preserved.
        query_item: Index of the query item node.
        features: Dictionary mapping modality names to feature arrays.
        mask: Binary mask indicating available modalities per node.
        max_iters: Maximum number of greedy iterations.

    Returns:
        Set of node indices in the expanded subgraph.
    """
    subgraph = acs(adj, anchors)
    if not subgraph:
        subgraph = set(anchors)
    subgraph = set(subgraph)

    neighbors = {}
    for u in range(adj.shape[0]):
        row_start = adj.indptr[u]
        row_end = adj.indptr[u + 1]
        neighbors[u] = set(adj.indices[row_start:row_end])

    def boundary(node_set: set[int]) -> set[int]:
        b = set()
        for u in node_set:
            b.update(neighbors[u] - node_set)
        return b

    def is_connected(subset: set[int]) -> bool:
        if not subset:
            return True
        start = next(iter(subset))
        visited = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in neighbors[u] & subset:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return visited == subset

    best_score = mean_relevance(query_item, subgraph, features, mask)
    for _ in range(max_iters):
        changed = False
        boundary_nodes = boundary(subgraph)
        for v in boundary_nodes:
            new_subgraph = subgraph | {v}
            if not is_connected(new_subgraph):
                continue
            score = mean_relevance(query_item, new_subgraph, features, mask)
            if score > best_score:
                subgraph = new_subgraph
                best_score = score
                changed = True
                break
        if changed:
            continue
        for v in list(subgraph):
            if v in anchors:
                continue
            new_subgraph = subgraph - {v}
            if not is_connected(new_subgraph):
                continue
            score = mean_relevance(query_item, new_subgraph, features, mask)
            if score > best_score:
                subgraph = new_subgraph
                best_score = score
                changed = True
                break
        if not changed:
            break
    return subgraph


def anchor_retrieval(
    query_item: int,
    query_modality: str,
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    top_k: int = 10,
) -> list[int]:
    """Retrieve top-K anchor nodes via cosine nearest-neighbor search.

    Searches over all nodes that have the query modality observed and
    returns the top-K most similar ones (excluding the query itself).

    Args:
        query_item: Index of the query item.
        query_modality: Name of the modality used for retrieval.
        features: Dictionary mapping modality names to feature arrays.
        mask: Binary mask of shape (num_nodes, num_modalities).
        top_k: Number of anchors to retrieve.

    Returns:
        List of top-K anchor node indices.
    """
    mods = list(features.keys())
    if query_modality not in mods:
        return []
    mod_idx = mods.index(query_modality)
    observed = np.where(mask[:, mod_idx] > 0)[0]

    query_vec = features[query_modality][query_item]
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0.0:
        return observed[:top_k].tolist()

    similarities = []
    for v in observed:
        if v == query_item:
            continue
        vec = features[query_modality][v]
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            continue
        sim = float(np.dot(query_vec, vec) / (query_norm * norm))
        similarities.append((sim, int(v)))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in similarities[:top_k]]
