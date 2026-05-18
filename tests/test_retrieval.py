import numpy as np
import scipy.sparse as sp

from rmr.models.retrieval import (
    acs,
    anchor_retrieval,
    bfs_multi_source,
    mage,
    relevance_score,
    shortest_path_nodes,
)


def test_acs_basic():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    anchors = [0, 3]
    S = acs(adj, anchors)
    assert 0 in S and 3 in S
    # Path 0-1-2-3 should be included
    assert len(S) >= 4


def test_mage_returns_anchors():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    features = {
        "visual": np.eye(4, dtype=np.float32),
    }
    mask = np.ones((4, 1), dtype=np.float32)
    subgraph = mage(adj, [0, 3], 0, features, mask, max_iters=2)
    assert 0 in subgraph and 3 in subgraph


def test_bfs_multi_source_exact_distances():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    dist = bfs_multi_source(adj, [0])
    assert dist[0] == 0
    assert dist[1] == 1
    assert dist[2] == 2
    assert dist[3] == 3

    dist2 = bfs_multi_source(adj, [0, 3])
    assert dist2[0] == 0
    assert dist2[3] == 0
    assert dist2[1] == 1  # nearest to 0
    assert dist2[2] == 1  # nearest to 3


def test_shortest_path_nodes_start_eq_end():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32))
    path = shortest_path_nodes(adj, 1, 1)
    assert path == [1]


def test_shortest_path_nodes_no_path():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=np.float32))
    path = shortest_path_nodes(adj, 0, 2)
    assert path == []


def test_relevance_score_exact_value():
    features = {
        "visual": np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float32),
    }
    mask = np.ones((3, 1), dtype=np.float32)
    # cos([1,0], [0,1]) == 0
    assert relevance_score(0, 1, features, mask) == 0.0
    # cos([1,0], [1,1]) == 1/sqrt(2)
    assert abs(relevance_score(0, 2, features, mask) - 1.0 / np.sqrt(2)) < 1e-5
    # cos([1,1], [1,1]) == 1
    assert np.isclose(relevance_score(2, 2, features, mask), 1.0, atol=1e-5)


def test_acs_empty_anchors():
    adj = sp.csr_matrix(np.array([
        [0, 1],
        [1, 0],
    ], dtype=np.float32))
    S = acs(adj, [])
    assert set() == S


def test_acs_single_anchor():
    adj = sp.csr_matrix(np.array([
        [0, 1],
        [1, 0],
    ], dtype=np.float32))
    S = acs(adj, [1])
    assert {1} == S


def test_acs_disconnected_graph():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    anchors = [0, 3]
    S = acs(adj, anchors)
    # In a disconnected graph, not all anchors are reachable from each other
    # The code returns set(anchors) when len(dist) < len(anchors)
    assert set(anchors) == S


def test_mage_t_zero():
    adj = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32))
    features = {
        "visual": np.eye(4, dtype=np.float32),
    }
    mask = np.ones((4, 1), dtype=np.float32)
    subgraph = mage(adj, [0, 3], 0, features, mask, max_iters=0)
    # With max_iters=0, the loop doesn't run, so subgraph should be the ACS
    expected = acs(adj, [0, 3])
    assert expected == subgraph


def test_anchor_retrieval_basic():
    features = {
        "visual": np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ], dtype=np.float32),
    }
    mask = np.ones((4, 1), dtype=np.float32)
    anchors = anchor_retrieval(0, "visual", features, mask, top_k=2)
    # Node 2 is identical to query, node 3 has positive similarity, node 1 orthogonal
    assert len(anchors) == 2
    assert anchors[0] == 2
    assert anchors[1] == 3


def test_anchor_retrieval_excludes_query_and_missing():
    features = {
        "text": np.eye(5, dtype=np.float32),
    }
    mask = np.ones((5, 1), dtype=np.float32)
    mask[2, 0] = 0.0  # node 2 missing text
    anchors = anchor_retrieval(0, "text", features, mask, top_k=10)
    # Should return all observed except query (0), and not include missing node 2
    assert 0 not in anchors
    assert 2 not in anchors
    assert set(anchors) == {1, 3, 4}


def test_acs_collision_root_on_star():
    # Star graph: center 0 connected to leaves 1,2,3,4
    adj = sp.csr_matrix(np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.float32))
    anchors = [1, 2, 3, 4]
    S = acs(adj, anchors)
    # Collision root should be node 0 (center), paths are 0-1, 0-2, 0-3, 0-4
    assert {0, 1, 2, 3, 4} == S
