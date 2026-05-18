import numpy as np
import scipy.sparse as sp

from rmr.data.graph_builder import build_item_graph, build_user_item_graph


def test_build_item_graph():
    # 4 users, 3 items
    ui = sp.csr_matrix(np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32))
    ii = build_item_graph(ui)
    assert ii.shape == (3, 3)
    # Items 0 and 1 share user 2
    assert ii[0, 1] > 0
    # Items 0 and 2 share user 0
    assert ii[0, 2] > 0


def test_build_user_item_graph():
    user_ids = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    item_ids = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    n_users = 3
    n_items = 3
    ui = build_user_item_graph(user_ids, item_ids, n_users, n_items)
    assert ui.shape == (n_users, n_items)
    assert ui.dtype == np.float32
    # Check specific entries
    assert ui[0, 0] == 1.0
    assert ui[0, 1] == 1.0
    assert ui[1, 2] == 1.0
    assert ui[2, 0] == 1.0
    assert ui[2, 1] == 1.0
    assert ui[2, 2] == 1.0
    assert ui[1, 0] == 0.0


def test_build_item_graph_no_shared_users():
    # 3 users, 3 items — each user interacts with exactly one distinct item
    ui = sp.csr_matrix(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32))
    ii = build_item_graph(ui)
    assert ii.shape == (3, 3)
    # No shared users means all zeros (self-loops removed)
    assert ii.nnz == 0
    np.testing.assert_array_equal(ii.toarray(), np.zeros((3, 3), dtype=np.float32))


def test_build_item_graph_self_loop_removal():
    # 2 users, 2 items — both users interact with both items
    ui = sp.csr_matrix(np.array([
        [1, 1],
        [1, 1],
    ], dtype=np.float32))
    ii = build_item_graph(ui)
    assert ii.shape == (2, 2)
    # There should be an edge between item 0 and 1
    assert ii[0, 1] > 0
    assert ii[1, 0] > 0
    # Self-loops should be removed
    assert ii[0, 0] == 0.0
    assert ii[1, 1] == 0.0
