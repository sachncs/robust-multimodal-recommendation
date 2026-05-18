import numpy as np
import scipy.sparse as sp
import torch

from rmr.models.downstream import LightGCN


def test_lightgcn_forward():
    # User-item bipartite graph: 2 users, 3 items
    ui = sp.csr_matrix(np.array([
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=np.float32))
    model = LightGCN(num_users=2, num_items=3, embedding_dim=8, num_layers=2)
    scores = model(torch.tensor([0, 1]), torch.tensor([0, 1, 2]), ui)
    assert scores.shape == (2, 3)


def test_lightgcn_zero_degree_nodes():
    # 3 users, 3 items; user 2 and item 2 have no interactions
    ui = sp.csr_matrix(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.float32))
    model = LightGCN(num_users=3, num_items=3, embedding_dim=4, num_layers=1)
    # Should not crash even with zero-degree nodes
    scores = model(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), ui)
    assert scores.shape == (3, 3)
    assert torch.isfinite(scores).all()


def test_embedding_initialization():
    model = LightGCN(num_users=5, num_items=4, embedding_dim=8, num_layers=2)
    # Xavier uniform initialization should keep weights within a reasonable range
    u_w = model.user_emb.weight.detach().numpy()
    i_w = model.item_emb.weight.detach().numpy()
    # Check no NaNs
    assert np.isfinite(u_w).all()
    assert np.isfinite(i_w).all()
    # Xavier uniform gain for tanh is sqrt(6/(fan_in+fan_out)), for uniform it's sqrt(6)/sqrt(fan_in+fan_out)
    # For Embedding, nn.init.xavier_uniform_ uses gain=1 which is sqrt(6/(fan_in+fan_out))
    # with fan_in=embedding_dim, fan_out=1 for embedding. Actually it's computed differently.
    # We'll just check that std is reasonable and not too large.
    assert u_w.std() < 1.0
    assert i_w.std() < 1.0
    # Check that different rows are not all identical
    assert not np.allclose(u_w[0], u_w[1], atol=1e-4)
    assert not np.allclose(i_w[0], i_w[1], atol=1e-4)
