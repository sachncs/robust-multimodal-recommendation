import numpy as np
import scipy.sparse as sp

from rmr.data.dataset import GREMCGraphDataset


def test_dataset_len_and_item():
    features = {
        "visual": np.random.randn(5, 10).astype(np.float32),
        "text": np.random.randn(5, 8).astype(np.float32),
    }
    E = np.ones((5, 2), dtype=np.float32)
    adj = sp.csr_matrix(np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.float32))
    ds = GREMCGraphDataset(features, E, adj)
    assert len(ds) == 5
    item = ds[0]
    assert "features" in item
    assert "mask" in item
    assert item["features"]["visual"].shape == (10,)
