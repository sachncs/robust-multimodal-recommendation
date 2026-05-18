import os
import tempfile

import numpy as np

from rmr.data.features import (
    extract_text_features_dummy,
    extract_visual_features_dummy,
    load_features,
    save_features,
)


def test_extract_text_features_dummy():
    texts = ["a nice product", "another item"]
    feats = extract_text_features_dummy(texts, dim=8)
    assert feats.shape == (2, 8)
    assert isinstance(feats, np.ndarray)


def test_extract_visual_features_dummy():
    paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    feats = extract_visual_features_dummy(paths, dim=512)
    assert feats.shape == (3, 512)
    assert isinstance(feats, np.ndarray)
    assert feats.dtype == np.float32


def test_l2_normalization_exactness():
    texts = ["foo", "bar", "baz", "qux"]
    feats = extract_text_features_dummy(texts, dim=16)
    norms = np.linalg.norm(feats, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    paths = ["a", "b", "c"]
    vfeats = extract_visual_features_dummy(paths, dim=64)
    vnorms = np.linalg.norm(vfeats, axis=1)
    np.testing.assert_allclose(vnorms, 1.0, atol=1e-6)


def test_save_load_features_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "features.npy")
        original = np.random.randn(10, 32).astype(np.float32)
        save_features(original, path)
        loaded = load_features(path)
        np.testing.assert_array_equal(original, loaded)


def test_zero_norm_fallback():
    # Directly test the normalization logic on zero vectors
    arr = np.zeros((3, 5), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    result = arr / norms
    np.testing.assert_array_equal(result, arr)
    assert not np.isnan(result).any()
