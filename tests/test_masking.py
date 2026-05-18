import numpy as np

from rmr.data.masking import apply_modality_mask


def test_masking_shape_and_minimum():
    n_items = 10
    n_modalities = 3
    E = apply_modality_mask(n_items, n_modalities, mask_ratio=0.4, seed=42)
    assert E.shape == (n_items, n_modalities)
    # Every item must retain at least one modality
    assert np.all(E.sum(axis=1) >= 1)
    # Roughly 40% masked
    masked = 1 - E
    assert 0.2 <= masked.mean() <= 0.6


def test_mask_ratio_zero():
    n_items = 20
    n_modalities = 4
    E = apply_modality_mask(n_items, n_modalities, mask_ratio=0.0, seed=123)
    assert E.shape == (n_items, n_modalities)
    # Nothing masked
    np.testing.assert_array_equal(E, np.ones((n_items, n_modalities), dtype=np.float32))


def test_mask_ratio_one():
    n_items = 20
    n_modalities = 4
    E = apply_modality_mask(n_items, n_modalities, mask_ratio=1.0, seed=123)
    assert E.shape == (n_items, n_modalities)
    # Must still retain at least one per item
    assert np.all(E.sum(axis=1) >= 1)
    # With mask_ratio=1.0 and n_modalities=4, only exactly 1 should be kept per item
    np.testing.assert_array_equal(E.sum(axis=1), np.ones(n_items, dtype=np.float32))


def test_n_modalities_one():
    n_items = 10
    n_modalities = 1
    E = apply_modality_mask(n_items, n_modalities, mask_ratio=0.5, seed=42)
    assert E.shape == (n_items, 1)
    # Must retain at least one per item, but there's only one modality
    np.testing.assert_array_equal(E, np.ones((n_items, 1), dtype=np.float32))


def test_seed_reproducibility():
    n_items = 50
    n_modalities = 3
    E1 = apply_modality_mask(n_items, n_modalities, mask_ratio=0.4, seed=99)
    E2 = apply_modality_mask(n_items, n_modalities, mask_ratio=0.4, seed=99)
    np.testing.assert_array_equal(E1, E2)
