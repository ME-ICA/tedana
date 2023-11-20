"""Tests for tedana.combine."""

import numpy as np

from tedana import combine


def test__combine_t2s():
    """Test tedana.combine._combine_t2s."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[10, 20, 30]])  # 1 x E

    # Voxel- and volume-wise T2* estimates
    t2s = np.random.random((n_voxels, n_trs, 1))  # M x T x 1
    comb = combine._combine_t2s(data, tes, t2s)
    assert comb.shape == (n_voxels, n_trs)

    # Voxel-wise T2* estimates
    t2s = np.random.random((n_voxels, 1))  # M x 1
    comb = combine._combine_t2s(data, tes, t2s)
    assert comb.shape == (n_voxels, n_trs)


def test__combine_paid():
    """Test tedana.combine._combine_paid."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[10, 20, 30]])  # 1 x E
    comb = combine._combine_paid(data, tes)
    assert comb.shape == (n_voxels, n_trs)


def test_make_optcom():
    """Test tedana.combine.make_optcom."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    n_mask = 5
    data = np.random.random((n_voxels, n_echos, n_trs))
    mask = np.zeros(n_voxels).astype(bool)
    mask[:n_mask] = True
    tes = np.array([10, 20, 30])  # E

    # Voxel- and volume-wise T2* estimates
    t2s = np.random.random((n_voxels, n_trs))
    comb = combine.make_optcom(data, tes, mask, t2s=t2s, combmode="t2s")
    assert comb.shape == (n_voxels, n_trs)

    # Voxel-wise T2* estimates
    t2s = np.random.random(n_voxels)
    comb = combine.make_optcom(data, tes, mask, t2s=t2s, combmode="t2s")
    assert comb.shape == (n_voxels, n_trs)

    # STE with erroneously included T2* argument
    comb = combine.make_optcom(data, tes, mask, t2s=t2s, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    # Normal STE call
    comb = combine.make_optcom(data, tes, mask, t2s=None, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)
