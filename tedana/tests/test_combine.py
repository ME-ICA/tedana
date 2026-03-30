"""Tests for tedana.combine."""

import numpy as np
import pytest

from tedana import combine


def test__combine_t2s():
    """Test tedana.combine._combine_t2s."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[10, 20, 30]])  # 1 x E

    # Voxel- and volume-wise T2* estimates
    t2s = np.random.random((n_voxels, n_trs, 1))  # Mb x T x 1
    comb = combine._combine_t2s(data, tes, t2s)
    assert comb.shape == (n_voxels, n_trs)

    # Voxel-wise T2* estimates
    t2s = np.random.random((n_voxels, 1))  # M x 1
    comb = combine._combine_t2s(data, tes, t2s)
    assert comb.shape == (n_voxels, n_trs)

    # Extra dimension for t2s
    bad_t2s = np.random.random((n_voxels))
    with pytest.raises(ValueError, match="Invalid shape for alpha"):
        combine._combine_t2s(data, tes, bad_t2s)


def test__combine_paid():
    """Test tedana.combine._combine_paid."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[10, 20, 30]])  # 1 x E
    comb = combine._combine_paid(data, tes)
    assert comb.shape == (n_voxels, n_trs)

    comb2 = combine._combine_paid(data, tes, report=False)
    assert np.array_equal(comb, comb2)


def test_make_optcom():
    """Test tedana.combine.make_optcom."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    n_mask = 5
    data = np.random.random((n_voxels, n_echos, n_trs))
    adaptive_mask = np.zeros(n_voxels, dtype=int)
    adaptive_mask[:n_mask] = 1
    adaptive_mask[0] = 2
    tes = np.array([10, 20, 30])  # E

    # Voxel- and volume-wise T2* estimates
    t2s = np.random.random((n_voxels, n_trs))
    comb = combine.make_optcom(data, tes, adaptive_mask, t2s=t2s, combmode="t2s")
    assert comb.shape == (n_voxels, n_trs)

    # Voxel-wise T2* estimates
    t2s = np.random.random(n_voxels)
    comb = combine.make_optcom(data, tes, adaptive_mask, t2s=t2s, combmode="t2s")
    assert comb.shape == (n_voxels, n_trs)

    # STE with erroneously included T2* argument
    comb = combine.make_optcom(data, tes, adaptive_mask, t2s=t2s, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    # Normal STE call
    comb = combine.make_optcom(data, tes, adaptive_mask, t2s=None, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    # Test with invalid shapes
    bad_data = np.random.random((n_voxels, n_echos))
    with pytest.raises(ValueError, match="Input data must be 3D"):
        combine.make_optcom(bad_data, tes, adaptive_mask, t2s=t2s, combmode="t2s")

    bad_tes = np.array([10, 20])
    with pytest.raises(ValueError, match="Number of echos provided does not match second"):
        combine.make_optcom(data, bad_tes, adaptive_mask, t2s=t2s, combmode="t2s")

    bad_adaptive_mask = np.ones((n_voxels, 2), dtype=int)
    with pytest.raises(ValueError, match="Mask is not 1D"):
        combine.make_optcom(data, tes, bad_adaptive_mask, t2s=t2s, combmode="t2s")

    bad_adaptive_mask2 = np.ones(n_voxels - 1, dtype=int)
    with pytest.raises(ValueError, match="Mask and data do not have same number of voxels"):
        combine.make_optcom(data, tes, bad_adaptive_mask2, t2s=t2s, combmode="t2s")

    with pytest.raises(ValueError, match="Argument 'combmode' must be either 't2s' or 'paid'"):
        combine.make_optcom(data, tes, adaptive_mask, t2s=t2s, combmode="bad")

    with pytest.raises(
        ValueError,
        match="Argument 't2s' must be supplied if 'combmode' is set to 't2s'",
    ):
        combine.make_optcom(data, tes, adaptive_mask, t2s=None, combmode="t2s")

    bad_t2s = np.random.random((n_voxels - 1))
    with pytest.raises(
        ValueError,
        match="estimates and data do not have same number of voxels",
    ):
        combine.make_optcom(data, tes, adaptive_mask, t2s=bad_t2s, combmode="t2s")
