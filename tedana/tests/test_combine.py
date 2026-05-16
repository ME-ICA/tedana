"""Tests for tedana.combine."""

import numpy as np
import pytest

from tedana import combine


def test__combine_r2s():
    """Test tedana.combine._combine_r2s."""
    n_voxels, n_trs, n_echos = 100, 20, 5
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[0.014, 0.028, 0.042, 0.056, 0.070]])  # 1 x E seconds

    r2s = np.random.random((n_voxels, n_trs, 1)) * 50  # Mb x T x 1, R2* in s⁻¹
    comb = combine._combine_r2s(data, tes, r2s)
    assert comb.shape == (n_voxels, n_trs)

    r2s = np.random.random((n_voxels, 1)) * 50  # M x 1
    comb = combine._combine_r2s(data, tes, r2s)
    assert comb.shape == (n_voxels, n_trs)


def test__combine_paid():
    """Test tedana.combine._combine_paid."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[0.010, 0.020, 0.030]])  # 1 x E
    comb = combine._combine_paid(data, tes)
    assert comb.shape == (n_voxels, n_trs)

    comb2 = combine._combine_paid(data, tes, report=False)
    assert np.array_equal(comb, comb2)


def test_make_optcom():
    """Test tedana.combine.make_optcom."""
    n_voxels, n_trs, n_echos = 100, 20, 5
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([0.014, 0.028, 0.042, 0.056, 0.070])  # seconds
    adaptive_mask = np.ones(n_voxels, dtype=int) * n_echos

    r2s = np.random.random((n_voxels, n_trs)) * 50
    comb = combine.make_optcom(data, tes, adaptive_mask, r2s=r2s, combmode="t2s")
    assert comb.shape == (n_voxels, n_trs)

    r2s = np.random.random(n_voxels) * 50
    comb = combine.make_optcom(data, tes, adaptive_mask, r2s=r2s, combmode="t2s")
    assert comb.shape == (n_voxels, n_trs)

    comb = combine.make_optcom(data, tes, adaptive_mask, r2s=r2s, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    comb = combine.make_optcom(data, tes, adaptive_mask, r2s=None, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    bad_data = np.random.random((n_voxels, n_echos))
    with pytest.raises(ValueError, match="Input data must be 3D"):
        combine.make_optcom(bad_data, tes, adaptive_mask, r2s=r2s, combmode="t2s")

    bad_tes = np.array([0.014, 0.028])
    with pytest.raises(ValueError, match="Number of echos provided does not match"):
        combine.make_optcom(data, bad_tes, adaptive_mask, r2s=r2s, combmode="t2s")

    bad_adaptive_mask = np.ones((n_voxels, 2), dtype=int)
    with pytest.raises(ValueError, match="Mask is not 1D"):
        combine.make_optcom(data, tes, bad_adaptive_mask, r2s=r2s, combmode="t2s")

    bad_adaptive_mask2 = np.ones(n_voxels - 1, dtype=int)
    with pytest.raises(ValueError, match="Mask and data do not have same number"):
        combine.make_optcom(data, tes, bad_adaptive_mask2, r2s=r2s, combmode="t2s")

    with pytest.raises(ValueError, match="Argument 'combmode' must be either 't2s' or 'paid'"):
        combine.make_optcom(data, tes, adaptive_mask, r2s=r2s, combmode="bad")

    with pytest.raises(
        ValueError, match="Argument 'r2s' must be supplied if 'combmode' is set to 't2s'"
    ):
        combine.make_optcom(data, tes, adaptive_mask, r2s=None, combmode="t2s")

    bad_r2s = np.random.random(n_voxels - 1) * 50
    with pytest.raises(ValueError, match="R2\\* estimates and data do not have same number"):
        combine.make_optcom(data, tes, adaptive_mask, r2s=bad_r2s, combmode="t2s")
