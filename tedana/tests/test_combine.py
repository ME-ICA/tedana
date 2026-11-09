"""Tests for tedana.combine."""

import numpy as np
import pytest

from tedana import combine


def test__combine_t2s():
    """Test tedana.combine._combine_t2s."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[0.010, 0.020, 0.030]])  # 1 x E

    # Voxel- and volume-wise T2* estimates
    t2s = np.random.random((n_voxels, n_trs, 1))  # Mb x T x 1
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
    tes = np.array([[0.010, 0.020, 0.030]])  # 1 x E
    comb = combine._combine_paid(data, tes)
    assert comb.shape == (n_voxels, n_trs)

    comb2 = combine._combine_paid(data, tes, report=False)
    assert np.array_equal(comb, comb2)


def test__combine_paid_sample_mask():
    """PAID weights come from retained volumes only, but apply to every volume."""
    rng = np.random.default_rng(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    data = rng.random((n_voxels, n_echos, n_trs)) + 1
    tes = np.array([[0.010, 0.020, 0.030]])
    sample_mask = np.ones(n_trs, dtype=bool)
    sample_mask[[2, 7]] = False

    # An all-True mask matches no mask (to rounding, since boolean indexing copies the data)
    np.testing.assert_allclose(
        combine._combine_paid(data, tes, sample_mask=np.ones(n_trs, dtype=bool)),
        combine._combine_paid(data, tes),
    )

    comb = combine._combine_paid(data, tes, sample_mask=sample_mask)
    assert comb.shape == (n_voxels, n_trs)

    # Spikes in censored volumes leave retained volumes unchanged
    spiked = data.copy()
    spiked[..., ~sample_mask] = spiked[..., ~sample_mask] * 100 + 1e4
    comb_spiked = combine._combine_paid(spiked, tes, sample_mask=sample_mask)
    np.testing.assert_allclose(comb_spiked[:, sample_mask], comb[:, sample_mask])

    # Without the sample mask, the spikes do change the retained volumes
    comb_spiked_nomask = combine._combine_paid(spiked, tes)
    assert not np.allclose(comb_spiked_nomask[:, sample_mask], comb[:, sample_mask])


def test__combine_paid_constant_voxel():
    """Voxels with no signal variance fall back to TE weighting instead of failing."""
    rng = np.random.default_rng(0)
    data = rng.random((4, 3, 6)) + 1
    data[0] = np.array([[3.0], [2.0], [1.0]])  # constant over time in every echo
    tes = np.array([[0.010, 0.020, 0.030]])
    comb = combine._combine_paid(data, tes)
    expected = np.average(np.array([3.0, 2.0, 1.0]), weights=tes[0])
    np.testing.assert_allclose(comb[0], expected)
    assert np.isfinite(comb).all()


def test_make_optcom_sample_mask_validation():
    """make_optcom rejects sample masks of the wrong length or with no retained volumes."""
    data = np.random.random((5, 3, 4))
    adaptive_mask = np.full(5, 3)
    tes = np.array([0.010, 0.020, 0.030])
    with pytest.raises(ValueError, match="one value per volume"):
        combine.make_optcom(
            data,
            tes,
            adaptive_mask,
            combmode="paid",
            sample_mask=np.ones(3, dtype=bool),
        )

    with pytest.raises(ValueError, match="at least one volume"):
        combine.make_optcom(
            data,
            tes,
            adaptive_mask,
            combmode="paid",
            sample_mask=np.zeros(4, dtype=bool),
        )


def test_make_optcom():
    """Test tedana.combine.make_optcom."""
    np.random.seed(0)
    n_voxels, n_echos, n_trs = 20, 3, 10
    n_mask = 5
    data = np.random.random((n_voxels, n_echos, n_trs))
    adaptive_mask = np.zeros(n_voxels, dtype=int)
    adaptive_mask[:n_mask] = 1
    adaptive_mask[0] = 2
    tes = np.array([0.010, 0.020, 0.030])  # E

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

    bad_tes = np.array([0.010, 0.020])
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
