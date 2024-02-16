"""Tests for the tedana stats module."""

import random

import numpy as np
import pytest

from tedana.stats import computefeats2, get_coeffs, getfbounds


def test_break_computefeats2():
    """Ensure that computefeats2 fails when input data do not have the right shapes."""
    n_samples, n_vols, n_comps = 10000, 100, 50
    data = np.empty((n_samples, n_vols))
    mmix = np.empty((n_vols, n_comps))
    mask = np.empty(n_samples)

    data = np.empty(n_samples)
    with pytest.raises(ValueError):
        computefeats2(data, mmix, mask, normalize=True)

    data = np.empty((n_samples, n_vols))
    mmix = np.empty(n_vols)
    with pytest.raises(ValueError):
        computefeats2(data, mmix, mask, normalize=True)

    mmix = np.empty((n_vols, n_comps))
    mask = np.empty((n_samples, n_vols))
    with pytest.raises(ValueError):
        computefeats2(data, mmix, mask, normalize=True)

    mask = np.empty(n_samples + 1)
    with pytest.raises(ValueError):
        computefeats2(data, mmix, mask, normalize=True)
    data.shape[1] != mmix.shape[0]
    mask = np.empty(n_samples)
    mmix = np.empty((n_vols + 1, n_comps))
    with pytest.raises(ValueError):
        computefeats2(data, mmix, mask, normalize=True)


def test_smoke_computefeats2():
    """Ensures that computefeats2 works with random inputs and different optional parameters."""
    n_samples, n_times, n_components = 100, 20, 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)

    assert computefeats2(data, mmix) is not None
    assert computefeats2(data, mmix, mask=mask) is not None
    assert computefeats2(data, mmix, normalize=False) is not None


def test_get_coeffs():
    """Check least squares coefficients."""
    # Simulate one voxel with 40 TRs
    data = np.empty((2, 40))
    data[0, :] = np.arange(0, 200, 5)
    data[1, :] = np.arange(0, 200, 5)
    x = np.arange(0, 40)[:, np.newaxis]
    mask = np.array([True, False])

    betas = get_coeffs(data, x, mask=None, add_const=False)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5.0, 5.0]))

    betas = get_coeffs(data, x, mask=None, add_const=True)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5.0, 5.0]))

    betas = get_coeffs(data, x, mask=mask, add_const=False)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5, 0]))

    betas = get_coeffs(data, x, mask=mask, add_const=True)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5, 0]))


def test_break_get_coeffs():
    """
    Ensure that get_coeffs fails when input data do not have the right.

    shapes.
    """
    n_samples, n_echos, n_vols, n_comps = 10000, 5, 100, 50
    data = np.empty((n_samples, n_vols))
    x = np.empty((n_vols, n_comps))
    mask = np.empty(n_samples)

    data = np.empty(n_samples)
    with pytest.raises(ValueError):
        get_coeffs(data, x, mask, add_const=False)

    data = np.empty((n_samples, n_vols))
    x = np.empty(n_vols)
    with pytest.raises(ValueError):
        get_coeffs(data, x, mask, add_const=False)

    data = np.empty((n_samples, n_echos, n_vols + 1))
    x = np.empty((n_vols, n_comps))
    with pytest.raises(ValueError):
        get_coeffs(data, x, mask, add_const=False)

    data = np.empty((n_samples, n_echos, n_vols))
    mask = np.empty((n_samples, n_echos, n_vols))
    with pytest.raises(ValueError):
        get_coeffs(data, x, mask, add_const=False)

    mask = np.empty((n_samples + 1, n_echos))
    with pytest.raises(ValueError):
        get_coeffs(data, x, mask, add_const=False)


def test_smoke_get_coeffs():
    """Ensure that get_coeffs returns outputs with different inputs and optional paramters."""
    n_samples, _, n_times, n_components = 100, 5, 20, 6
    data_2d = np.random.random((n_samples, n_times))
    x = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)

    assert get_coeffs(data_2d, x) is not None
    # assert get_coeffs(data_3d, x) is not None TODO: submit an issue for the bug
    assert get_coeffs(data_2d, x, mask=mask) is not None
    assert get_coeffs(data_2d, x, add_const=True) is not None


def test_getfbounds():
    good_inputs = range(1, 12)

    for n_echos in good_inputs:
        getfbounds(n_echos)


def test_smoke_getfbounds():
    """Ensures that getfbounds returns outputs when fed in a random number of echo."""
    n_echos = random.randint(3, 10)  # At least two echos!
    f05, f025, f01 = getfbounds(n_echos)

    assert f05 is not None
    assert f025 is not None
    assert f01 is not None
