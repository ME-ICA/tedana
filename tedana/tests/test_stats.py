"""Tests for the tedana stats module."""

import random

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from numpy.matlib import repmat

from tedana.stats import fit_model, get_coeffs, getfbounds, voxelwise_univariate_zstats


def test_break_voxelwise_univariate_zstats():
    """Ensure that voxelwise_univariate_zstats fails when input data do not have right shapes."""
    n_samples, n_vols, n_comps = 10000, 100, 50
    data = np.empty((n_samples, n_vols))
    mixing = np.empty((n_vols, n_comps))

    data = np.empty(n_samples)
    with pytest.raises(ValueError):
        voxelwise_univariate_zstats(data, mixing)

    data = np.empty((n_samples, n_vols))
    mixing = np.empty(n_vols)
    with pytest.raises(ValueError):
        voxelwise_univariate_zstats(data, mixing)

    mixing = np.empty((n_vols + 1, n_comps))
    with pytest.raises(ValueError):
        voxelwise_univariate_zstats(data, mixing)


def test_smoke_voxelwise_univariate_zstats():
    """Ensures that voxelwise_univariate_zstats works with rand inputs and diff optional params."""
    n_samples, n_times, n_components = 100, 20, 6
    data = np.random.random((n_samples, n_times))
    mixing = np.random.random((n_times, n_components))

    assert voxelwise_univariate_zstats(data, mixing) is not None


def test_get_coeffs():
    """Check least squares coefficients."""
    # Simulate one voxel with 40 TRs
    data = np.empty((2, 40))
    data[0, :] = np.arange(0, 200, 5)
    data[1, :] = np.arange(0, 200, 5)
    x = np.arange(0, 40)[:, np.newaxis]

    betas = get_coeffs(data, x, add_const=False)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5.0, 5.0]))

    betas = get_coeffs(data, x, add_const=True)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5.0, 5.0]))


def test_break_get_coeffs():
    """
    Ensure that get_coeffs fails when input data do not have the right.

    shapes.
    """
    n_samples, n_echos, n_vols, n_comps = 10000, 5, 100, 50
    data = np.empty((n_samples, n_vols))
    x = np.empty((n_vols, n_comps))

    data = np.empty(n_samples)
    with pytest.raises(ValueError):
        get_coeffs(data, x, add_const=False)

    data = np.empty((n_samples, n_vols))
    x = np.empty(n_vols)
    with pytest.raises(ValueError):
        get_coeffs(data, x, add_const=False)

    data = np.empty((n_samples, n_echos, n_vols + 1))
    x = np.empty((n_vols, n_comps))
    with pytest.raises(ValueError, match="does not match first dimension of x"):
        get_coeffs(data, x, add_const=False)

    data = np.empty((n_samples, n_echos, n_vols))
    with pytest.raises((LinAlgError, ValueError), match="3-dimensional array given"):
        get_coeffs(data, x, add_const=False)


def test_smoke_get_coeffs():
    """Ensure that get_coeffs returns outputs with different inputs and optional paramters."""
    n_samples, _, n_times, n_components = 100, 5, 20, 6
    data_2d = np.random.random((n_samples, n_times))
    x = np.random.random((n_times, n_components))

    assert get_coeffs(data_2d, x) is not None
    # assert get_coeffs(data_3d, x) is not None TODO: submit an issue for the bug
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


def test_fit_model():
    """Tests for fit_model."""

    # set up data where y = weights*x + residuals
    r = 15  # number of regressors
    t = 300  # number of time points
    c = 50  # number of components
    rng = np.random.default_rng(42)  # using a fixed seed
    x = rng.random(size=(t, r))
    weights = rng.random(size=(r, c))
    # Making the residuals sufficiently small for the fit to be precise to 4 decimals
    residuals = rng.random(size=(t, c)) / 1000000
    y = np.empty((t, c))
    for cidx in range(c):
        y[:, cidx] = (x * repmat(weights[:, cidx], t, 1)).sum(axis=1)
    y = y + residuals

    # Fitting model and confirming outputs are the correct shape
    # and beta fits match inputted weights to four decimal places
    betas, sse, df = fit_model(x, y)
    assert df == (t - r)
    assert sse.shape == (c,)
    assert (np.round(betas, decimals=4) == np.round(weights, decimals=4)).all()

    # Outputting the residual and checking it matches the inputted residual
    fit_residuals = fit_model(x, y, output_residual=True)
    assert (np.round(fit_residuals, decimals=4) == np.round(residuals, decimals=4)).all()
