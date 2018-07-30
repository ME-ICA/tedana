"""
Tests for tedana.model.fit
"""

import numpy as np

from tedana.model import fit


def test_get_coeffs():
    """
    Check least squares coefficients.
    """
    # Simulate one voxel with 40 TRs
    data = np.empty((2, 40))
    data[0, :] = np.arange(0, 200, 5)
    data[1, :] = np.arange(0, 200, 5)
    X = np.arange(0, 40)[:, np.newaxis]
    mask = np.array([True, False])

    betas = fit.get_coeffs(data, X, mask=None, add_const=False)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5., 5.]))

    betas = fit.get_coeffs(data, X, mask=None, add_const=True)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5., 5.]))

    betas = fit.get_coeffs(data, X, mask=mask, add_const=False)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5, 0]))

    betas = fit.get_coeffs(data, X, mask=mask, add_const=True)
    betas = np.squeeze(betas)
    assert np.allclose(betas, np.array([5, 0]))
