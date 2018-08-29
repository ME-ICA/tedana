"""
Tests for tedana.model.fit
"""

import numpy as np
import pytest

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


def test_break_get_coeffs():
    """
    Ensure that get_coeffs fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_vols, n_comps = 10000, 5, 100, 50
    data = np.empty((n_samples, n_vols))
    X = np.empty((n_vols, n_comps))
    mask = np.empty((n_samples))

    data = np.empty((n_samples))
    with pytest.raises(ValueError) as e_info:
        fit.get_coeffs(data, X, mask, add_const=False)
    assert str(e_info.value) == ('Parameter data should be 2d or 3d, not {0}d'.format(data.ndim))

    data = np.empty((n_samples, n_vols))
    X = np.empty((n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.get_coeffs(data, X, mask, add_const=False)
    assert str(e_info.value) == ('Parameter X should be 2d, not {0}d'.format(X.ndim))

    data = np.empty((n_samples, n_echos, n_vols+1))
    X = np.empty((n_vols, n_comps))
    with pytest.raises(ValueError) as e_info:
        fit.get_coeffs(data, X, mask, add_const=False)
    assert str(e_info.value) == ('Last dimension (dimension {0}) of data ({1}) does not '
                                 'match first dimension of '
                                 'X ({2})'.format(data.ndim, data.shape[-1], X.shape[0]))

    data = np.empty((n_samples, n_echos, n_vols))
    mask = np.empty((n_samples, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.get_coeffs(data, X, mask, add_const=False)
    assert str(e_info.value) == ('Parameter data should be 1d or 2d, not {0}d'.format(mask.ndim))

    mask = np.empty((n_samples+1, n_echos))
    with pytest.raises(ValueError) as e_info:
        fit.get_coeffs(data, X, mask, add_const=False)
    assert str(e_info.value) == ('First dimensions of data ({0}) and mask ({1}) do not '
                                 'match'.format(data.shape[0], mask.shape[0]))
