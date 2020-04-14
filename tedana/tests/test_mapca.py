"""
Tests for maPCA
"""

import numpy as np
import nibabel as nib
from tedana import decomposition
from pytest import raises
from tedana.decomposition.ma_pca import _autocorr, _check_order, _parzen_win
from tedana.decomposition.ma_pca import _subsampling, _kurtn, _icatb_svd, _eigensp_adj


def test_autocorr():
    """
    Unit test on _autocorr function
    """
    test_data = np.array([1, 2, 3, 4])
    test_result = np.array([30, 20, 11, 4])
    autocorr = _autocorr(test_data)
    assert np.array_equal(autocorr, test_result)


def test_check_order():
    """
    Unit test on _check_order function
    """
    test_order = -1
    with raises(ValueError) as errorinfo:
        ord_out, w, trivwin = _check_order(test_order)
    assert 'Order cannot be less than zero' in str(errorinfo.value)

    test_order = 0
    ord_out, w, trivwin = _check_order(test_order)
    assert ord_out == test_order
    assert trivwin

    test_order = 1
    ord_out, w, trivwin = _check_order(test_order)
    assert ord_out == test_order
    assert w == 1
    assert trivwin

    test_order = 4
    ord_out, w, trivwin = _check_order(test_order)
    assert ord_out == test_order
    assert not trivwin


def test_parzen_win():
    test_npoints = 3
    test_result = np.array([0.07407407, 1, 0.07407407])
    win = _parzen_win(test_npoints)
    assert np.allclose(win, test_result)

    test_npoints = 1
    win = _parzen_win(test_npoints)
    assert win == 1


def test_ent_rate_sp():
    """
    Check that ent_rate_sp runs correctly, i.e. returns a float
    """
    test_data = np.random.rand(200, 10, 10)
    ent_rate = decomposition.ent_rate_sp(test_data, 1)
    assert isinstance(ent_rate, float)
    assert ent_rate.ndim == 0
    assert ent_rate.size == 1

    # Checks ValueError with std = 0
    test_data = np.ones((200, 10, 10))
    with raises(ValueError) as errorinfo:
        ent_rate = decomposition.ent_rate_sp(test_data, 1)
    assert 'Divide by zero encountered' in str(errorinfo.value)

    # Checks ValueError with incorrect matrix dimensions
    test_data = np.ones((200, 10, 10, 200))
    with raises(ValueError) as errorinfo:
        ent_rate = decomposition.ent_rate_sp(test_data, 1)
    assert 'Incorrect matrix dimensions' in str(errorinfo.value)


def test_subsampling():
    """
    Unit test for subsampling function
    """
    test_data = np.array([1])
    with raises(ValueError) as errorinfo:
        sub_data = _subsampling(test_data, 1)
    assert 'Unrecognized matrix dimension' in str(errorinfo.value)

    test_data = np.random.rand(2, 3, 4)
    sub_data = _subsampling(test_data, sub_depth=2)
    assert sub_data.shape == (1, 2, 2)


def test_kurtn():
    """
    Unit test for _kurtn function
    """
    test_data = np.random.rand(2, 3, 4)
    kurt = _kurtn(test_data)
    assert kurt.shape == (3, 1)


def test_icatb_svd():
    """
    Unit test for icatb_svd function.
    """
    test_data = np.diag(np.random.rand(5))
    V, Lambda = _icatb_svd(test_data)
    assert np.allclose(np.sum(V, axis=0), np.ones((5,)))


def test_eigensp_adj():
    """
    Unit test for eigensp_adj function
    """
    test_eigen = np.array([0.9, 0.5, 0.2, 0.1, 0])
    n_effective = 2
    test_result = np.array([0.13508894, 0.11653465, 0.06727316, 0.05211424, 0.])
    lambd_adj = _eigensp_adj(test_eigen, n_effective, p=test_eigen.shape[0])
    assert np.allclose(lambd_adj, test_result)


def test_ma_pca():
    """
    Check that ma_pca runs correctly with all three options
    """

    timepoints = 200
    nvox = 20
    n_vox_total = nvox ** 3

    # Creates fake data to test with
    test_data = np.random.random((nvox, nvox, nvox, timepoints))
    time = np.linspace(0, 400, timepoints)
    freq = 1
    test_data = test_data + np.sin(2 * np.pi * freq * time)
    xform = np.eye(4) * 2
    test_img = nib.nifti1.Nifti1Image(test_data, xform)

    # Creates mask
    test_mask = np.ones((nvox, nvox, nvox))
    test_mask_img = nib.nifti1.Nifti1Image(test_mask, xform)

    # Testing AIC option
    u, s, varex_norm, v = decomposition.ma_pca(test_img, test_mask_img, 'aic')

    assert u.shape[0] == n_vox_total
    assert s.shape[0] == 1
    assert varex_norm.shape[0] == 1
    assert v.shape[0] == timepoints

    del u, s, varex_norm, v

    # Testing KIC option
    u, s, varex_norm, v = decomposition.ma_pca(test_img, test_mask_img, 'kic')

    assert u.shape[0] == n_vox_total
    assert s.shape[0] == 1
    assert varex_norm.shape[0] == 1
    assert v.shape[0] == timepoints

    del u, s, varex_norm, v

    # Testing MDL option
    u, s, varex_norm, v = decomposition.ma_pca(test_img, test_mask_img, 'mdl')

    assert u.shape[0] == n_vox_total
    assert s.shape[0] == 1
    assert varex_norm.shape[0] == 1
    assert v.shape[0] == timepoints
