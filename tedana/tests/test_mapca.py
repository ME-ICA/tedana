"""
Tests for maPCA
"""

import numpy as np
import nibabel as nib
from tedana import decomposition
from pytest import raises


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
