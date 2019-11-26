"""
Tests for maPCA
"""

import numpy as np
import nibabel as nib
from tedana import decomposition


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

    # Testing AIC option
    u, s, varex_norm, v = decomposition.ma_pca(test_img, test_mask_img, 'kic')

    assert u.shape[0] == n_vox_total
    assert s.shape[0] == 1
    assert varex_norm.shape[0] == 1
    assert v.shape[0] == timepoints

    del u, s, varex_norm, v

    # Testing AIC option
    u, s, varex_norm, v = decomposition.ma_pca(test_img, test_mask_img, 'mdl')

    assert u.shape[0] == n_vox_total
    assert s.shape[0] == 1
    assert varex_norm.shape[0] == 1
    assert v.shape[0] == timepoints
