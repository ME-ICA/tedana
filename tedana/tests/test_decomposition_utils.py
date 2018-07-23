"""
Tests for tedana.decomposition._utils
"""

import numpy as np

from tedana.decomposition import _utils


def test_wavelet_transforms():
    """
    Ensure that wavelet transform and inverse apply to rows (not columns) and
    retrieve original data.
    """
    n_samps = 20
    n_trs = 100
    data = np.random.random((n_samps, n_trs))
    data_wv, n_coefs = _utils.dwtmat(data)
    assert data_wv.shape[0] == n_samps

    data_recon = _utils.idwtmat(data_wv, n_coefs)
    assert np.allclose(data, data_recon)
