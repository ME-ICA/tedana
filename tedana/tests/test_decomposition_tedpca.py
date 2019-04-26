"""
Tests for tedana.decomposition.tedpca
"""

import numpy as np
import pytest

from tedana.decomposition import tedpca


def test_break_tedpca():
    """
    Ensure that computefeats2 fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_time = 10000, 4, 50
    data_cat = np.empty((n_samples, n_echos, n_time))
    data_oc = np.empty((n_samples, n_time))
    combmode = 't2s'
    mask = np.empty((n_samples))
    t2s = np.empty((n_samples))
    t2sG = np.empty((n_samples))
    ref_img = ''
    tes = ''
    method = 'mle'
    
    with pytest.raises(ValueError) as e_info:
        tedpca(data_cat, data_oc, combmode, mask, t2s, t2sG, ref_img, tes, method)
    assert str(e_info.value) == ('Parameter data should be 2d, not {0}d'.format(data_oc.ndim))
