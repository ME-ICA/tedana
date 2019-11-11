"""
Tests for tedana.stats.computefeats2
"""

import numpy as np
import pytest

from tedana.stats import computefeats2


def test_break_computefeats2():
    """
    Ensure that computefeats2 fails when input data do not have the right
    shapes.
    """
    n_samples, n_vols, n_comps = 10000, 100, 50
    data = np.empty((n_samples, n_vols))
    mmix = np.empty((n_vols, n_comps))
    mask = np.empty((n_samples))

    data = np.empty((n_samples))
    with pytest.raises(ValueError) as e_info:
        computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('Parameter data should be 2d, not {0}d'.format(data.ndim))

    data = np.empty((n_samples, n_vols))
    mmix = np.empty((n_vols))
    with pytest.raises(ValueError) as e_info:
        computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('Parameter mmix should be 2d, not {0}d'.format(mmix.ndim))

    mmix = np.empty((n_vols, n_comps))
    mask = np.empty((n_samples, n_vols))
    with pytest.raises(ValueError) as e_info:
        computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('Parameter mask should be 1d, not {0}d'.format(mask.ndim))

    mask = np.empty((n_samples+1))
    with pytest.raises(ValueError) as e_info:
        computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('First dimensions (number of samples) of data ({0}) '
                                 'and mask ({1}) do not match.'.format(data.shape[0],
                                                                       mask.shape[0]))
    data.shape[1] != mmix.shape[0]
    mask = np.empty((n_samples))
    mmix = np.empty((n_vols+1, n_comps))
    with pytest.raises(ValueError) as e_info:
        computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('Second dimensions (number of volumes) of data ({0}) '
                                 'and mmix ({1}) do not match.'.format(data.shape[0],
                                                                       mmix.shape[0]))


# SMOKE TEST

def test_smoke_computefeats2():
    """
    Ensures that computefeats2 works with random inputs and different optional parameters
    """
    n_samples, n_times, n_components = 100, 20, 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))  
    mask = np.random.randint(2, size=n_samples)     

    assert computefeats2(data, mmix) is not None
    assert computefeats2(data, mmix, mask=mask) is not None
    assert computefeats2(data, mmix, normalize=False) is not None     


