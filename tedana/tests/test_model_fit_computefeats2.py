"""
Tests for tedana.model.fit
"""

import numpy as np
import pytest

from tedana.model import fit


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
        fit.computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('Parameter data should be 2d, not {0}d'.format(data.ndim))

    data = np.empty((n_samples, n_vols))
    mmix = np.empty((n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('Parameter mmix should be 2d, not {0}d'.format(mmix.ndim))

    mmix = np.empty((n_vols, n_comps))
    mask = np.empty((n_samples, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('Parameter mask should be 1d, not {0}d'.format(mask.ndim))

    mask = np.empty((n_samples+1))
    with pytest.raises(ValueError) as e_info:
        fit.computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('First dimensions (number of samples) of data ({0}) '
                                 'and mask ({1}) do not match.'.format(data.shape[0],
                                                                       mask.shape[0]))
    data.shape[1] != mmix.shape[0]
    mask = np.empty((n_samples))
    mmix = np.empty((n_vols+1, n_comps))
    with pytest.raises(ValueError) as e_info:
        fit.computefeats2(data, mmix, mask, normalize=True)
    assert str(e_info.value) == ('Second dimensions (number of volumes) of data ({0}) '
                                 'and mmix ({1}) do not match.'.format(data.shape[0],
                                                                       mmix.shape[0]))
