"""
Tests for tedana.model.fit
"""

import numpy as np
import pytest

from tedana.model import fit


def test_break_fitmodels_direct():
    """
    Ensure that fitmodels_direct fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_vols, n_comps = 10000, 4, 100, 50
    catd = np.empty((n_samples, n_echos, n_vols))
    mmix = np.empty((n_vols, n_comps))
    mask = np.empty((n_samples))
    t2s = np.empty((n_samples, n_vols))
    t2s_full = np.empty((n_samples, n_vols))
    tes = np.empty((n_echos))
    combmode = 't2s'
    ref_img = ''

    catd = np.empty((n_samples+1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.fitmodels_direct(catd=catd, mmix=mmix, mask=mask, t2s=t2s,
                             t2s_full=t2s_full, tes=tes, combmode=combmode,
                             ref_img=ref_img,
                             reindex=False, mmixN=None, full_sel=True)
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), '
                                 't2s ({1}), and mask ({2}) do not '
                                 'match'.format(catd.shape[0], t2s.shape[0],
                                                mask.shape[0]))

    catd = np.empty((n_samples, n_echos, n_vols))
    t2s = np.empty((n_samples+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.fitmodels_direct(catd=catd, mmix=mmix, mask=mask, t2s=t2s,
                             t2s_full=t2s_full, tes=tes, combmode=combmode,
                             ref_img=ref_img,
                             reindex=False, mmixN=None, full_sel=True)
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), '
                                 't2s ({1}), and mask ({2}) do not '
                                 'match'.format(catd.shape[0], t2s.shape[0],
                                                mask.shape[0]))

    t2s = np.empty((n_samples, n_vols))
    t2s_full = np.empty((n_samples+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.fitmodels_direct(catd=catd, mmix=mmix, mask=mask, t2s=t2s,
                             t2s_full=t2s_full, tes=tes, combmode=combmode,
                             ref_img=ref_img,
                             reindex=False, mmixN=None, full_sel=True)
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), '
                                 't2s ({1}), and mask ({2}) do not '
                                 'match'.format(catd.shape[0], t2s.shape[0],
                                                mask.shape[0]))

    catd = np.empty((n_samples, n_echos+1, n_vols))
    t2s_full = np.empty((n_samples, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.fitmodels_direct(catd=catd, mmix=mmix, mask=mask, t2s=t2s,
                             t2s_full=t2s_full, tes=tes, combmode=combmode,
                             ref_img=ref_img,
                             reindex=False, mmixN=None, full_sel=True)
    assert str(e_info.value) == ('Second dimension of catd ({0}) does not '
                                 'match number '
                                 'of echoes provided (tes; '
                                 '{1})'.format(catd.shape[1], len(tes)))

    catd = np.empty((n_samples, n_echos, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        fit.fitmodels_direct(catd=catd, mmix=mmix, mask=mask, t2s=t2s,
                             t2s_full=t2s_full, tes=tes, combmode=combmode,
                             ref_img=ref_img,
                             reindex=False, mmixN=None, full_sel=True)
    assert str(e_info.value) == ('Third dimension (number of volumes) of catd ({0}) '
                                 'does not match first dimension of '
                                 'mmix ({1})'.format(catd.shape[2], mmix.shape[0]))

    catd = np.empty((n_samples, n_echos, n_vols))
    t2s = np.empty((n_samples, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        fit.fitmodels_direct(catd=catd, mmix=mmix, mask=mask, t2s=t2s,
                             t2s_full=t2s_full, tes=tes, combmode=combmode,
                             ref_img=ref_img,
                             reindex=False, mmixN=None, full_sel=True)
    assert str(e_info.value) == ('Shape of t2s array {0} does not match shape of '
                                 't2s_full array {1}'.format(t2s.shape, t2s_full.shape))

    t2s_full = np.empty((n_samples, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        fit.fitmodels_direct(catd=catd, mmix=mmix, mask=mask, t2s=t2s,
                             t2s_full=t2s_full, tes=tes, combmode=combmode,
                             ref_img=ref_img,
                             reindex=False, mmixN=None, full_sel=True)
    assert str(e_info.value) == ('Third dimension (number of volumes) of catd ({0}) '
                                 'does not match second dimension of '
                                 't2s ({1})'.format(catd.shape[2], t2s.shape[1]))
