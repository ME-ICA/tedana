"""
Tests for tedana.model.fit
"""

import numpy as np
import pytest

from tedana.model import fit


def test_break_dependence_metrics():
    """
    Ensure that dependence_metrics fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_vols, n_comps = 10000, 4, 100, 50
    catd = np.empty((n_samples, n_echos, n_vols))
    tsoc = np.empty((n_samples, n_vols))
    mmix = np.empty((n_vols, n_comps))
    mask = np.empty((n_samples))
    t2s = np.empty((n_samples, n_vols))
    t2s_full = np.empty((n_samples, n_vols))
    tes = np.empty((n_echos))
    combmode = 't2s'
    ref_img = ''

    catd = np.empty((n_samples+1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix, mask=mask,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, method='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), t2s ({2}), and mask '
                                 '({3}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0], t2s.shape[0],
                                    mask.shape[0]))

    catd = np.empty((n_samples, n_echos, n_vols))
    t2s = np.empty((n_samples+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix, mask=mask,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, method='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), t2s ({2}), and mask '
                                 '({3}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0], t2s.shape[0],
                                    mask.shape[0]))

    t2s = np.empty((n_samples, n_vols))
    tsoc = np.empty((n_samples+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix, mask=mask,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, method='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), t2s ({2}), and mask '
                                 '({3}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0], t2s.shape[0],
                                    mask.shape[0]))

    catd = np.empty((n_samples, n_echos+1, n_vols))
    tsoc = np.empty((n_samples, n_vols))
    with pytest.raises(ValueError) as e_info:
        fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix, mask=mask,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, method='kundu_v3')
    assert str(e_info.value) == ('Second dimension of catd ({0}) does not '
                                 'match number '
                                 'of echoes provided (tes; '
                                 '{1})'.format(catd.shape[1], len(tes)))

    catd = np.empty((n_samples, n_echos, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix, mask=mask,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, method='kundu_v3')
    assert str(e_info.value) == ('Number of volumes in catd '
                                 '({0}), tsoc ({1}), and '
                                 'mmix ({2}) do not '
                                 'match.'.format(catd.shape[2], tsoc.shape[1],
                                                 mmix.shape[0]))

    catd = np.empty((n_samples, n_echos, n_vols))
    t2s = np.empty((n_samples, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        fit.dependence_metrics(catd=catd, tsoc=tsoc, mmix=mmix, mask=mask,
                               t2s=t2s, tes=tes, ref_img=ref_img,
                               reindex=False, mmixN=None, method='kundu_v3')
    assert str(e_info.value) == ('Third dimension (number of volumes) of catd ({0}) '
                                 'does not match second dimension of '
                                 't2s ({1})'.format(catd.shape[2], t2s.shape[1]))
