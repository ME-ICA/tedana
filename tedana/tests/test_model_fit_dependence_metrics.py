"""
Tests for tedana.metrics.fit
"""

import numpy as np
import pytest

from tedana.metrics import kundu_fit


def test_break_dependence_metrics():
    """
    Ensure that dependence_metrics fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_vols, n_comps = 10000, 4, 100, 50
    catd = np.empty((n_samples, n_echos, n_vols))
    tsoc = np.empty((n_samples, n_vols))
    mmix = np.empty((n_vols, n_comps))
    adaptive_mask = np.empty((n_samples))
    t2s_full = np.empty((n_samples, n_vols))
    tes = np.empty((n_echos))
    combmode = 'adaptive_mask'
    ref_img = ''

    # Shape of catd is wrong
    catd = np.empty((n_samples+1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, ref_img=ref_img,
            reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), '
                                 'and adaptive_mask ({2}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0],
                                    adaptive_mask.shape[0]))

    # Shape of adaptive_mask is wrong
    catd = np.empty((n_samples, n_echos, n_vols))
    adaptive_mask = np.empty((n_samples+1))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, ref_img=ref_img,
            reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), '
                                 'and adaptive_mask ({2}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0],
                                    adaptive_mask.shape[0]))

    # Shape of tsoc is wrong
    adaptive_mask = np.empty((n_samples))
    tsoc = np.empty((n_samples+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, ref_img=ref_img,
            reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('First dimensions (number of samples) of '
                                 'catd ({0}), tsoc ({1}), '
                                 'and adaptive_mask ({2}) do not match'.format(
                                    catd.shape[0], tsoc.shape[0],
                                    adaptive_mask.shape[0]))

    # Shape of catd is wrong
    catd = np.empty((n_samples, n_echos+1, n_vols))
    tsoc = np.empty((n_samples, n_vols))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, ref_img=ref_img,
            reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('Second dimension of catd ({0}) does not '
                                 'match number '
                                 'of echoes provided (tes; '
                                 '{1})'.format(catd.shape[1], len(tes)))

    # Shape of catd is wrong
    catd = np.empty((n_samples, n_echos, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, ref_img=ref_img,
            reindex=False, mmixN=None, algorithm='kundu_v3')
    assert str(e_info.value) == ('Number of volumes in catd '
                                 '({0}), tsoc ({1}), and '
                                 'mmix ({2}) do not '
                                 'match.'.format(catd.shape[2], tsoc.shape[1],
                                                 mmix.shape[0]))
