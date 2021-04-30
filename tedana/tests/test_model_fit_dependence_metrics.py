"""
Tests for tedana.metrics.fit
"""
import os

import numpy as np
import pytest

from tedana.metrics import kundu_fit
from tedana.io import OutputGenerator
from tedana.tests.utils import get_test_data_path

data_dir = get_test_data_path()
ref_img = os.path.join(data_dir, 'mask.nii.gz')


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
    tes = np.empty((n_echos))
    generator = OutputGenerator(ref_img)

    # Shape of catd is wrong
    catd = np.empty((n_samples + 1, n_echos, n_vols))
    with pytest.raises(ValueError):
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, generator=generator,
            reindex=False, mmixN=None, algorithm='kundu_v3')

    # Shape of adaptive_mask is wrong
    catd = np.empty((n_samples, n_echos, n_vols))
    adaptive_mask = np.empty((n_samples + 1))
    with pytest.raises(ValueError):
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, generator=generator,
            reindex=False, mmixN=None, algorithm='kundu_v3')

    # Shape of tsoc is wrong
    adaptive_mask = np.empty((n_samples))
    tsoc = np.empty((n_samples + 1, n_vols))
    with pytest.raises(ValueError):
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, generator=generator,
            reindex=False, mmixN=None, algorithm='kundu_v3')

    # Shape of catd is wrong
    catd = np.empty((n_samples, n_echos + 1, n_vols))
    tsoc = np.empty((n_samples, n_vols))
    with pytest.raises(ValueError):
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, generator=generator,
            reindex=False, mmixN=None, algorithm='kundu_v3')

    # Shape of catd is wrong
    catd = np.empty((n_samples, n_echos, n_vols + 1))
    with pytest.raises(ValueError):
        kundu_fit.dependence_metrics(
            catd=catd, tsoc=tsoc, mmix=mmix,
            adaptive_mask=adaptive_mask, tes=tes, generator=generator,
            reindex=False, mmixN=None, algorithm='kundu_v3')
