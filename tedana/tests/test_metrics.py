"""
Tests for tedana.model.fit
"""
import os.path as op

import pytest
import numpy as np
import pandas as pd

from tedana import io, utils
from tedana.metrics import dependence, collect
from tedana.tests.utils import get_test_data_path


@pytest.fixture(scope='module')
def testdata1():
    tes = np.array([14.5, 38.5, 62.5])
    in_files = [op.join(get_test_data_path(), 'echo{0}.nii.gz'.format(i + 1))
                for i in range(3)]
    data_cat, ref_img = io.load_data(in_files, n_echos=len(tes))
    mask, adaptive_mask = utils.make_adaptive_mask(data_cat, getsum=True)
    data_optcom = np.mean(data_cat, axis=1)
    mixing = np.random.random((data_optcom.shape[1], 50))
    data_dict = {'data_cat': data_cat,
                 'tes': tes,
                 'mask': mask,
                 'data_optcom': data_optcom,
                 'adaptive_mask': adaptive_mask,
                 'ref_img': ref_img,
                 'mixing': mixing,
                 }
    return data_dict


def test_metrics_collect(testdata1):
    """
    Test our general-purpose metric collector.
    """
    metrics = ['kappa', 'rho', 'countnoise', 'countsigFT2', 'countsigFS0',
               'dice_FT2', 'dice_FS0', 'signal-noise_t', 'variance explained',
               'normalized variance explained', 'd_table_score']
    comptable, mixing = collect.generate_metrics(
        testdata1['data_cat'], testdata1['data_optcom'], testdata1['mixing'],
        testdata1['mask'], testdata1['tes'], testdata1['ref_img'],
        metrics=metrics, sort_by='kappa', ascending=False)
    assert isinstance(comptable, pd.DataFrame)


def test_kappa_rho():
    """
    Test tedana.metrics.dependence.calculate_dependence_metrics
    """
    n_voxels, n_components = 1000, 50
    F_T2_maps = np.random.random((n_voxels, n_components))
    F_S0_maps = np.random.random((n_voxels, n_components))
    Z_maps = np.random.random((n_voxels, n_components))
    kappas, rhos = dependence.calculate_dependence_metrics(
        F_T2_maps, F_S0_maps, Z_maps)
    assert kappas.shape == rhos.shape
