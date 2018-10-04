"""
Tests for tedana.model.monoexponential
"""

import os.path as op

import numpy as np
import pytest

from tedana import io, utils, decay as me
from tedana.tests.utils import get_test_data_path


@pytest.fixture(scope='module')
def testdata1():
    tes = np.array([14.5, 38.5, 62.5])
    in_files = [op.join(get_test_data_path(), 'echo{0}.nii.gz'.format(i+1))
                for i in range(3)]
    data, _ = io.load_data(in_files, n_echos=len(tes))
    mask, mask_sum = utils.make_adaptive_mask(data, minimum=False, getsum=True)
    data_dict = {'data': data,
                 'tes': tes,
                 'mask': mask,
                 'mask_sum': mask_sum,
                 }
    return data_dict


def test_fit_decay(testdata1):
    """
    fit_decay should return data in (samples,) shape.
    """
    t2sv, s0v, t2ss, s0vs, t2svG, s0vG = me.fit_decay(testdata1['data'],
                                                      testdata1['tes'],
                                                      testdata1['mask'],
                                                      testdata1['mask_sum'])
    assert t2sv.ndim == 1
    assert s0v.ndim == 1
    assert t2ss.ndim == 2
    assert s0vs.ndim == 2
    assert t2svG.ndim == 1
    assert s0vG.ndim == 1


def test_fit_decay_ts(testdata1):
    """
    fit_decay_ts should return data in samples x time shape.
    """
    t2sv, s0v, t2svG, s0vG = me.fit_decay_ts(testdata1['data'],
                                             testdata1['tes'],
                                             testdata1['mask'],
                                             testdata1['mask_sum'])
    assert t2sv.ndim == 2
    assert s0v.ndim == 2
    assert t2svG.ndim == 2
    assert s0vG.ndim == 2
