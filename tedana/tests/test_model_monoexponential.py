"""
Tests for tedana.model.monoexponential
"""
import os.path as op

import numpy as np

from tedana import utils
from tedana.model import monoexponential as me
from tedana.tests.utils import get_test_data_path


def test_fit_decay():
    tes = np.array([14.5, 38.5, 62.5])
    in_files = [op.join(get_test_data_path(), 'echo{0}.nii.gz'.format(i+1))
                for i in range(3)]
    data, _ = utils.load_data(in_files, n_echos=len(tes))
    mask, mask_sum = utils.make_adaptive_mask(data, minimum=False, getsum=True)
    t2sv, s0v, t2ss, s0vs, t2svG, s0vG = me.fit_decay(data, tes, mask,
                                                      mask_sum, start_echo=1)
    assert len(t2sv.shape) == 1
    assert len(s0v.shape) == 1
    assert len(t2ss.shape) == 2
    assert len(s0vs.shape) == 2
    assert len(t2svG.shape) == 1
    assert len(s0vG.shape) == 1


def test_fit_decay_ts():
    """
    fit_decay_ts should return data in samples x time shape.
    """
    tes = np.array([14.5, 38.5, 62.5])
    in_files = [op.join(get_test_data_path(), 'echo{0}.nii.gz'.format(i+1))
                for i in range(3)]
    data, _ = utils.load_data(in_files, n_echos=len(tes))
    mask, mask_sum = utils.make_adaptive_mask(data, minimum=False, getsum=True)
    t2sv, s0v, t2svG, s0vG = me.fit_decay_ts(data, tes, mask, mask_sum,
                                             start_echo=1)
    assert len(t2sv.shape) == 2
    assert len(s0v.shape) == 2
    assert len(t2svG.shape) == 2
    assert len(s0vG.shape) == 2
