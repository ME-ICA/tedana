"""
Tests for tedana.decay
"""

import os.path as op

import numpy as np
import pytest

from tedana import io, utils, decay as me
from tedana.tests.utils import get_test_data_path


@pytest.fixture(scope='module')
def testdata1():
    tes = np.array([14.5, 38.5, 62.5])
    in_files = [op.join(get_test_data_path(), 'echo{0}.nii.gz'.format(i + 1))
                for i in range(3)]
    data, _ = io.load_data(in_files, n_echos=len(tes))
    mask, mask_sum = utils.make_adaptive_mask(data, getsum=True)
    fittype = 'loglin'
    data_dict = {'data': data,
                 'tes': tes,
                 'mask': mask,
                 'mask_sum': mask_sum,
                 'fittype': fittype,
                 }
    return data_dict


def test_fit_decay(testdata1):
    """
    fit_decay should return data in (samples,) shape.
    """
    t2sv, s0v, t2ss, s0vs, t2svG, s0vG = me.fit_decay(testdata1['data'],
                                                      testdata1['tes'],
                                                      testdata1['mask'],
                                                      testdata1['mask_sum'],
                                                      testdata1['fittype'])
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
                                             testdata1['mask_sum'],
                                             testdata1['fittype'])
    assert t2sv.ndim == 2
    assert s0v.ndim == 2
    assert t2svG.ndim == 2
    assert s0vG.ndim == 2


# SMOKE TESTS

def test_smoke_fit_decay():
    """
    test_smoke_fit_decay tests that the function fit_decay returns reasonable
    objects with random inputs in the correct format
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    tes = np.random.random((n_echos)).tolist()
    mask = np.random.randint(2, size=n_samples)  # generate binary mask of random 0s and 1s
    masksum = np.random.random((n_samples))
    fittype = 'loglin'
    t2s_limited, s0_limited, t2ss, s0vs, t2s_full, s0_full = me.fit_decay(
                                                            data, tes,
                                                            mask, masksum,
                                                            fittype)
    assert t2s_limited is not None
    assert s0_limited is not None
    assert t2ss is not None
    assert s0vs is not None
    assert t2s_full is not None
    assert s0_full is not None


def test_smoke_fit_decay_curvefit():
    """
    test_smoke_fit_decay tests that the function fit_decay returns reasonable
    objects with random inputs in the correct format when using the direct
    monoexponetial approach
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    tes = np.random.random((n_echos)).tolist()
    mask = np.random.randint(2, size=n_samples)  # generate binary mask of random 0s and 1s
    masksum = np.random.random((n_samples))
    fittype = 'curvefit'
    t2s_limited, s0_limited, t2ss, s0vs, t2s_full, s0_full = me.fit_decay(
                                                                data, tes,
                                                                mask,
                                                                masksum,
                                                                fittype)
    assert t2s_limited is not None
    assert s0_limited is not None
    assert t2ss is not None
    assert s0vs is not None
    assert t2s_full is not None
    assert s0_full is not None


def test_smoke_fit_decay_ts():
    """
    test_smoke_fit_decay_ts tests that the function fit_decay_ts returns reasonable
    objects with random inputs in the correct format
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    tes = np.random.random((n_echos)).tolist()
    mask = np.random.randint(2, size=n_samples)  # generate binary mask of random 0s and 1s
    masksum = np.random.random((n_samples))
    fittype = 'loglin'
    t2s_limited_ts, s0_limited_ts, t2s_full_ts, s0_full_ts = me.fit_decay_ts(
                                                                data, tes,
                                                                mask,
                                                                masksum,
                                                                fittype)
    assert t2s_limited_ts is not None
    assert s0_limited_ts is not None
    assert t2s_full_ts is not None
    assert s0_full_ts is not None


def test_smoke_fit_decay_curvefit_ts():
    """
    test_smoke_fit_decay_ts tests that the function fit_decay_ts returns reasonable
    objects with random inputs in the correct format when using the direct
    monoexponetial approach
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    tes = np.random.random((n_echos)).tolist()
    mask = np.random.randint(2, size=n_samples)  # generate binary mask of random 0s and 1s
    masksum = np.random.random((n_samples))
    fittype = 'curvefit'
    t2s_limited_ts, s0_limited_ts, t2s_full_ts, s0_full_ts = me.fit_decay_ts(
                                                                data, tes,
                                                                mask,
                                                                masksum,
                                                                fittype)
    assert t2s_limited_ts is not None
    assert s0_limited_ts is not None
    assert t2s_full_ts is not None
    assert s0_full_ts is not None
# TODO: BREAK AND UNIT TESTS
