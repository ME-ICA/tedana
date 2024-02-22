"""Tests for tedana.decay."""

import os.path as op

import numpy as np
import pytest

from tedana import combine
from tedana import decay as me
from tedana import io, utils
from tedana.tests.utils import get_test_data_path


@pytest.fixture(scope="module")
def testdata1():
    tes = np.array([14.5, 38.5, 62.5])
    in_files = [op.join(get_test_data_path(), f"echo{i + 1}.nii.gz") for i in range(3)]
    data, _ = io.load_data(in_files, n_echos=len(tes))
    mask, adaptive_mask = utils.make_adaptive_mask(data, getsum=True)
    fittype = "loglin"
    data_dict = {
        "data": data,
        "tes": tes,
        "mask": mask,
        "adaptive_mask": adaptive_mask,
        "fittype": fittype,
    }
    return data_dict


def test_fit_decay(testdata1):
    """Fit_decay should return data in (samples,) shape."""
    t2sv, s0v, t2svg, s0vg = me.fit_decay(
        testdata1["data"],
        testdata1["tes"],
        testdata1["mask"],
        testdata1["adaptive_mask"],
        testdata1["fittype"],
    )
    assert t2sv.ndim == 1
    assert s0v.ndim == 1
    assert t2svg.ndim == 1
    assert s0vg.ndim == 1


def test_fit_decay_ts(testdata1):
    """Fit_decay_ts should return data in samples x time shape."""
    t2sv, s0v, t2svg, s0vg = me.fit_decay_ts(
        testdata1["data"],
        testdata1["tes"],
        testdata1["mask"],
        testdata1["adaptive_mask"],
        testdata1["fittype"],
    )
    assert t2sv.ndim == 2
    assert s0v.ndim == 2
    assert t2svg.ndim == 2
    assert s0vg.ndim == 2


def test__apply_t2s_floor():
    """
    _apply_t2s_floor applies a floor to T2* values to prevent a ZeroDivisionError during.

    optimal combination.
    """
    n_voxels, n_echos, n_trs = 100, 5, 25
    echo_times = np.array([2, 23, 54, 75, 96])
    me_data = np.random.random((n_voxels, n_echos, n_trs))
    t2s = np.random.random(n_voxels) * 1000
    t2s[t2s < 1] = 1  # Crop at 1 ms to be safe
    t2s[0] = 0.001

    # First establish a failure
    with pytest.raises(ZeroDivisionError):
        _ = combine._combine_t2s(me_data, echo_times[None, :], t2s[:, None])

    # Now correct the T2* map and get a successful result.
    t2s_corrected = me._apply_t2s_floor(t2s, echo_times)
    assert t2s_corrected[0] != t2s[0]  # First value should be corrected
    assert np.array_equal(t2s_corrected[1:], t2s[1:])  # No other values should be corrected
    combined = combine._combine_t2s(me_data, echo_times[None, :], t2s_corrected[:, None])
    assert np.all(combined != 0)


# SMOKE TESTS


def test_smoke_fit_decay():
    """
    Test_smoke_fit_decay tests that the function fit_decay returns reasonable.

    objects with semi-random inputs in the correct format.

    A mask with at least some "good" voxels and an adaptive mask where all
    good voxels have at least two good echoes are generated to ensure that
    the decay-fitting function has valid voxels on which to run.
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    tes = np.random.random(n_echos).tolist()
    mask = np.ones(n_samples, dtype=int)
    mask[n_samples // 2 :] = 0
    adaptive_mask = np.random.randint(2, n_echos, size=(n_samples)) * mask
    fittype = "loglin"
    t2s_limited, s0_limited, t2s_full, s0_full = me.fit_decay(
        data, tes, mask, adaptive_mask, fittype
    )
    assert t2s_limited is not None
    assert s0_limited is not None
    assert t2s_full is not None
    assert s0_full is not None


def test_smoke_fit_decay_curvefit():
    """
    Test_smoke_fit_decay tests that the function fit_decay returns reasonable.

    objects with random inputs in the correct format when using the direct.

    monoexponetial approach.
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    tes = np.random.random(n_echos).tolist()
    mask = np.ones(n_samples, dtype=int)
    mask[n_samples // 2 :] = 0
    adaptive_mask = np.random.randint(2, n_echos, size=(n_samples)) * mask
    fittype = "curvefit"
    t2s_limited, s0_limited, t2s_full, s0_full = me.fit_decay(
        data, tes, mask, adaptive_mask, fittype
    )
    assert t2s_limited is not None
    assert s0_limited is not None
    assert t2s_full is not None
    assert s0_full is not None


def test_smoke_fit_decay_ts():
    """
    Test_smoke_fit_decay_ts tests that the function fit_decay_ts returns reasonable.

    objects with random inputs in the correct format.
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    tes = np.random.random(n_echos).tolist()
    mask = np.ones(n_samples, dtype=int)
    mask[n_samples // 2 :] = 0
    adaptive_mask = np.random.randint(2, n_echos, size=(n_samples)) * mask
    fittype = "loglin"
    t2s_limited_ts, s0_limited_ts, t2s_full_ts, s0_full_ts = me.fit_decay_ts(
        data, tes, mask, adaptive_mask, fittype
    )
    assert t2s_limited_ts is not None
    assert s0_limited_ts is not None
    assert t2s_full_ts is not None
    assert s0_full_ts is not None


def test_smoke_fit_decay_curvefit_ts():
    """
    Test_smoke_fit_decay_ts tests that the function fit_decay_ts returns reasonable.

    objects with random inputs in the correct format when using the direct.

    monoexponetial approach.
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    tes = np.random.random(n_echos).tolist()
    mask = np.ones(n_samples, dtype=int)
    mask[n_samples // 2 :] = 0
    adaptive_mask = np.random.randint(2, n_echos, size=(n_samples)) * mask
    fittype = "curvefit"
    t2s_limited_ts, s0_limited_ts, t2s_full_ts, s0_full_ts = me.fit_decay_ts(
        data, tes, mask, adaptive_mask, fittype
    )
    assert t2s_limited_ts is not None
    assert s0_limited_ts is not None
    assert t2s_full_ts is not None
    assert s0_full_ts is not None


# TODO: BREAK AND UNIT TESTS
