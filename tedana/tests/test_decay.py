"""Tests for tedana.decay."""

import os.path as op

import nibabel as nb
import numpy as np
import pytest

from tedana import combine
from tedana import decay as me
from tedana import io, utils
from tedana.tests.utils import get_test_data_path


@pytest.fixture(scope="module")
def testdata1():
    tes = np.array([0.0145, 0.0385, 0.0625])
    in_files = [op.join(get_test_data_path(), f"echo{i + 1}.nii.gz") for i in range(3)]
    mask_file = op.join(get_test_data_path(), "mask.nii.gz")
    data = io.load_data_nilearn(in_files, mask_img=nb.load(mask_file), n_echos=len(tes))
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        methods=["dropout", "decay"],
    )
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
    masked_data = testdata1["data"][testdata1["mask"], ...]
    masked_adaptive_mask = testdata1["adaptive_mask"][testdata1["mask"]]
    t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar = me.fit_decay(
        masked_data,
        testdata1["tes"],
        masked_adaptive_mask,
        testdata1["fittype"],
    )
    assert t2s.ndim == 1
    assert s0.ndim == 1
    assert failures is None
    assert t2s_var is None
    assert s0_var is None
    assert t2s_s0_covar is None


def test_fit_decay_ts(testdata1):
    """Fit_decay_ts should return data in samples x time shape."""
    masked_data = testdata1["data"][testdata1["mask"], ...]
    masked_adaptive_mask = testdata1["adaptive_mask"][testdata1["mask"]]
    t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar = me.fit_decay_ts(
        data=masked_data,
        tes=testdata1["tes"],
        adaptive_mask=masked_adaptive_mask,
        fittype=testdata1["fittype"],
    )
    assert t2s.ndim == 2
    assert s0.ndim == 2
    assert failures is None
    assert t2s_var is None
    assert s0_var is None
    assert t2s_s0_covar is None


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
    masked_data = data[mask, ...]
    masked_adaptive_mask = adaptive_mask[mask]
    t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar = me.fit_decay(
        masked_data,
        tes,
        masked_adaptive_mask,
        fittype,
    )
    assert t2s.ndim == 1
    assert s0.ndim == 1
    assert failures is None
    assert t2s_var is None
    assert s0_var is None
    assert t2s_s0_covar is None


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
    masked_data = data[mask, ...]
    masked_adaptive_mask = adaptive_mask[mask]
    t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar = me.fit_decay(
        masked_data,
        tes,
        masked_adaptive_mask,
        fittype,
    )
    assert t2s.ndim == 1
    assert s0.ndim == 1
    assert failures.ndim == 1
    assert t2s_var.ndim == 1
    assert s0_var.ndim == 1
    assert t2s_s0_covar.ndim == 1


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
    masked_data = data[mask, ...]
    masked_adaptive_mask = adaptive_mask[mask]
    t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar = me.fit_decay_ts(
        masked_data,
        tes,
        masked_adaptive_mask,
        fittype,
    )
    assert t2s.ndim == 2
    assert s0.ndim == 2
    assert failures is None
    assert t2s_var is None
    assert s0_var is None
    assert t2s_s0_covar is None


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
    masked_data = data[mask, ...]
    masked_adaptive_mask = adaptive_mask[mask]
    t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar = me.fit_decay_ts(
        masked_data,
        tes,
        masked_adaptive_mask,
        fittype,
    )
    assert t2s.ndim == 2
    assert s0.ndim == 2
    assert failures.ndim == 2
    assert t2s_var.ndim == 2
    assert s0_var.ndim == 2
    assert t2s_s0_covar.ndim == 2


# TODO: BREAK AND UNIT TESTS


def test_complex_decay_model():
    """complex_decay_model returns the analytic complex signal."""
    tes = np.array([0.01, 0.02, 0.03])
    s0 = 100.0 * np.exp(1j * 0.5)
    r2star = 20.0
    freq = 3.0
    sig = me.complex_decay_model(tes, s0, r2star, freq)
    expected = s0 * np.exp((-r2star + 1j * 2 * np.pi * freq) * tes)
    assert np.allclose(sig, expected)


def test__fit_complex_decay_1d_recovers_params():
    """_fit_complex_decay_1d recovers known params from noiseless data."""
    tes = np.array([0.008, 0.020, 0.032, 0.044, 0.056])
    s0 = 250.0 * np.exp(1j * 0.7)
    r2star = 18.0
    freq = 4.0
    signal = me.complex_decay_model(tes, s0, r2star, freq)
    lower = np.array([-np.inf, 0.0, -np.inf, -np.inf])
    upper = np.array([np.inf, np.inf, np.inf, np.inf])
    out = me._fit_complex_decay_1d(signal, tes, lower_bounds=lower, upper_bounds=upper)
    assert out is not None
    assert np.isclose(out["r2star"], r2star, atol=1e-3)
    assert np.isclose(out["frequency_hz"], freq, atol=1e-3)
    assert np.isclose(np.abs(out["s0"]), np.abs(s0), rtol=1e-3)
    assert out["success"]


def test__fit_complex_decay_1d_too_few_echoes():
    """_fit_complex_decay_1d returns None with fewer than 2 finite echoes."""
    tes = np.array([0.01, 0.02])
    signal = np.array([np.nan + 1j * np.nan, 5 + 1j * 2])
    lower = np.array([-np.inf, 0.0, -np.inf, -np.inf])
    upper = np.array([np.inf, np.inf, np.inf, np.inf])
    assert me._fit_complex_decay_1d(signal, tes, lower_bounds=lower, upper_bounds=upper) is None
