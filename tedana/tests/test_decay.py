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
    r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar = me.fit_decay(
        masked_data,
        testdata1["tes"],
        masked_adaptive_mask,
        testdata1["fittype"],
    )
    assert r2s.ndim == 1
    assert s0.ndim == 1
    assert failures is None
    assert r2s_var is None
    assert s0_var is None
    assert r2s_s0_covar is None


def test_fit_decay_ts(testdata1):
    """Fit_decay_ts should return data in samples x time shape."""
    masked_data = testdata1["data"][testdata1["mask"], ...]
    masked_adaptive_mask = testdata1["adaptive_mask"][testdata1["mask"]]
    r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar = me.fit_decay_ts(
        data=masked_data,
        tes=testdata1["tes"],
        adaptive_mask=masked_adaptive_mask,
        fittype=testdata1["fittype"],
    )
    assert r2s.ndim == 2
    assert s0.ndim == 2
    assert failures is None
    assert r2s_var is None
    assert s0_var is None
    assert r2s_s0_covar is None


def test__apply_r2s_ceiling():
    """_apply_r2s_ceiling caps large R2* values to prevent exp underflow in optimal combination."""
    n_voxels, n_echos, n_trs = 100, 5, 25
    echo_times = np.array([0.002, 0.023, 0.054, 0.075, 0.096])  # seconds
    me_data = np.random.random((n_voxels, n_echos, n_trs))
    r2s = np.ones(n_voxels) * 10.0  # typical R2* ~10 s⁻¹
    r2s[0] = 1e15  # extreme value → exp underflow → alpha=0 → ZeroDivisionError

    # First establish that the failure mode exists
    with pytest.raises(ZeroDivisionError):
        _ = combine._combine_r2s(me_data, echo_times[None, :], r2s[:, None])

    # Now apply the ceiling and verify it fixes the problem
    r2s_corrected = me._apply_r2s_ceiling(r2s, echo_times)
    assert r2s_corrected[0] != r2s[0]  # first value was corrected
    assert np.array_equal(r2s_corrected[1:], r2s[1:])  # others unchanged

    combined = combine._combine_r2s(me_data, echo_times[None, :], r2s_corrected[:, None])
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
    r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar = me.fit_decay(
        masked_data,
        tes,
        masked_adaptive_mask,
        fittype,
    )
    assert r2s.ndim == 1
    assert s0.ndim == 1
    assert failures is None
    assert r2s_var is None
    assert s0_var is None
    assert r2s_s0_covar is None


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
    r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar = me.fit_decay(
        masked_data,
        tes,
        masked_adaptive_mask,
        fittype,
    )
    assert r2s.ndim == 1
    assert s0.ndim == 1
    assert failures.ndim == 1
    assert r2s_var.ndim == 1
    assert s0_var.ndim == 1
    assert r2s_s0_covar.ndim == 1


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
    r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar = me.fit_decay_ts(
        masked_data,
        tes,
        masked_adaptive_mask,
        fittype,
    )
    assert r2s.ndim == 2
    assert s0.ndim == 2
    assert failures is None
    assert r2s_var is None
    assert s0_var is None
    assert r2s_s0_covar is None


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
    r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar = me.fit_decay_ts(
        masked_data,
        tes,
        masked_adaptive_mask,
        fittype,
    )
    assert r2s.ndim == 2
    assert s0.ndim == 2
    assert failures.ndim == 2
    assert r2s_var.ndim == 2
    assert s0_var.ndim == 2
    assert r2s_s0_covar.ndim == 2


# TODO: BREAK AND UNIT TESTS
