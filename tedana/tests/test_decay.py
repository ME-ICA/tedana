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


def _make_decaying_data(adaptive_mask, tes, n_times=5, seed=0):
    """Synthesize multi-echo data that follows a clean monoexponential decay.

    Used so that curve_fit converges and produces finite covariances.
    """
    rng = np.random.RandomState(seed)
    n_samples = adaptive_mask.shape[0]
    n_echos = len(tes)
    tes = np.asarray(tes)
    s0_true = rng.uniform(800, 1200, size=n_samples)
    t2s_true = rng.uniform(0.02, 0.06, size=n_samples)
    data = np.zeros((n_samples, n_echos, n_times))
    for echo in range(n_echos):
        signal = s0_true * np.exp(-tes[echo] / t2s_true)
        data[:, echo, :] = signal[:, None]
    data += rng.normal(scale=1.0, size=data.shape)
    return data


def test_fit_decay_curvefit_populates_variance_at_adaptive_mask_one():
    """Voxels with adaptive_mask == 1 should get real variance/covariance values.

    They are fit with two echoes (like adaptive_mask == 2 voxels), so their
    curve_fit covariances must be carried into the full maps rather than dropped
    (left as zero), which would render as empty voxels in the report figures.
    """
    tes = [0.0145, 0.0385, 0.0625]
    n_samples = 60
    adaptive_mask = np.full(n_samples, 3, dtype=int)
    adaptive_mask[:20] = 1  # voxels with only one good echo
    data = _make_decaying_data(adaptive_mask, tes)

    t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar = me.fit_decay(
        data, tes, adaptive_mask, "curvefit"
    )

    am1 = adaptive_mask == 1
    converged = am1 & ~failures
    assert converged.any(), "Expected at least one converged adaptive_mask == 1 voxel"
    # Converged adaptive_mask == 1 voxels must have real (nonzero) variances,
    # matching the fact that their T2*/S0 estimates are also populated.
    assert np.all(t2s_var[converged] != 0)
    assert np.all(s0_var[converged] != 0)
    assert np.all(t2s_s0_covar[converged] != 0)


def test_rmse_includes_adaptive_mask_one():
    """rmse_of_fit_decay_ts should return finite RMSE for adaptive_mask == 1 voxels."""
    tes = [0.0145, 0.0385, 0.0625]
    n_samples = 60
    n_times = 5
    adaptive_mask = np.full(n_samples, 3, dtype=int)
    adaptive_mask[:20] = 1
    data = _make_decaying_data(adaptive_mask, tes, n_times=n_times)

    t2s, s0, _, _, _, _ = me.fit_decay(data, tes, adaptive_mask, "curvefit")

    rmse_map, rmse_df = me.rmse_of_fit_decay_ts(
        data=data,
        tes=tes,
        adaptive_mask=adaptive_mask,
        t2s=t2s,
        s0=s0,
        fitmode="all",
    )
    am1 = adaptive_mask == 1
    assert np.all(np.isfinite(rmse_map[am1]))


def test_generate_decay_metrics_basic():
    # 5 base-mask voxels; voxel 0 has 0 good echoes (outside fit mask).
    adaptive_mask = np.array([0, 1, 2, 3, 3])
    t2star = np.array([np.nan, 20.0, 40.0, 60.0, 80.0])
    s0 = np.array([np.nan, 100.0, 200.0, 300.0, 400.0])
    rmse_map = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])

    metrics = me.generate_decay_metrics(
        t2star=t2star,
        s0=s0,
        rmse_map=rmse_map,
        adaptive_mask=adaptive_mask,
        n_fit_failures=2,
        n_fit_failures_after_interpolation=1,
    )

    assert metrics["n_voxels_base_mask"] == 5
    assert metrics["n_voxels_fit_mask"] == 4
    assert metrics["good_echo_voxel_counts"] == {1: 1, 2: 1, 3: 2}
    # Means/medians computed over the 4 fit-mask voxels only.
    assert metrics["t2star_mean"] == 50.0
    assert metrics["t2star_median"] == 50.0
    assert metrics["rmse_median"] == 2.5
    assert metrics["n_fit_failures"] == 2
    assert metrics["n_fit_failures_after_interpolation"] == 1


def test_generate_decay_metrics_omits_failures_when_none():
    metrics = me.generate_decay_metrics(
        t2star=np.array([10.0, 20.0]),
        s0=np.array([100.0, 200.0]),
        rmse_map=np.array([1.0, 2.0]),
        adaptive_mask=np.array([1, 2]),
    )
    assert "n_fit_failures" not in metrics
    assert "n_fit_failures_after_interpolation" not in metrics


def test_generate_decay_metrics_rejects_2d_input():
    """2D (voxels x time) maps (e.g. fitmode == "ts") must raise, not silently flatten."""
    t2star_2d = np.ones((4, 3))
    with pytest.raises(ValueError, match="1D"):
        me.generate_decay_metrics(
            t2star=t2star_2d,
            s0=np.ones((4, 3)),
            rmse_map=np.ones(4),
            adaptive_mask=np.array([1, 2, 3, 3]),
        )


# TODO: BREAK AND UNIT TESTS
