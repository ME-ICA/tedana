"""Tests for tedana.decay."""

import os.path as op
import sys
import types

import nibabel as nb
import numpy as np
import pytest

from tedana import combine
from tedana import decay as me
from tedana import io, utils
from tedana.tests.utils import get_test_data_path


def test_backend_validation_and_auto_fallback(monkeypatch, caplog):
    """The optional CUDA backend must not affect the default CPU installation."""
    with pytest.raises(ValueError, match="backend must be one of"):
        utils._resolve_backend("not-a-backend")

    monkeypatch.setitem(sys.modules, "cupy", None)
    assert utils._resolve_backend("auto") == "cpu"
    assert "falling back to CPU" in caplog.text


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


def test__fit_complex_decay_1d_optimizer_failure(monkeypatch):
    """On optimizer error, falls back to the initial estimate with success=False."""
    import scipy.optimize

    def boom(*args, **kwargs):  # noqa: U100
        raise RuntimeError("forced failure")

    monkeypatch.setattr(scipy.optimize, "least_squares", boom)
    tes = np.array([0.008, 0.020, 0.032])
    signal = me.complex_decay_model(tes, 100.0 + 0j, 20.0, 0.0)
    lower = np.array([-np.inf, 0.0, -np.inf, -np.inf])
    upper = np.array([np.inf, np.inf, np.inf, np.inf])
    out = me._fit_complex_decay_1d(signal, tes, lower_bounds=lower, upper_bounds=upper)
    assert out is not None
    assert out["success"] is False


def test_fit_complex_monoexponential_recovers_maps():
    """fit_complex_monoexponential recovers R2*/S0 over the adaptive mask."""
    rng = np.random.default_rng(0)
    tes = np.array([0.008, 0.020, 0.032, 0.044, 0.056])
    n_vox, n_echo, n_vol = 6, 5, 3
    r2star_true = rng.uniform(10, 30, n_vox)
    s0_true = rng.uniform(100, 300, n_vox) * np.exp(1j * rng.uniform(-1, 1, n_vox))
    data = np.zeros((n_vox, n_echo, n_vol), dtype=np.complex128)
    for v in range(n_vox):
        sig = me.complex_decay_model(tes, s0_true[v], r2star_true[v], 2.0)
        data[v] = np.repeat(sig[:, None], n_vol, axis=1)
    adaptive_mask = np.full(n_vox, n_echo, dtype=int)
    out = me.fit_complex_monoexponential(data, tes, adaptive_mask, report=False)
    assert out["t2s"].shape == (n_vox,)
    assert np.allclose(out["t2s"], 1.0 / r2star_true, rtol=1e-2)
    assert np.allclose(out["s0"], np.abs(s0_true), rtol=1e-2)
    assert out["frequency_hz"].shape == (n_vox,)
    assert not out["failures"].any()


def test__fit_complex_decay_joint_shared_params():
    """Joint fit recovers shared R2*/frequency with per-volume S0."""
    tes = np.array([0.008, 0.020, 0.032, 0.044, 0.056])
    r2star, freq = 15.0, 3.0
    s0_per_vol = np.array([100.0, 150.0, 200.0]) * np.exp(1j * np.array([0.2, -0.4, 0.6]))
    signal = np.stack(
        [me.complex_decay_model(tes, s0, r2star, freq) for s0 in s0_per_vol]
    )  # (T, E)
    out = me._fit_complex_decay_joint(signal, tes)
    assert out is not None
    assert np.isclose(out["r2star"], r2star, atol=1e-3)
    assert np.isclose(out["frequency_hz"], freq, atol=1e-3)
    assert out["s0"].shape == (3,)
    assert np.allclose(np.abs(out["s0"]), np.abs(s0_per_vol), rtol=1e-3)


def test__solve_complex_s0_all_volumes():
    """_solve_complex_s0 returns one complex S0 per volume."""
    tes = np.array([0.01, 0.02, 0.03])
    s0_per_vol = np.array([100.0 + 0j, 200.0 + 0j])
    signal = np.stack([me.complex_decay_model(tes, s0, 20.0, 0.0) for s0 in s0_per_vol])
    out = me._solve_complex_s0(signal, tes, 20.0, 0.0)
    assert out.shape == (2,)
    assert np.allclose(np.abs(out), np.abs(s0_per_vol), rtol=1e-6)


@pytest.fixture
def complex_testdata():
    rng = np.random.default_rng(1)
    tes = np.array([0.008, 0.020, 0.032, 0.044, 0.056])
    n_vox, n_echo, n_vol = 5, 5, 4
    r2star = rng.uniform(10, 25, n_vox)
    s0 = rng.uniform(100, 250, n_vox)
    mag = np.zeros((n_vox, n_echo, n_vol))
    phase = np.zeros((n_vox, n_echo, n_vol))
    for v in range(n_vox):
        sig = me.complex_decay_model(tes, s0[v] * np.exp(1j * 0.3), r2star[v], 2.0)
        mag[v] = np.repeat(np.abs(sig)[:, None], n_vol, axis=1)
        phase[v] = np.repeat(np.angle(sig)[:, None], n_vol, axis=1)
    adaptive_mask = np.full(n_vox, n_echo, dtype=int)
    return dict(
        mag=mag,
        phase=phase,
        tes=tes,
        adaptive_mask=adaptive_mask,
        r2star=r2star,
        s0=s0,
        n_vol=n_vol,
    )


def test_fit_complex_decay_all(complex_testdata):
    d = complex_testdata
    out = me.fit_complex_decay(d["mag"], d["phase"], d["tes"], d["adaptive_mask"], "all")
    assert out["t2s"].shape == (5,)
    assert out["s0"].shape == (5,)
    assert np.allclose(out["t2s"], 1.0 / d["r2star"], rtol=1e-2)


def test_fit_complex_decay_ts(complex_testdata):
    d = complex_testdata
    out = me.fit_complex_decay(d["mag"], d["phase"], d["tes"], d["adaptive_mask"], "ts")
    assert out["t2s"].shape == (5, d["n_vol"])
    assert out["s0"].shape == (5, d["n_vol"])


def test_fit_complex_decay_varys0(complex_testdata):
    d = complex_testdata
    out = me.fit_complex_decay(d["mag"], d["phase"], d["tes"], d["adaptive_mask"], "varys0")
    assert out["t2s"].shape == (5,)
    assert out["frequency_hz"].shape == (5,)
    assert out["s0"].shape == (5, d["n_vol"])
    assert out["phase0"].shape == (5, d["n_vol"])
    assert np.allclose(out["t2s"], 1.0 / d["r2star"], rtol=1e-2)


def test_fit_complex_decay_varys0_use_volumes(complex_testdata):
    """use_volumes restricts the shared fit but S0 is returned for all volumes."""
    d = complex_testdata
    use_volumes = np.array([True, True, False, True])
    out = me.fit_complex_decay(
        d["mag"],
        d["phase"],
        d["tes"],
        d["adaptive_mask"],
        "varys0",
        use_volumes=use_volumes,
    )
    assert out["s0"].shape == (5, d["n_vol"])
    assert np.isfinite(out["s0"][:, 2]).all()  # excluded volume still gets S0


def test_fit_complex_decay_invalid_fitmode(complex_testdata):
    """fit_complex_decay raises ValueError on an unknown fitmode."""
    d = complex_testdata
    with pytest.raises(ValueError, match="Unknown fitmode"):
        me.fit_complex_decay(d["mag"], d["phase"], d["tes"], d["adaptive_mask"], "bogus")


def test_fit_complex_monoexponential_caps_extreme_t2star():
    """R2* is bounded so a no-decay signal cannot produce an infinite T2*.

    With R2* unbounded below at 0, a flat (R2*=0) decay drives T2* = 1/R2* toward
    infinity. The fit must instead pin R2* at MIN_R2STAR, keeping T2* finite and
    at or below MAX_PHYSIOLOGICAL_T2STAR.
    """
    tes = np.array([0.008, 0.020, 0.032, 0.044, 0.056])
    n_vol = 3
    # No decay across echoes -> the optimizer wants R2* = 0 (T2* = inf).
    sig = me.complex_decay_model(tes, 200.0 * np.exp(1j * 0.3), 0.0, 1.0)
    data = np.repeat(sig[:, None], n_vol, axis=1)[None]  # (1, E, T)
    adaptive_mask = np.array([5])
    out = me.fit_complex_monoexponential(data, tes, adaptive_mask, report=False)
    assert np.isfinite(out["t2s"]).all()
    assert (out["t2s"] <= me.MAX_PHYSIOLOGICAL_T2STAR + 1e-6).all()
    assert (out["r2star"] >= me.MIN_R2STAR - 1e-9).all()


def test_fit_complex_decay_varys0_caps_extreme_t2star():
    """The varys0 driver bounds R2* so a no-decay signal cannot yield infinite T2*."""
    tes = np.array([0.008, 0.020, 0.032, 0.044, 0.056])
    n_vox, n_echo, n_vol = 2, 5, 3
    mag = np.zeros((n_vox, n_echo, n_vol))
    phase = np.zeros((n_vox, n_echo, n_vol))
    for v in range(n_vox):
        for t in range(n_vol):
            sig = me.complex_decay_model(tes, (100.0 + 50 * t) * np.exp(1j * 0.3), 0.0, 1.0)
            mag[v, :, t] = np.abs(sig)
            phase[v, :, t] = np.angle(sig)
    adaptive_mask = np.full(n_vox, n_echo, dtype=int)
    out = me.fit_complex_decay(mag, phase, tes, adaptive_mask, "varys0")
    assert np.isfinite(out["t2s"]).all()
    assert (out["t2s"] <= me.MAX_PHYSIOLOGICAL_T2STAR + 1e-6).all()


def test_modify_t2s_s0_maps_caps_full_map():
    """The full T2* map is capped, not just the limited map.

    A blown-up finite estimate (as produced by an unbounded fit) must be brought
    down in the full map, instead of leaking through to be saved (and overflow to
    +Inf when written as float32).
    """
    tes = np.array([0.01, 0.02, 0.03])
    n = 100
    t2s = np.full(n, 0.05)
    t2s[0] = 1e38  # blown-up value, finite in float64 but absurd
    s0 = np.full(n, 100.0)
    adaptive_mask = np.full(n, len(tes), dtype=int)
    t2s_full, _, _, _ = me.modify_t2s_s0_maps(t2s.copy(), s0.copy(), adaptive_mask, tes)
    assert np.isfinite(t2s_full).all()
    assert t2s_full.max() < 1e6


def test_rmse_of_fit_decay_ts_varys0():
    """rmse handles 3D t2s with 4D s0 for fitmode='varys0'."""
    rng = np.random.default_rng(2)
    tes = [0.01, 0.02, 0.03]
    n_vox, n_echo, n_vol = 4, 3, 5
    data = rng.uniform(50, 200, (n_vox, n_echo, n_vol))
    adaptive_mask = np.full(n_vox, n_echo, dtype=int)
    t2s = np.full(n_vox, 0.04)  # (Mb,)
    s0 = np.full((n_vox, n_vol), 150.0)  # (Mb, T)
    rmse_map, rmse_df = me.rmse_of_fit_decay_ts(
        data=data, tes=tes, adaptive_mask=adaptive_mask, t2s=t2s, s0=s0, fitmode="varys0"
    )
    assert rmse_map.shape == (n_vox,)
    assert len(rmse_df) == n_vol


def _cuda_available():
    try:
        return utils._resolve_backend("cuda") == "cuda"
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA backend unavailable")


@pytest.fixture
def noisy_complex_testdata():
    """Noisy complex data with per-volume S0 and a mixed adaptive mask."""
    rng = np.random.default_rng(3)
    tes = np.array([0.012, 0.028, 0.044, 0.060])
    n_vox, n_vol = 40, 6
    r2star = rng.uniform(15, 50, n_vox)
    freq = rng.uniform(-8, 8, n_vox)
    # S0 phase is shared across volumes so that every fit mode is well posed.
    s0 = (
        rng.uniform(500, 1500, n_vox)[:, None]
        * rng.uniform(0.9, 1.1, (n_vox, n_vol))
        * np.exp(1j * rng.uniform(-3, 3, n_vox))[:, None]
    )
    data = s0[:, None, :] * np.exp((-r2star + 2j * np.pi * freq)[:, None, None] * tes[:, None])
    data += rng.normal(scale=5, size=data.shape) + 1j * rng.normal(scale=5, size=data.shape)
    adaptive_mask = rng.choice([1, 2, 3, 4], size=n_vox, p=[0.1, 0.2, 0.2, 0.5])
    return dict(
        mag=np.abs(data),
        phase=np.angle(data),
        tes=tes,
        adaptive_mask=adaptive_mask,
        r2star=r2star,
        freq=freq,
    )


def test__fit_complex_groups_batched_matches_scipy_joint_fit(noisy_complex_testdata):
    """The vectorized solver reaches the same optimum as the SciPy varys0 fit."""
    d = noisy_complex_testdata
    complex_data = d["mag"] * np.exp(1j * d["phase"])
    signal = complex_data.transpose(0, 2, 1)  # (B, T, E)
    use_volumes = np.array([True, True, False, True, True, True])
    r2star, freq, s0, converged = me._fit_complex_groups_batched(
        np, signal, d["tes"], fit_groups=use_volumes
    )
    assert converged.all()
    assert s0.shape == signal.shape[:2]
    for i_vox in range(signal.shape[0]):
        ref = me._fit_complex_decay_joint(
            signal[i_vox, use_volumes], d["tes"], min_r2star=me.MIN_R2STAR
        )
        assert np.isclose(r2star[i_vox], ref["r2star"], rtol=1e-5)
        assert np.isclose(freq[i_vox], ref["frequency_hz"], rtol=1e-5, atol=1e-5)
        ref_s0 = me._solve_complex_s0(signal[i_vox], d["tes"], ref["r2star"], ref["frequency_hz"])
        assert np.allclose(s0[i_vox], ref_s0, rtol=1e-5)


def test__fit_complex_groups_batched_respects_r2star_bound():
    """A flat decay is pinned at MIN_R2STAR and still reported as converged."""
    tes = np.array([0.008, 0.020, 0.032, 0.044, 0.056])
    signal = me.complex_decay_model(tes, 200.0 * np.exp(1j * 0.3), 0.0, 1.0)[None, None, :]
    r2star, freq, _, converged = me._fit_complex_groups_batched(np, signal, tes)
    assert converged.all()
    assert np.isclose(r2star[0], me.MIN_R2STAR)
    assert np.isclose(freq[0], 1.0, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("fitmode", ["all", "ts", "varys0"])
def test_fit_complex_decay_cuda_matches_cpu(noisy_complex_testdata, fitmode):
    """The CUDA backend reproduces the SciPy estimates for every complex fit mode."""
    d = noisy_complex_testdata
    args = (d["mag"], d["phase"], d["tes"], d["adaptive_mask"], fitmode)
    cpu = me.fit_complex_decay(*args, backend="cpu")
    gpu = me.fit_complex_decay(*args, backend="cuda")
    _assert_complex_fits_match(gpu, cpu)


def _assert_complex_fits_match(test, reference):
    assert np.array_equal(test["failures"], reference["failures"])
    for key in ("t2s", "s0", "frequency_hz"):
        assert np.allclose(test[key], reference[key], rtol=1e-4, atol=1e-6), key
    phase_diff = np.angle(np.exp(1j * (test["phase0"] - reference["phase0"])))
    assert np.allclose(phase_diff, 0, atol=1e-4)


@pytest.fixture
def numpy_as_cupy(monkeypatch):
    """Run the CUDA code paths on the CPU by standing NumPy in for CuPy."""
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.__getattr__ = lambda name: getattr(np, name)
    fake_cupy.asnumpy = np.asarray
    fake_cupy.cuda = types.SimpleNamespace(
        Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None))
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(utils, "_resolve_backend", lambda backend: backend)


@pytest.mark.usefixtures("numpy_as_cupy")
@pytest.mark.parametrize("fitmode", ["all", "ts", "varys0"])
def test_fit_complex_decay_cuda_path_on_cpu(noisy_complex_testdata, monkeypatch, fitmode):
    """The CUDA driver code, including its SciPy fallback, matches the CPU backend.

    An all-NaN voxel cannot be fit on the device, so it exercises the fallback
    to SciPy, which then reports it as a failure.
    """
    d = noisy_complex_testdata
    mag, phase = d["mag"].copy(), d["phase"].copy()
    mag[0] = np.nan
    if fitmode == "all":
        # One problem per batch, so the all-NaN voxel forms a batch with no valid data.
        monkeypatch.setenv("TEDANA_CUDA_BATCH_SIZE", "1")
    args = (mag, phase, d["tes"], d["adaptive_mask"], fitmode)
    cpu = me.fit_complex_decay(*args, n_threads=None, backend="cpu")
    cuda = me.fit_complex_decay(*args, backend="cuda")
    assert np.all(cuda["failures"][0])
    _assert_complex_fits_match(cuda, cpu)


@pytest.mark.usefixtures("numpy_as_cupy")
def test_cuda_batch_size_must_be_positive(complex_testdata, monkeypatch):
    """An invalid TEDANA_CUDA_BATCH_SIZE is rejected."""
    d = complex_testdata
    monkeypatch.setenv("TEDANA_CUDA_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="TEDANA_CUDA_BATCH_SIZE"):
        me.fit_complex_decay(
            d["mag"], d["phase"], d["tes"], d["adaptive_mask"], "all", backend="cuda"
        )


def test__initial_complex_decay_params_single_echo():
    """With one echo, R2* and frequency default to 0 and S0 comes from the echo."""
    signal = np.array([50.0 * np.exp(1j * 0.4)])
    params = me._initial_complex_decay_params(signal, np.array([0.01]))
    assert np.allclose(params, [np.log(50.0), 0.0, 0.0, 0.4])


def test__solve_complex_s0_unsolvable_volumes():
    """S0 is NaN for volumes with too few echoes or a vanishing decay basis."""
    tes = np.array([0.01, 0.02, 0.03])
    signal = np.stack([me.complex_decay_model(tes, 100.0 + 0j, 20.0, 0.0)] * 2)
    signal[1, :2] = np.nan
    out = me._solve_complex_s0(signal, tes, 20.0, 0.0)
    assert np.isfinite(out[0]) and np.isnan(out[1])
    # exp(-1e6 * TE) underflows to zero, so no volume can be solved.
    assert np.isnan(me._solve_complex_s0(signal, tes, 1e6, 0.0)).all()


def test__fit_complex_decay_joint_unfittable(monkeypatch):
    """The joint fit returns None without usable volumes or when the optimizer errors."""
    import scipy.optimize

    tes = np.array([0.008, 0.020, 0.032])
    assert me._fit_complex_decay_joint(np.full((2, 3), np.nan + 0j), tes) is None

    def boom(*args, **kwargs):  # noqa: U100
        raise RuntimeError("forced failure")

    monkeypatch.setattr(scipy.optimize, "least_squares", boom)
    signal = np.stack([me.complex_decay_model(tes, 100.0 + 0j, 20.0, 0.0)] * 2)
    assert me._fit_complex_decay_joint(signal, tes) is None


def test_t2smap_subworkflow_nlls_requires_phase():
    """The nlls branch of the subworkflow needs phase data."""
    data = np.ones((2, 3, 1))
    with pytest.raises(ValueError, match="requires phase_cat"):
        me.t2smap_subworkflow(
            data_cat=data,
            tes=[0.01, 0.02, 0.03],
            fittype="nlls",
            fitmode="all",
            mask_denoise=np.ones(2, dtype=bool),
            masksum_denoise=np.full(2, 3),
            interpolate_failing_voxels=False,
            io_generator=None,
        )
