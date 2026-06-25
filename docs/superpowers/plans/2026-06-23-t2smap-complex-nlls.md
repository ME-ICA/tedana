# t2smap Complex NLLS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complex non-linear least-squares (NLLS) T2*/S0 fit to the `t2smap` workflow, exposed via `--fittype nlls`, a new `--phase` argument, and a new `--fitmode varys0`.

**Architecture:** Port the core complex decay estimator from the `dahnke` subpackage into `tedana/decay.py` (model, per-sample fitter, adaptive-mask driver, and joint fit), stripped of the Dahnke modulation term (implicitly 1, kept as a future hook). The `t2smap` workflow gains a dedicated nlls branch calling a new `fit_complex_decay` dispatcher; `fit_decay`/`fit_decay_ts` are left untouched so the main `tedana` workflow is unaffected. Magnitude data still drives the adaptive mask, optimal combination, and RMSE; only T2*/S0 estimation uses the complex data.

**Tech Stack:** Python, NumPy, SciPy (`scipy.optimize.least_squares`), joblib (`Parallel`/`delayed`), pytest. Reference source: `/mnt/c/Users/tsalo/Documents/tsalo/complex-tedana/dahnke/src/dahnke/correction.py`.

## Global Constraints

- Modulation is **not** ported. The model keeps a `modulation=1.0` parameter as a forward-compatibility hook only; no gradient/slice-profile code.
- The complex model is `S(TE) = S0 * exp((-R2* + 1j*2*pi*f) * TE) * modulation`, with `S0` complex; fit parameters are `[log(|S0|), R2*, frequency_hz, phase0]` and `S0 = exp(log|S0|) * exp(1j*phase0)`.
- Echo times are in **seconds**.
- `fit_decay` and `fit_decay_ts` signatures and return tuples MUST NOT change (called by `tedana/workflows/tedana.py`).
- `--fitmode varys0` is valid only with `--fittype nlls`. `--phase` is required for nlls and rejected for other fittypes. `--phase` must have one file per echo, matching `-d`.
- `varys0` permits `--exclude`; excluded volumes are dropped from the shared R2*/frequency estimate only, while per-volume S0 is computed for all volumes.
- All new public functions get NumPy-style docstrings matching the surrounding file. Follow existing logging (`LGR`, `RepLGR`) and `joblib`/`tqdm` patterns from `fit_monoexponential`.
- Run tests with the project venv: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest`.

---

### Task 1: Complex decay model, parameter initialization, and per-sample fitter

**Files:**
- Modify: `tedana/decay.py` (add functions after `_fit_single_voxel`, near line 121; `scipy`/`numpy` already imported)
- Test: `tedana/tests/test_decay.py`

**Interfaces:**
- Produces:
  - `complex_decay_model(echo_times, s0, r2star, frequency_hz=0.0, modulation=1.0) -> np.ndarray` (complex)
  - `_initial_complex_decay_params(signal, echo_times) -> np.ndarray` shape `(4,)` = `[log|S0|, R2*, frequency_hz, phase0]`
  - `_fit_complex_decay_1d(signal, echo_times, *, lower_bounds, upper_bounds, max_nfev=None) -> dict | None` with keys `s0` (complex), `r2star`, `frequency_hz`, `phase0`, `cost`, `nfev`, `success`. Returns `None` only when fewer than 2 finite echoes.

- [ ] **Step 1: Write failing tests**

Add to `tedana/tests/test_decay.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py -k complex_decay -v`
Expected: FAIL with `AttributeError: module 'tedana.decay' has no attribute 'complex_decay_model'`.

- [ ] **Step 3: Implement the functions**

Add to `tedana/decay.py` after `_fit_single_voxel` (after line 120):

```python
def complex_decay_model(echo_times, s0, r2star, frequency_hz=0.0, modulation=1.0):
    """Evaluate a single-pool complex R2* decay model.

    The model is ``S(TE) = S0 * exp((-R2* + 1j*2*pi*f) * TE) * modulation``.
    ``s0`` is complex and absorbs the initial phase. ``modulation`` is a
    forward-compatibility hook for a future Dahnke correction; pass 1 for none.

    Parameters
    ----------
    echo_times : (E,) array_like
        Echo times in seconds.
    s0 : complex or array_like
        Complex signal at TE=0.
    r2star : float or array_like
        R2* decay rate in s^-1.
    frequency_hz : float or array_like, optional
        Off-resonance frequency in Hz. Default is 0.0.
    modulation : complex or array_like, optional
        Through-slice modulation term. Default is 1.0 (no modulation).

    Returns
    -------
    signal : :obj:`numpy.ndarray`
        Complex predicted signal, broadcast over the inputs.
    """
    echo_times = np.asarray(echo_times, dtype=float)
    decay = -np.asarray(r2star) + 1j * 2.0 * np.pi * np.asarray(frequency_hz)
    return np.asarray(s0) * np.exp(decay * echo_times) * np.asarray(modulation)


def _initial_complex_decay_params(signal, echo_times):
    """Derive initial ``[log|S0|, R2*, frequency_hz, phase0]`` for one echo train.

    Parameters
    ----------
    signal : (E,) array_like
        Complex echo train for one sample.
    echo_times : (E,) array_like
        Echo times in seconds.

    Returns
    -------
    params : (4,) :obj:`numpy.ndarray`
        Initial estimates from a log-linear magnitude fit (slope -> R2*,
        intercept -> log|S0|) and an unwrapped-phase linear fit
        (slope -> frequency_hz, intercept -> phase0).
    """
    amplitude = np.maximum(np.abs(signal), np.finfo(float).tiny)
    if echo_times.size > 1:
        slope, intercept = np.polyfit(echo_times, np.log(amplitude), 1)
        r2star = max(0.0, -float(slope))
        phase = np.unwrap(np.angle(signal))
        phase_slope, phase_intercept = np.polyfit(echo_times, phase, 1)
        frequency_hz = float(phase_slope / (2.0 * np.pi))
        phase0 = float(phase_intercept)
    else:
        intercept = float(np.log(amplitude[0]))
        r2star = 0.0
        frequency_hz = 0.0
        phase0 = float(np.angle(signal[0]))
    return np.array([float(intercept), r2star, frequency_hz, phase0], dtype=float)


def _fit_complex_decay_1d(signal, echo_times, *, lower_bounds, upper_bounds, max_nfev=None):
    """Fit one complex echo train with nonlinear least squares.

    Parameters
    ----------
    signal : (E,) array_like
        Complex-valued data for one sample.
    echo_times : (E,) array_like
        Echo times in seconds.
    lower_bounds, upper_bounds : (4,) array_like
        Bounds for ``[log|S0|, R2*, frequency_hz, phase0]``.
    max_nfev : int or None, optional
        Maximum function evaluations for :func:`scipy.optimize.least_squares`.

    Returns
    -------
    result : dict or None
        Dict with ``s0`` (complex), ``r2star``, ``frequency_hz``, ``phase0``,
        ``cost``, ``nfev``, ``success``. ``None`` only when fewer than 2 finite
        echoes are available. On optimizer error, falls back to the initial
        estimate with ``success=False``.
    """
    signal = np.asarray(signal)
    echo_times = np.asarray(echo_times, dtype=float)
    valid = np.isfinite(signal.real) & np.isfinite(signal.imag)
    if int(valid.sum()) < 2:
        return None

    y_valid = signal[valid]
    te_valid = echo_times[valid]
    x0 = _initial_complex_decay_params(y_valid, te_valid)
    x0 = np.minimum(np.maximum(x0, lower_bounds), upper_bounds)

    def residuals(params):
        log_s0_abs, r2, freq, phi0 = params
        pred = complex_decay_model(te_valid, np.exp(log_s0_abs) * np.exp(1j * phi0), r2, freq)
        residual = pred - y_valid
        return np.concatenate([residual.real, residual.imag])

    try:
        result = scipy.optimize.least_squares(
            residuals, x0, bounds=(lower_bounds, upper_bounds), max_nfev=max_nfev
        )
    except (ValueError, RuntimeError, FloatingPointError):
        log_s0_abs, r2star, frequency_hz, phase0 = x0
        return {
            "s0": np.exp(log_s0_abs) * np.exp(1j * phase0),
            "r2star": r2star,
            "frequency_hz": frequency_hz,
            "phase0": phase0,
            "cost": np.nan,
            "nfev": 0,
            "success": False,
        }

    log_s0_abs, r2star, frequency_hz, phase0 = result.x
    return {
        "s0": np.exp(log_s0_abs) * np.exp(1j * phase0),
        "r2star": r2star,
        "frequency_hz": frequency_hz,
        "phase0": phase0,
        "cost": result.cost,
        "nfev": result.nfev,
        "success": result.success,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py -k complex_decay -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tedana/decay.py tedana/tests/test_decay.py
git commit -m "feat(decay): add complex decay model and per-sample NLLS fitter"
```

---

### Task 2: Adaptive-mask complex driver (`fit_complex_monoexponential`)

**Files:**
- Modify: `tedana/decay.py` (add after `_fit_complex_decay_1d`)
- Test: `tedana/tests/test_decay.py`

**Interfaces:**
- Consumes: `_fit_complex_decay_1d`, `complex_decay_model` (Task 1); pattern mirrors `fit_monoexponential` (lines 123-282) and its `echos_to_run`/`echo_masks` logic.
- Produces: `fit_complex_monoexponential(data_cat, echo_times, adaptive_mask, report=True, n_threads=1) -> dict` with `(Md,)` arrays `t2s`, `s0` (magnitude), `r2star`, `frequency_hz`, `phase0`, `failures` (bool). `data_cat` is **complex** `(Md x E x T)`; echoes and timepoints are flattened into one fit per voxel.

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py::test_fit_complex_monoexponential_recovers_maps -v`
Expected: FAIL with `AttributeError: ... has no attribute 'fit_complex_monoexponential'`.

- [ ] **Step 3: Implement**

Add to `tedana/decay.py` after `_fit_complex_decay_1d`:

```python
def fit_complex_monoexponential(data_cat, echo_times, adaptive_mask, report=True, n_threads=1):
    """Fit a complex monoexponential decay model per voxel across all timepoints.

    Echoes and timepoints are flattened into a single fit per voxel (the
    complex analog of the ``fittype="curvefit"``/``fitmode="all"`` scheme),
    using the adaptive mask to choose how many echoes each voxel uses.

    Parameters
    ----------
    data_cat : (Md x E x T) :obj:`numpy.ndarray`
        Complex multi-echo data. Md is samples in the denoising mask.
    echo_times : (E,) array_like
        Echo times in seconds.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Number of good echoes per voxel. See ``make_adaptive_mask``.
    report : bool, optional
        Whether to log a description of this step. Default is True.
    n_threads : int, optional
        Number of threads. If None or <= 0, uses all CPU cores.

    Returns
    -------
    result : dict
        ``(Md,)`` arrays ``t2s``, ``s0`` (magnitude), ``r2star``,
        ``frequency_hz``, ``phase0``, and boolean ``failures``.
    """
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    if report:
        RepLGR.info(
            "A complex monoexponential model was fit to the magnitude and "
            "phase data at each voxel using nonlinear least squares, jointly "
            "estimating T2*, S0, off-resonance frequency, and initial phase."
        )
    echo_times = np.asarray(echo_times, dtype=float)
    n_samp, _, n_vols = data_cat.shape

    echos_to_run = np.unique(adaptive_mask)
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    lower_bounds = np.array([-np.inf, 0.0, -np.inf, -np.inf])
    upper_bounds = np.array([np.inf, np.inf, np.inf, np.inf])

    r2star = np.zeros(n_samp)
    s0_mag = np.zeros(n_samp)
    frequency_hz = np.zeros(n_samp)
    phase0 = np.zeros(n_samp)
    failures = np.zeros(n_samp, dtype=bool)

    for echo_num in echos_to_run:
        if echo_num == 2:
            voxel_idx = np.where(adaptive_mask <= echo_num)[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        data_2d = data_cat[:, :echo_num, :].reshape(n_samp, -1)
        echo_times_1d = np.repeat(echo_times[:echo_num], n_vols)

        results = Parallel(n_jobs=n_threads)(
            delayed(_fit_complex_decay_1d)(
                data_2d[voxel],
                echo_times_1d,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            for voxel in tqdm(voxel_idx, desc=f"{echo_num}-echo complex NLLS")
        )

        for voxel, res in zip(voxel_idx, results):
            if res is None:
                failures[voxel] = True
                continue
            r2star[voxel] = res["r2star"]
            s0_mag[voxel] = np.abs(res["s0"])
            frequency_hz[voxel] = res["frequency_hz"]
            phase0[voxel] = res["phase0"]
            failures[voxel] = not res["success"]

    with np.errstate(divide="ignore", invalid="ignore"):
        t2s = np.where(r2star > 0, 1.0 / r2star, np.inf)

    return {
        "t2s": t2s,
        "s0": s0_mag,
        "r2star": r2star,
        "frequency_hz": frequency_hz,
        "phase0": phase0,
        "failures": failures,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py::test_fit_complex_monoexponential_recovers_maps -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tedana/decay.py tedana/tests/test_decay.py
git commit -m "feat(decay): add adaptive-mask complex monoexponential driver"
```

---

### Task 3: Joint fit (`_fit_complex_decay_joint`) and shared S0 solver

**Files:**
- Modify: `tedana/decay.py` (add after `fit_complex_monoexponential`)
- Test: `tedana/tests/test_decay.py`

**Interfaces:**
- Consumes: `complex_decay_model`, `_initial_complex_decay_params` (Task 1).
- Produces:
  - `_solve_complex_s0(signal, echo_times, r2star, frequency_hz) -> np.ndarray` complex, shape `(T,)`. `signal` is `(T, E)` complex; analytic least-squares S0 per volume given fixed R2*/frequency; NaN for volumes with < 2 finite echoes.
  - `_fit_complex_decay_joint(signal, echo_times, *, max_r2star=np.inf, max_frequency_hz=np.inf, max_nfev=None) -> dict | None` with scalar `r2star`, `frequency_hz`, `cost`, `nfev`, `success` and `(T,)` arrays `s0` (complex), `phase0`. `signal` is `(T, E)` complex.

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py -k "joint or solve_complex" -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Add to `tedana/decay.py` after `fit_complex_monoexponential`:

```python
def _solve_complex_s0(signal, echo_times, r2star, frequency_hz):
    """Analytically solve per-volume complex S0 for fixed R2*/frequency.

    Parameters
    ----------
    signal : (T, E) array_like
        Complex data for one voxel; volumes on axis 0, echoes on axis 1.
    echo_times : (E,) array_like
        Echo times in seconds.
    r2star : float
        Shared R2* in s^-1.
    frequency_hz : float
        Shared off-resonance frequency in Hz.

    Returns
    -------
    s0 : (T,) :obj:`numpy.ndarray`
        Complex S0 per volume. NaN for volumes with fewer than 2 finite echoes.
    """
    signal = np.asarray(signal)
    echo_times = np.asarray(echo_times, dtype=float)
    n_vols = signal.shape[0]
    decay = np.exp((-r2star + 1j * 2.0 * np.pi * frequency_hz) * echo_times)
    s0 = np.full(n_vols, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.isfinite(signal.real) & np.isfinite(signal.imag)
    for vol in range(n_vols):
        echo_mask = valid[vol]
        if int(echo_mask.sum()) < 2:
            continue
        basis = decay[echo_mask]
        denom = np.sum(np.abs(basis) ** 2)
        if denom <= 0 or not np.isfinite(denom):
            continue
        s0[vol] = np.sum(np.conj(basis) * signal[vol, echo_mask]) / denom
    return s0


def _fit_complex_decay_joint(
    signal, echo_times, *, max_r2star=np.inf, max_frequency_hz=np.inf, max_nfev=None
):
    """Fit one voxel with shared R2*/frequency and per-volume complex S0.

    Per-volume complex S0 is linear given fixed R2*/frequency and is solved
    analytically inside the residual, keeping the optimizer to two parameters.

    Parameters
    ----------
    signal : (T, E) array_like
        Complex data for one voxel; volumes on axis 0, echoes on axis 1.
    echo_times : (E,) array_like
        Echo times in seconds.
    max_r2star : float, optional
        Upper bound for R2* in s^-1. Default is inf.
    max_frequency_hz : float, optional
        Symmetric bound for off-resonance in Hz. Default is inf.
    max_nfev : int or None, optional
        Maximum function evaluations.

    Returns
    -------
    result : dict or None
        Scalar ``r2star``, ``frequency_hz``, ``cost``, ``nfev``, ``success``
        and ``(T,)`` arrays ``s0`` (complex) and ``phase0``. ``None`` if no
        volume has >= 2 finite echoes or optimization fails.
    """
    signal = np.asarray(signal)
    echo_times = np.asarray(echo_times, dtype=float)
    n_vols = signal.shape[0]
    valid = np.isfinite(signal.real) & np.isfinite(signal.imag)
    valid_vols = np.flatnonzero(valid.sum(axis=1) >= 2)
    if valid_vols.size == 0:
        return None

    initial = np.asarray(
        [
            _initial_complex_decay_params(signal[vol, valid[vol]], echo_times[valid[vol]])
            for vol in valid_vols
        ]
    )
    lower_bounds = np.array([0.0, -max_frequency_hz])
    upper_bounds = np.array([max_r2star, max_frequency_hz])
    x0 = np.array([float(np.nanmedian(initial[:, 1])), float(np.nanmedian(initial[:, 2]))])
    x0 = np.minimum(np.maximum(x0, lower_bounds), upper_bounds)

    def residuals(params):
        r2, freq = params
        s0_est = _solve_complex_s0(signal, echo_times, r2, freq)
        parts = []
        for vol in valid_vols:
            if not np.isfinite(s0_est[vol]):
                continue
            echo_mask = valid[vol]
            pred = complex_decay_model(echo_times[echo_mask], s0_est[vol], r2, freq)
            residual = pred - signal[vol, echo_mask]
            parts.extend([residual.real, residual.imag])
        return np.concatenate(parts)

    try:
        result = scipy.optimize.least_squares(
            residuals, x0, bounds=(lower_bounds, upper_bounds), max_nfev=max_nfev
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None

    r2star, frequency_hz = result.x
    s0 = _solve_complex_s0(signal, echo_times, r2star, frequency_hz)
    return {
        "s0": s0,
        "phase0": np.angle(s0),
        "r2star": float(r2star),
        "frequency_hz": float(frequency_hz),
        "cost": result.cost,
        "nfev": result.nfev,
        "success": result.success,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py -k "joint or solve_complex" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tedana/decay.py tedana/tests/test_decay.py
git commit -m "feat(decay): add joint complex fit and analytic S0 solver"
```

---

### Task 4: Public nlls dispatcher (`fit_complex_decay`)

**Files:**
- Modify: `tedana/decay.py` (add after `_fit_complex_decay_joint`)
- Test: `tedana/tests/test_decay.py`

**Interfaces:**
- Consumes: `fit_complex_monoexponential` (Task 2), `_fit_complex_decay_joint`, `_solve_complex_s0` (Task 3).
- Produces: `fit_complex_decay(data, phase, tes, adaptive_mask, fitmode, use_volumes=None, n_threads=1) -> dict`.
  - `data`, `phase`: real magnitude/phase `(Md x E x T)`; complex formed internally as `data * exp(1j*phase)`.
  - `fitmode="all"`: dict of `(Md,)` arrays `t2s`, `s0`, `frequency_hz`, `phase0`, `failures`.
  - `fitmode="ts"`: same keys as `(Md, T)` arrays.
  - `fitmode="varys0"`: `t2s`, `frequency_hz`, `failures` `(Md,)`; `s0`, `phase0` `(Md, T)`. `use_volumes` (bool `(T,)` or None) selects volumes used for the shared R2*/frequency estimate; per-volume S0 computed for all volumes.

- [ ] **Step 1: Write failing tests**

```python
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
    return dict(mag=mag, phase=phase, tes=tes, adaptive_mask=adaptive_mask,
               r2star=r2star, s0=s0, n_vol=n_vol)


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
        d["mag"], d["phase"], d["tes"], d["adaptive_mask"], "varys0",
        use_volumes=use_volumes,
    )
    assert out["s0"].shape == (5, d["n_vol"])
    assert np.isfinite(out["s0"][:, 2]).all()  # excluded volume still gets S0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py -k fit_complex_decay -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement**

Add to `tedana/decay.py` after `_fit_complex_decay_joint`:

```python
def fit_complex_decay(data, phase, tes, adaptive_mask, fitmode, use_volumes=None, n_threads=1):
    """Estimate complex T2*/S0 maps from magnitude and phase data.

    Parameters
    ----------
    data : (Md x E x T) :obj:`numpy.ndarray`
        Magnitude multi-echo data in the denoising mask.
    phase : (Md x E x T) :obj:`numpy.ndarray`
        Phase multi-echo data in radians, same shape as ``data``.
    tes : (E,) array_like
        Echo times in seconds.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Number of good echoes per voxel.
    fitmode : {"all", "ts", "varys0"}
        ``"all"`` fits one model per voxel across all timepoints. ``"ts"`` fits
        per voxel and timepoint. ``"varys0"`` shares R2*/frequency across
        timepoints with per-volume complex S0.
    use_volumes : (T,) :obj:`numpy.ndarray` of bool or None, optional
        For ``"varys0"`` only: volumes used to estimate the shared
        R2*/frequency. Per-volume S0 is computed for all volumes regardless.
        ``None`` uses all volumes.
    n_threads : int, optional
        Number of threads. If None or <= 0, uses all CPU cores.

    Returns
    -------
    result : dict
        For ``"all"``: ``(Md,)`` arrays ``t2s``, ``s0``, ``r2star``,
        ``frequency_hz``, ``phase0``, ``failures``. For ``"ts"``: same keys as
        ``(Md, T)``. For ``"varys0"``: ``t2s``/``r2star``/``frequency_hz``/
        ``failures`` are ``(Md,)`` and ``s0``/``phase0`` are ``(Md, T)``.
    """
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    data = np.asarray(data)
    phase = np.asarray(phase)
    tes = np.asarray(tes, dtype=float)
    complex_data = data * np.exp(1j * phase)
    n_samp, _, n_vols = complex_data.shape

    if fitmode == "all":
        return fit_complex_monoexponential(
            complex_data, tes, adaptive_mask, n_threads=n_threads
        )

    if fitmode == "ts":
        keys = ("t2s", "s0", "r2star", "frequency_hz", "phase0")
        out = {k: np.zeros((n_samp, n_vols)) for k in keys}
        out["failures"] = np.zeros((n_samp, n_vols), dtype=bool)
        report = True
        for vol in range(n_vols):
            res = fit_complex_monoexponential(
                complex_data[:, :, vol][:, :, None],
                tes,
                adaptive_mask,
                report=report,
                n_threads=n_threads,
            )
            for k in keys:
                out[k][:, vol] = res[k]
            out["failures"][:, vol] = res["failures"]
            report = False
        return out

    if fitmode == "varys0":
        if use_volumes is None:
            use_volumes = np.ones(n_vols, dtype=bool)
        return _fit_complex_decay_varys0(
            complex_data, tes, adaptive_mask, use_volumes, n_threads
        )

    raise ValueError(f"Unknown fitmode option: {fitmode}")


def _fit_complex_decay_varys0(complex_data, tes, adaptive_mask, use_volumes, n_threads):
    """Joint-fit driver: shared R2*/frequency, per-volume complex S0.

    Parameters
    ----------
    complex_data : (Md x E x T) :obj:`numpy.ndarray`
        Complex multi-echo data.
    tes : (E,) :obj:`numpy.ndarray`
        Echo times in seconds.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Number of good echoes per voxel.
    use_volumes : (T,) :obj:`numpy.ndarray` of bool
        Volumes used for the shared R2*/frequency estimate.
    n_threads : int
        Number of threads.

    Returns
    -------
    result : dict
        ``(Md,)`` ``t2s``, ``r2star``, ``frequency_hz``, ``failures`` and
        ``(Md, T)`` ``s0`` (magnitude) and ``phase0``.
    """
    n_samp, _, n_vols = complex_data.shape
    echos_to_run = np.unique(adaptive_mask)
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    r2star = np.zeros(n_samp)
    frequency_hz = np.zeros(n_samp)
    failures = np.zeros(n_samp, dtype=bool)
    s0_mag = np.zeros((n_samp, n_vols))
    phase0 = np.zeros((n_samp, n_vols))

    def _fit_voxel(voxel, echo_num):
        # (T, E) for the shared fit (used volumes only) and all volumes for S0
        signal_all = complex_data[voxel, :echo_num, :].T
        joint = _fit_complex_decay_joint(signal_all[use_volumes], tes[:echo_num])
        if joint is None:
            return voxel, None
        s0_all = _solve_complex_s0(
            signal_all, tes[:echo_num], joint["r2star"], joint["frequency_hz"]
        )
        return voxel, {"joint": joint, "s0_all": s0_all}

    for echo_num in echos_to_run:
        if echo_num == 2:
            voxel_idx = np.where(adaptive_mask <= echo_num)[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        results = Parallel(n_jobs=n_threads)(
            delayed(_fit_voxel)(voxel, echo_num)
            for voxel in tqdm(voxel_idx, desc=f"{echo_num}-echo varys0 NLLS")
        )

        for voxel, res in results:
            if res is None:
                failures[voxel] = True
                continue
            joint = res["joint"]
            r2star[voxel] = joint["r2star"]
            frequency_hz[voxel] = joint["frequency_hz"]
            failures[voxel] = not joint["success"]
            s0_all = res["s0_all"]
            s0_mag[voxel, :] = np.abs(s0_all)
            phase0[voxel, :] = np.angle(s0_all)

    with np.errstate(divide="ignore", invalid="ignore"):
        t2s = np.where(r2star > 0, 1.0 / r2star, np.inf)

    return {
        "t2s": t2s,
        "r2star": r2star,
        "frequency_hz": frequency_hz,
        "s0": s0_mag,
        "phase0": phase0,
        "failures": failures,
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py -k fit_complex_decay -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tedana/decay.py tedana/tests/test_decay.py
git commit -m "feat(decay): add fit_complex_decay nlls dispatcher (all/ts/varys0)"
```

---

### Task 5: `rmse_of_fit_decay_ts` varys0 branch

**Files:**
- Modify: `tedana/decay.py:630-701` (`rmse_of_fit_decay_ts`)
- Test: `tedana/tests/test_decay.py`

**Interfaces:**
- Consumes: existing `rmse_of_fit_decay_ts` (lines 630-736).
- Produces: `rmse_of_fit_decay_ts(..., fitmode="varys0")` handling 3D `t2s` `(Mb,)` + 4D `s0` `(Mb, T)`. (`modify_t2s_s0_maps` is unchanged: row-indexing already broadcasts a `(Md,)` t2s and `(Md, T)` s0.)

- [ ] **Step 1: Write failing test**

```python
def test_rmse_of_fit_decay_ts_varys0():
    """rmse handles 3D t2s with 4D s0 for fitmode='varys0'."""
    rng = np.random.default_rng(2)
    tes = [0.01, 0.02, 0.03]
    n_vox, n_echo, n_vol = 4, 3, 5
    data = rng.uniform(50, 200, (n_vox, n_echo, n_vol))
    adaptive_mask = np.full(n_vox, n_echo, dtype=int)
    t2s = np.full(n_vox, 0.04)            # (Mb,)
    s0 = np.full((n_vox, n_vol), 150.0)   # (Mb, T)
    rmse_map, rmse_df = me.rmse_of_fit_decay_ts(
        data=data, tes=tes, adaptive_mask=adaptive_mask, t2s=t2s, s0=s0, fitmode="varys0"
    )
    assert rmse_map.shape == (n_vox,)
    assert len(rmse_df) == n_vol
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py::test_rmse_of_fit_decay_ts_varys0 -v`
Expected: FAIL with `ValueError: Unknown fitmode option varys0`.

- [ ] **Step 3: Implement**

Update the `Literal` type hint on line 637 from `Literal["all", "ts"]` to `Literal["all", "ts", "varys0"]`, then add a branch in the loop. Change lines 685-689:

```python
        elif fitmode == "ts":
            s0_echo = s0[use_vox, :]
            t2s_echo = t2s[use_vox, :]
        elif fitmode == "varys0":
            s0_echo = s0[use_vox, :]
            t2s_echo = np.tile(t2s[use_vox][:, np.newaxis], (1, n_vols))
        else:
            raise ValueError(f"Unknown fitmode option {fitmode}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py::test_rmse_of_fit_decay_ts_varys0 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tedana/decay.py tedana/tests/test_decay.py
git commit -m "feat(decay): support fitmode='varys0' in rmse_of_fit_decay_ts"
```

---

### Task 6: Output descriptors

**Files:**
- Modify: `tedana/resources/config/outputs.json` (after the `s0 variance img` entry, ~line 25)
- Test: `tedana/tests/test_io.py` (config-load sanity) — or validate via Task 7 integration.

**Interfaces:**
- Produces config keys consumed by Task 7: `"frequency img"`, `"phase0 img"`, `"fit cost img"`, `"fit nfev img"`.

- [ ] **Step 1: Add descriptors**

Insert into `tedana/resources/config/outputs.json` after the `s0 variance img` block:

```json
    "frequency img": {
        "orig": "frequencyHz",
        "bidsv1.5.0": "frequencyHzmap"
    },
    "phase0 img": {
        "orig": "phase0",
        "bidsv1.5.0": "phase0map"
    },
    "fit cost img": {
        "orig": "fit_cost",
        "bidsv1.5.0": "desc-fitCost_statmap"
    },
    "fit nfev img": {
        "orig": "fit_nfev",
        "bidsv1.5.0": "desc-fitNfev_statmap"
    },
```

- [ ] **Step 2: Verify JSON parses and keys load**

Run:
```bash
cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -c "import json; from tedana.tests.utils import get_test_data_path; from importlib.resources import files; d=json.load(open(files('tedana.resources.config')/'outputs.json')); print(d['frequency img'], d['phase0 img'], d['fit cost img'], d['fit nfev img'])"
```
Expected: prints the four dicts without `KeyError`/`JSONDecodeError`.

- [ ] **Step 3: Commit**

```bash
git add tedana/resources/config/outputs.json
git commit -m "feat(io): add nlls output descriptors (frequency, phase0, cost, nfev)"
```

---

### Task 7: t2smap CLI and workflow integration

**Files:**
- Modify: `tedana/workflows/t2smap.py` (parser ~lines 150-170; `--phase` near data arg; workflow signature ~line 250; validation/loading/dispatch/saving in the body)
- Test: deferred to Task 8 (integration).

**Interfaces:**
- Consumes: `decay.fit_complex_decay` (Task 4), output descriptors (Task 6), `decay.rmse_of_fit_decay_ts(fitmode="varys0")` (Task 5).
- Produces: `t2smap_workflow(..., phase=None, fittype in {loglin,curvefit,nlls}, fitmode in {all,ts,varys0})`.

- [ ] **Step 1: Update the parser**

In `_get_parser`, change `--fittype` choices (line 154) to include `nlls` and update help:

```python
        choices=["loglin", "curvefit", "nlls"],
        help=(
            "Desired T2*/S0 fitting method. "
            '"loglin" means that a linear model is fit to the log of the data. '
            '"curvefit" means that a more computationally demanding monoexponential model is fit '
            "to the raw data. "
            '"nlls" fits a complex monoexponential model to magnitude+phase data and requires '
            "--phase. "
        ),
```

Change `--fitmode` choices (line 166) to `["all", "ts", "varys0"]` and append to its help:

```python
            '"varys0" (nlls only) shares R2*/frequency across timepoints while '
            "allowing complex S0 to vary per timepoint. "
```

Add a `--phase` argument in the Required/data group, immediately after the `-d` block (after line 42):

```python
    required_args.add_argument(
        "--phase",
        dest="phase",
        nargs="+",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Phase images for each echo, in radians, in ascending echo order "
            "matching -d. Required when --fittype is 'nlls'."
        ),
        default=None,
    )
```

- [ ] **Step 2: Update workflow signature and validation**

Add `phase=None` to `t2smap_workflow` parameters (after `data,`/`tes,`, near line 241). Add validation after the existing `fitmode == "ts" and n_exclude > 0` check (after line 387):

```python
    if fittype == "nlls" and phase is None:
        raise ValueError("fittype='nlls' requires phase images (--phase).")
    if phase is not None and fittype != "nlls":
        raise ValueError("phase images are only used with fittype='nlls'.")
    if fitmode == "varys0" and fittype != "nlls":
        raise ValueError("fitmode='varys0' is only valid with fittype='nlls'.")
    if fitmode == "ts" and n_exclude > 0:
        # already handled above; varys0 intentionally permits exclude
        pass
    if isinstance(phase, str):
        phase = [phase]
    if phase is not None and len(phase) != len(data if isinstance(data, list) else [data]):
        raise ValueError("--phase must have the same number of files as -d.")
```

(Place the `isinstance(data, str)` coercion at line 395 before this length check, or reuse the existing coercion — ensure `data` is a list first.)

- [ ] **Step 3: Load phase and dispatch the complex fit**

After `data_cat` is loaded and trimmed (after line 446, where `data_without_excluded_vols` is built), load phase and branch. Replace the estimation block (lines 458-487) with:

```python
    LGR.info("Computing T2* map")
    data_fit = data_without_excluded_vols[mask_denoise, ...]
    masksum_masked = masksum_denoise[mask_denoise]

    if fittype == "nlls":
        phase_cat = io.load_data_nilearn(phase, mask_img=mask_img, n_echos=n_echos)
        if dummy_scans > 0:
            phase_cat = phase_cat[..., dummy_scans:]
        if phase_cat.shape != data_cat.shape:
            raise ValueError(
                f"Phase shape {phase_cat.shape} does not match data shape {data_cat.shape}."
            )
        # For varys0, pass all volumes plus a use_volumes mask; otherwise drop excluded volumes.
        if fitmode == "varys0":
            phase_fit = phase_cat[mask_denoise, ...]
            data_complex_mag = data_cat[mask_denoise, ...]
            if n_exclude > 0:
                vol_mask = use_volumes
            else:
                vol_mask = np.ones(n_vols, dtype=bool)
            complex_results = decay.fit_complex_decay(
                data=data_complex_mag, phase=phase_fit, tes=tes,
                adaptive_mask=masksum_masked, fitmode="varys0",
                use_volumes=vol_mask, n_threads=n_threads,
            )
        else:
            phase_fit = phase_cat[..., :][mask_denoise, ...] if n_exclude == 0 else \
                phase_cat[:, :, use_volumes][mask_denoise, ...]
            complex_results = decay.fit_complex_decay(
                data=data_fit, phase=phase_fit, tes=tes,
                adaptive_mask=masksum_masked, fitmode=fitmode, n_threads=n_threads,
            )
        t2s_full = complex_results["t2s"]
        s0_full = complex_results["s0"]
        failures = complex_results["failures"]
        t2s_var = s0_var = t2s_s0_covar = None
    else:
        decay_function = decay.fit_decay if fitmode == "all" else decay.fit_decay_ts
        t2s_full, s0_full, failures, t2s_var, s0_var, t2s_s0_covar = decay_function(
            data=data_fit, tes=tes, adaptive_mask=masksum_masked,
            fittype=fittype, n_threads=n_threads,
        )
    del data_without_excluded_vols
```

Note: `data_fit` replaces the prior in-place reassignment of `data_without_excluded_vols`; keep `del data_without_excluded_vols` and remove the old `data_without_excluded_vols = data_without_excluded_vols[mask_denoise, ...]` line (459) and `masksum_masked` line (460), since they are now set above.

- [ ] **Step 4: Save fit-quality and nlls maps**

Replace the `if fittype == "curvefit":` save block (lines 471-484) with:

```python
    if fittype in ("curvefit", "nlls"):
        io_generator.save_file(
            failures.astype(np.uint8), "fit failures img", mask=mask_denoise
        )
    if fittype == "curvefit" and verbose:
        io_generator.save_file(t2s_var, "t2star variance img", mask=mask_denoise)
        io_generator.save_file(s0_var, "s0 variance img", mask=mask_denoise)
        io_generator.save_file(t2s_s0_covar, "t2star-s0 covariance img", mask=mask_denoise)

    if fittype == "nlls":
        io_generator.save_file(
            utils.unmask(complex_results["frequency_hz"], mask_denoise), "frequency img"
        )
        io_generator.save_file(
            utils.unmask(complex_results["phase0"], mask_denoise), "phase0 img"
        )
```

(Keep the existing `del failures, t2s_var, s0_var, t2s_s0_covar` line; guard it or set the unused ones to `None` for nlls as above so the `del` still works.)

- [ ] **Step 5: Pass fitmode to RMSE**

The RMSE call (line 506-513) already passes `fitmode=fitmode`; confirm it now receives `"varys0"` when applicable (no change needed beyond Task 5).

- [ ] **Step 6: Run the full t2smap test module (regression)**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_t2smap.py -v`
Expected: existing tests PASS (no nlls tests yet; this confirms no regressions).

- [ ] **Step 7: Commit**

```bash
git add tedana/workflows/t2smap.py
git commit -m "feat(t2smap): add --fittype nlls, --phase, and --fitmode varys0"
```

---

### Task 8: Integration tests for nlls

**Files:**
- Modify: `tedana/tests/test_t2smap.py`
- Reference: existing fixtures/patterns in `test_t2smap.py` and three-echo test data via `tedana.tests.utils.get_test_data_path` (`echo1.nii.gz`..`echo3.nii.gz`, `mask.nii.gz`).

**Interfaces:**
- Consumes: `t2smap_workflow` (Task 7).

- [ ] **Step 1: Write a helper to synthesize phase files and the failing tests**

Add to `tedana/tests/test_t2smap.py` (imports `os.path as op`, `numpy as np`, `nibabel as nb`, `pytest`, `get_test_data_path`, and `from tedana.workflows import t2smap_workflow`):

```python
def _make_zero_phase_files(echo_files, out_dir):
    """Write zero-valued phase NIfTIs matching the given magnitude echoes."""
    phase_files = []
    for i, f in enumerate(echo_files):
        img = nb.load(f)
        phase = nb.Nifti1Image(np.zeros(img.shape, dtype=np.float32), img.affine, img.header)
        path = op.join(out_dir, f"phase{i + 1}.nii.gz")
        phase.to_filename(path)
        phase_files.append(path)
    return phase_files


@pytest.mark.parametrize("fitmode", ["all", "ts", "varys0"])
def test_t2smap_nlls(tmp_path, fitmode):
    """t2smap runs with fittype='nlls' across fitmodes and writes expected maps."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    mask = op.join(data_dir, "mask.nii.gz")
    phase_files = _make_zero_phase_files(echo_files, str(tmp_path))
    t2smap_workflow(
        data=echo_files,
        tes=[14.5, 38.5, 62.5],
        phase=phase_files,
        mask=mask,
        fittype="nlls",
        fitmode=fitmode,
        out_dir=str(tmp_path),
    )
    assert op.exists(op.join(tmp_path, "T2starmap.nii.gz"))
    assert op.exists(op.join(tmp_path, "S0map.nii.gz"))
    assert op.exists(op.join(tmp_path, "frequencyHzmap.nii.gz"))
    assert op.exists(op.join(tmp_path, "phase0map.nii.gz"))
    assert op.exists(op.join(tmp_path, "desc-optcom_bold.nii.gz"))
    # varys0 should produce a 4D S0 timeseries; all/ts produce maps too
    s0 = nb.load(op.join(tmp_path, "S0map.nii.gz"))
    if fitmode == "varys0":
        assert s0.ndim == 4


def test_t2smap_nlls_requires_phase(tmp_path):
    """fittype='nlls' without --phase raises."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    with pytest.raises(ValueError, match="requires phase"):
        t2smap_workflow(
            data=echo_files, tes=[14.5, 38.5, 62.5],
            fittype="nlls", out_dir=str(tmp_path),
        )


def test_t2smap_varys0_requires_nlls(tmp_path):
    """fitmode='varys0' with a non-nlls fittype raises."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    with pytest.raises(ValueError, match="varys0"):
        t2smap_workflow(
            data=echo_files, tes=[14.5, 38.5, 62.5],
            fittype="loglin", fitmode="varys0", out_dir=str(tmp_path),
        )
```

(Confirm the actual TEs for the test data from existing `test_t2smap.py`; reuse whatever value the existing tests pass to `tes`.)

- [ ] **Step 2: Run to verify failures (pre-fix) / passes (post-Task-7)**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_t2smap.py -k nlls -v`
Expected: PASS (Tasks 4-7 already implemented). If a 4D-S0 assertion fails, inspect the `varys0` save path in Task 7 Step 3/4.

- [ ] **Step 3: Run the whole decay + t2smap suites**

Run: `cd /mnt/c/Users/tsalo/Documents/tsalo/tedana && python -m pytest tedana/tests/test_decay.py tedana/tests/test_t2smap.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tedana/tests/test_t2smap.py
git commit -m "test(t2smap): integration tests for complex nlls fitting"
```

---

## Self-Review notes

- **Spec coverage:** `--fittype nlls` (Task 7), `--phase` (Task 7), `--fitmode varys0` (Tasks 4/5/7), complex model + per-sample fit (Task 1), adaptive-mask driver/`all` (Task 2), joint fit (Task 3), dispatcher/`ts`/`varys0` + `use_volumes`/exclude (Task 4/7), downstream RMSE shape handling (Task 5), `modify_t2s_s0_maps` reuse verified unchanged (Task 5 note), extra output maps (Task 6/7), tests (Tasks 1-5, 8). Modulation explicitly deferred (Task 1 model hook). Dahnke future work: out of scope, no task — correct.
- **Placeholder scan:** none — all steps carry runnable code/commands.
- **Type consistency:** `fit_complex_decay` returns a dict consumed by the workflow in Task 7; `frequency_hz`/`phase0`/`failures`/`t2s`/`s0` keys match across Tasks 2, 4, 7. `_solve_complex_s0`/`_fit_complex_decay_joint` signatures match their callers in Task 4.
- **Open verification during execution:** in Task 7, confirm the exact line numbers for the estimation block and the `data_without_excluded_vols[mask_denoise]` reassignment, since edits there are the trickiest; and confirm the test-data TEs in Task 8 against the existing `test_t2smap.py`.
