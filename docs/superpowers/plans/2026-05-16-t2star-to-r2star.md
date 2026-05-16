# T2* → R2* Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace tedana's internal T2* representation (seconds) with R2* (s⁻¹ = 1/T2*) throughout the codebase, add `--r2smap` CLI input, deprecate `--t2smap`, and rename output files from `T2starmap.nii.gz` to `R2starmap.nii.gz`.

**Architecture:** Sequential module-by-module rename in dependency order. Each module is self-consistent after its task completes. The log-linear fit natively yields R2* (the slope coefficient), eliminating the `1/betas[1]` inversion. Floor/cap logic in `modify_t2s_s0_maps` is inverted for R2* space.

**Tech Stack:** Python, NumPy, SciPy, NiBabel, Nilearn, argparse, Jinja2 (HTML reporting)

---

## File Map

| File | Change |
|------|--------|
| `tedana/decay.py` | Rename all `t2s*` → `r2s*`; invert math; `_apply_t2s_floor` → `_apply_r2s_ceiling`; `modify_t2s_s0_maps` → `modify_r2s_s0_maps` |
| `tedana/combine.py` | `_combine_t2s` → `_combine_r2s`; `make_optcom` param `t2s` → `r2s`; `combmode="t2s"` → `"r2s"` |
| `tedana/utils.py` | Add `check_r2s_values`; update `load_mask` to return R2* and accept `r2smap` |
| `tedana/resources/config/outputs.json` | Rename `t2star` → `r2star` in keys and BIDS filename stems |
| `tedana/workflows/tedana.py` | Add `--r2smap`; deprecate `--t2smap`; rename internal variables; update save calls |
| `tedana/workflows/t2smap.py` | Same as tedana.py |
| `tedana/reporting/static_figures.py` | `plot_t2star_and_s0` → `plot_r2star_and_s0`; rename figure filenames |
| `tedana/reporting/html_report.py` | Rename `t2star_*` locals and Jinja template keys |
| `tedana/reporting/data/html/report_body_template.html` | Rename template variables |
| `tedana/rica.py` | `"T2starmap.nii"` → `"R2starmap.nii"` |
| `docs/outputs.rst` | Update T2* → R2* descriptions and units |
| `tedana/tests/test_decay.py` | Rename variables; update `test__apply_t2s_floor` → `test__apply_r2s_ceiling` |
| `tedana/tests/test_combine.py` | Rename `t2s` → `r2s`; update `combmode` strings |
| `tedana/tests/test_utils.py` | Update `load_mask` tests; add `check_r2s_values` tests |
| `tedana/tests/test_io.py` | `"t2star img"` → `"r2star img"` |
| `tedana/tests/test_t2smap.py` | `T2starmap.nii.gz` → `R2starmap.nii.gz` |
| `tedana/tests/test_integration.py` | Switch `t2smap=` to `r2smap=` with inverted values |

---

## Task 1: Update `decay.py` — math primitives

**Files:**
- Modify: `tedana/decay.py`
- Test: `tedana/tests/test_decay.py`

### Step 1.1: Write a failing test for `_apply_r2s_ceiling`

In `tedana/tests/test_decay.py`, replace `test__apply_t2s_floor` with:

```python
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
```

- [ ] Add the test above to `tedana/tests/test_decay.py`, replacing the `test__apply_t2s_floor` function.

- [ ] **Run to confirm it fails:**
  ```bash
  cd /mnt/c/Users/tsalo/Documents/tsalo/tedana
  python -m pytest tedana/tests/test_decay.py::test__apply_r2s_ceiling -v 2>&1 | tail -20
  ```
  Expected: `FAILED` — `AttributeError: module 'tedana.decay' has no attribute '_apply_r2s_ceiling'` and `AttributeError: module 'tedana.combine' has no attribute '_combine_r2s'`

### Step 1.2: Rename `_apply_t2s_floor` → `_apply_r2s_ceiling` in `decay.py`

Replace the entire `_apply_t2s_floor` function (lines 20–63) with:

```python
def _apply_r2s_ceiling(r2s, echo_times):
    """Apply a ceiling to R2* values to prevent exp underflow during optimal combination.

    Parameters
    ----------
    r2s : (S [x T]) array_like
        R2* estimates in s⁻¹.
    echo_times : (E,) array_like
        Echo times in seconds.

    Returns
    -------
    r2s_corrected : (S,) array_like
        R2* estimates with very large values replaced with a ceiling value.
    """
    r2s_corrected = r2s.copy()

    if r2s.ndim == 2:
        for i_vol in range(r2s.shape[1]):
            r2s_corrected[:, i_vol] = _apply_r2s_ceiling(r2s[:, i_vol], echo_times)
        return r2s_corrected

    echo_times = np.asarray(echo_times)
    if echo_times.ndim == 1:
        echo_times = echo_times[:, None]

    eps = np.finfo(dtype=r2s.dtype).eps
    nonzerovox = r2s != 0
    temp_arr = np.zeros((len(echo_times), len(r2s)))
    temp_arr[:, nonzerovox] = np.exp(-echo_times * r2s[nonzerovox])  # (E x V) array
    bad_voxel_idx = np.any(temp_arr == 0, axis=0) & (r2s != 0)
    n_bad_voxels = np.sum(bad_voxel_idx)
    if n_bad_voxels > 0:
        n_voxels = temp_arr.size
        ceiling_percent = 100 * n_bad_voxels / n_voxels
        LGR.debug(
            f"R2* values for {n_bad_voxels}/{n_voxels} voxels ({ceiling_percent:.2f}%) have been "
            "identified as too large and have been adjusted"
        )
    r2s_corrected[bad_voxel_idx] = -np.log(eps) / np.max(echo_times)
    return r2s_corrected
```

- [ ] Make the replacement in `tedana/decay.py`.

### Step 1.3: Update `monoexponential` in `decay.py`

Replace lines 66–83:

```python
def monoexponential(tes, s0, r2star):
    """Specify a monoexponential model for use with scipy curve fitting.

    Parameters
    ----------
    tes : (E,) :obj:`list`
        Echo times in seconds.
    s0 : :obj:`float`
        Initial signal parameter.
    r2star : :obj:`float`
        R2* parameter in s⁻¹.

    Returns
    -------
    :obj:`float`
        Predicted signal.
    """
    return s0 * np.exp(-tes * r2star)
```

- [ ] Make the replacement in `tedana/decay.py`.

### Step 1.4: Update `fit_loglinear` in `decay.py`

- [ ] In `fit_loglinear` (around line 343): rename `t2s_asc_maps` → `r2s_asc_maps` throughout the function body.

- [ ] Replace the inversion line (currently `t2s = 1.0 / betas[1, :].T`) with:
  ```python
  r2s = betas[1, :].T
  ```

- [ ] Update the lines that set the map values and build the return value. The end of `fit_loglinear` (lines 374–383) becomes:
  ```python
      r2s_asc_maps[voxel_idx, i_echo] = r2s
      s0_asc_maps[voxel_idx, i_echo] = s0

  r2s = utils.unmask(r2s_asc_maps[echo_masks], adaptive_mask > 1)
  s0 = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask > 1)
  r2s[adaptive_mask == 1] = r2s_asc_maps[adaptive_mask == 1, 0]
  s0[adaptive_mask == 1] = s0_asc_maps[adaptive_mask == 1, 0]

  return r2s, s0
  ```

- [ ] Update the docstring of `fit_loglinear`: replace all "T2*" / `t2s` references with "R2*" / `r2s`; change the Notes paragraph from "The slope estimate is inverted (i.e., 1 / slope) to get :math:`T_2^*`" to "The slope estimate is :math:`R_2^*` directly."

### Step 1.5: Update `fit_monoexponential` in `decay.py`

- [ ] Rename `t2s_init` → `r2s_init`, `t2s_asc_maps` → `r2s_asc_maps`, `t2s_var_asc_maps` → `r2s_var_asc_maps`, `t2s_s0_covar_asc_maps` → `r2s_s0_covar_asc_maps` throughout the function body.

- [ ] In `_fit_single_voxel`, rename parameter `t2s_init` → `r2s_init` (line 100 in the docstring and line 115 in `p0=(s0_init, t2s_init)`):
  ```python
  p0=(s0_init, r2s_init),
  ```
  Note: the variable `t2s_voxel` in the results loop (line ~243) and `t2s_init[voxel] = t2s_voxel` (line ~254) become `r2s_voxel` and `r2s_init[voxel] = r2s_voxel`.

- [ ] Update the `fit_monoexponential` call to `fit_loglinear` for initial estimates (lines 186–191). The returned variable name changes:
  ```python
  r2s_init, s0_init = fit_loglinear(
      data_cat=data_cat,
      echo_times=echo_times,
      adaptive_mask=adaptive_mask,
      report=False,
  )
  ```

- [ ] Update the return value assembly at the bottom of `fit_monoexponential` (lines 271–282):
  ```python
  r2s = utils.unmask(r2s_asc_maps[echo_masks], adaptive_mask > 1)
  s0 = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask > 1)
  failures = utils.unmask(failures_asc_maps[echo_masks], adaptive_mask > 1)
  r2s_var = utils.unmask(r2s_var_asc_maps[echo_masks], adaptive_mask > 1)
  s0_var = utils.unmask(s0_var_asc_maps[echo_masks], adaptive_mask > 1)
  r2s_s0_covar = utils.unmask(r2s_s0_covar_asc_maps[echo_masks], adaptive_mask > 1)

  r2s[adaptive_mask == 1] = r2s_asc_maps[adaptive_mask == 1, 0]
  s0[adaptive_mask == 1] = s0_asc_maps[adaptive_mask == 1, 0]

  return r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar
  ```

- [ ] Update `fit_monoexponential` docstring: replace all "T2*" / `t2s` return-value descriptions with "R2*" / `r2s`.

### Step 1.6: Update `fit_decay` and `fit_decay_ts` in `decay.py`

- [ ] In `fit_decay` (lines 386–472): rename all `t2s` local variables and return tuple positions to `r2s`. The return statement becomes:
  ```python
  return r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar
  ```
  Update the docstring: returns section "t2s" → "r2s", "T2*" → "R2*".

- [ ] In `fit_decay_ts` (lines 475–562): rename `t2s` → `r2s` in the function signature, body, and docstring. The return becomes:
  ```python
  return r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar
  ```

### Step 1.7: Update `modify_t2s_s0_maps` → `modify_r2s_s0_maps` in `decay.py`

Replace the entire function with:

```python
def modify_r2s_s0_maps(r2s, s0, adaptive_mask, tes):
    """Modify R2* and S0 maps to include estimates for voxels with adaptive mask == 1.

    Parameters
    ----------
    r2s : (Md,) :obj:`numpy.ndarray`
        "Full" R2* map in s⁻¹. Includes estimates for all voxels with adaptive mask >= 1.
    s0 : (Md,) :obj:`numpy.ndarray`
        "Full" S0 map. Includes estimates for all voxels with adaptive mask >= 1.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Adaptive mask array. Values indicate the number of echoes with good signal.
    tes : (E,) :obj:`list`
        Echo times in seconds.

    Returns
    -------
    r2s : (Md,) :obj:`numpy.ndarray`
        "Full" R2* map with floors and ceilings applied.
    s0 : (Md,) :obj:`numpy.ndarray`
        "Full" S0 map with NaN → 0.
    r2s_limited : (Md,) :obj:`numpy.ndarray`
        "Limited" R2* map. Voxels with adaptive mask == 1 are set to 0.
    s0_limited : (Md,) :obj:`numpy.ndarray`
        "Limited" S0 map. Voxels with adaptive mask == 1 are set to 0.
    """
    # R2*=inf means T2*=0 (physically impossible); clamp to 1.0 s⁻¹
    r2s[np.isinf(r2s)] = 1.0
    # R2*≤0 means T2*→∞ or negative (bad fit); clamp to very small positive
    r2s[r2s <= 0] = 1 / 500.0
    r2s = _apply_r2s_ceiling(r2s, tes)
    s0[np.isnan(s0)] = 0.0

    r2s_limited = r2s.copy()
    s0_limited = s0.copy()
    r2s_limited[adaptive_mask == 1] = 0
    s0_limited[adaptive_mask == 1] = 0

    # Set a hard floor for the R2* limited map.
    # Voxels with R2* more than 10× below the 0.5th percentile (of positive values) are reset.
    positive_r2s = r2s_limited[r2s_limited > 0].flatten()
    floor_r2s = stats.scoreatpercentile(positive_r2s, 0.5, interpolation_method="lower")
    LGR.debug(f"Setting floor on R2* map at {floor_r2s / 10:.5f}")
    r2s_limited[(r2s_limited > 0) & (r2s_limited < floor_r2s / 10)] = floor_r2s

    return r2s, s0, r2s_limited, s0_limited
```

- [ ] Make the replacement in `tedana/decay.py`.

### Step 1.8: Update `rmse_of_fit_decay_ts` in `decay.py`

- [ ] Rename parameter `t2s` → `r2s` in the function signature and docstring.

- [ ] Rename all `t2s_echo` local variables to `r2s_echo`.

- [ ] Update the call to `monoexponential` (line ~699):
  ```python
  predicted_data[:, echo_num, :] = monoexponential(
      tes=tes[echo_num],
      s0=s0_echo,
      r2star=r2s_echo,
  )
  ```

### Step 1.9: Update `test_decay.py` smoke tests

- [ ] In `test_fit_decay` and all similar smoke tests, rename `t2s` → `r2s` in variable names and assertions. Example:
  ```python
  r2s, s0, failures, r2s_var, s0_var, r2s_s0_covar = me.fit_decay(...)
  assert r2s.ndim == 1
  assert r2s_var is None
  assert r2s_s0_covar is None
  ```

- [ ] Apply the same rename in `test_fit_decay_ts`, `test_smoke_fit_decay`, `test_smoke_fit_decay_curvefit`, `test_smoke_fit_decay_ts`, `test_smoke_fit_decay_ts_curvefit`.

### Step 1.10: Run decay tests and commit

- [ ] **Run:**
  ```bash
  python -m pytest tedana/tests/test_decay.py -v 2>&1 | tail -30
  ```
  Expected: all tests pass.

- [ ] **Commit:**
  ```bash
  git add tedana/decay.py tedana/tests/test_decay.py
  git commit -m "refactor: rename T2* → R2* in decay.py"
  ```

---

## Task 2: Update `combine.py`

**Files:**
- Modify: `tedana/combine.py`
- Test: `tedana/tests/test_combine.py`

### Step 2.1: Write failing tests for `_combine_r2s` and updated `make_optcom`

- [ ] In `tedana/tests/test_combine.py`, rename `test__combine_t2s` → `test__combine_r2s` and update all `t2s` variable names to `r2s`. Also update all `combmode="t2s"` → `combmode="r2s"`, `"Argument 't2s' must be supplied"` error messages, etc.:

```python
def test__combine_r2s():
    """Test tedana.combine._combine_r2s."""
    n_voxels, n_trs, n_echos = 100, 20, 5
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([[0.014, 0.028, 0.042, 0.056, 0.070]])  # 1 x E seconds

    r2s = np.random.random((n_voxels, n_trs, 1)) * 50  # Mb x T x 1, R2* in s⁻¹
    comb = combine._combine_r2s(data, tes, r2s)
    assert comb.shape == (n_voxels, n_trs)

    r2s = np.random.random((n_voxels, 1)) * 50  # M x 1
    comb = combine._combine_r2s(data, tes, r2s)
    assert comb.shape == (n_voxels, n_trs)


def test_make_optcom():
    """Test tedana.combine.make_optcom."""
    n_voxels, n_trs, n_echos = 100, 20, 5
    data = np.random.random((n_voxels, n_echos, n_trs))
    tes = np.array([0.014, 0.028, 0.042, 0.056, 0.070])  # seconds
    adaptive_mask = np.ones(n_voxels, dtype=int) * n_echos

    r2s = np.random.random((n_voxels, n_trs)) * 50
    comb = combine.make_optcom(data, tes, adaptive_mask, r2s=r2s, combmode="r2s")
    assert comb.shape == (n_voxels, n_trs)

    r2s = np.random.random(n_voxels) * 50
    comb = combine.make_optcom(data, tes, adaptive_mask, r2s=r2s, combmode="r2s")
    assert comb.shape == (n_voxels, n_trs)

    comb = combine.make_optcom(data, tes, adaptive_mask, r2s=r2s, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    comb = combine.make_optcom(data, tes, adaptive_mask, r2s=None, combmode="paid")
    assert comb.shape == (n_voxels, n_trs)

    bad_data = np.random.random((n_voxels, n_echos))
    with pytest.raises(ValueError, match="Input data must be 3D"):
        combine.make_optcom(bad_data, tes, adaptive_mask, r2s=r2s, combmode="r2s")

    bad_tes = np.array([0.014, 0.028])
    with pytest.raises(ValueError, match="Number of echos provided does not match"):
        combine.make_optcom(data, bad_tes, adaptive_mask, r2s=r2s, combmode="r2s")

    bad_adaptive_mask = np.ones((n_voxels, 2), dtype=int)
    with pytest.raises(ValueError, match="Mask is not 1D"):
        combine.make_optcom(data, tes, bad_adaptive_mask, r2s=r2s, combmode="r2s")

    bad_adaptive_mask2 = np.ones(n_voxels - 1, dtype=int)
    with pytest.raises(ValueError, match="Mask and data do not have same number"):
        combine.make_optcom(data, tes, bad_adaptive_mask2, r2s=r2s, combmode="r2s")

    with pytest.raises(ValueError, match="Argument 'combmode' must be either 'r2s' or 'paid'"):
        combine.make_optcom(data, tes, adaptive_mask, r2s=r2s, combmode="bad")

    with pytest.raises(ValueError, match="Argument 'r2s' must be supplied if 'combmode' is set to 'r2s'"):
        combine.make_optcom(data, tes, adaptive_mask, r2s=None, combmode="r2s")

    bad_r2s = np.random.random(n_voxels - 1) * 50
    with pytest.raises(ValueError, match="R2\\* estimates and data do not have same number"):
        combine.make_optcom(data, tes, adaptive_mask, r2s=bad_r2s, combmode="r2s")
```

- [ ] **Run to confirm it fails:**
  ```bash
  python -m pytest tedana/tests/test_combine.py -v 2>&1 | tail -20
  ```
  Expected: `FAILED` — functions renamed don't exist yet.

### Step 2.2: Update `combine.py`

- [ ] Rename `_combine_t2s` → `_combine_r2s`, rename parameter `ft2s` → `fr2s`, and update the weight formula:
  ```python
  alpha = tes * np.exp(-tes * fr2s)
  ```

- [ ] In `make_optcom`: rename parameter `t2s` → `r2s`; update `combmode` default and validation:
  ```python
  def make_optcom(data, tes, adaptive_mask, r2s=None, combmode="r2s"):
  ```
  Update the validation block:
  ```python
  if combmode not in ["r2s", "paid"]:
      raise ValueError("Argument 'combmode' must be either 'r2s' or 'paid'")
  elif combmode == "r2s" and r2s is None:
      raise ValueError("Argument 'r2s' must be supplied if 'combmode' is set to 'r2s'.")
  elif combmode == "paid" and r2s is not None:
      LGR.warning(
          "Argument 'r2s' is not required if 'combmode' is 'paid'. "
          "'r2s' array will not be used."
      )
  ```
  Update the shape check and log messages (replace "T2*" with "R2*"):
  ```python
  if r2s.shape[0] != data.shape[0]:
      raise ValueError(
          "R2* estimates and data do not have same number of "
          f"voxels/samples: {r2s.shape[0]} != {data.shape[0]}"
      )
  if r2s.ndim == 1:
      msg = "Optimally combining data with voxel-wise R2* estimates"
  else:
      msg = "Optimally combining data with voxel- and volume-wise R2* estimates"
  ```
  Update the internal call:
  ```python
  r2s_ = r2s[..., np.newaxis]
  combined[voxel_idx, :] = _combine_r2s(
      data[voxel_idx, :echo_num, :],
      tes[:, :echo_num],
      r2s_[voxel_idx, ...],
      report=report,
  )
  ```

- [ ] Update docstrings: replace all "T2*"/"t2s" with "R2*"/"r2s"; update the math formula in the Notes section to show `R_2^*`.

### Step 2.3: Run combine tests and commit

- [ ] **Run:**
  ```bash
  python -m pytest tedana/tests/test_combine.py -v 2>&1 | tail -20
  ```
  Expected: all tests pass.

- [ ] **Commit:**
  ```bash
  git add tedana/combine.py tedana/tests/test_combine.py
  git commit -m "refactor: rename T2* → R2* in combine.py"
  ```

---

## Task 3: Update `utils.py` — add `check_r2s_values`

**Files:**
- Modify: `tedana/utils.py`
- Test: `tedana/tests/test_utils.py`

### Step 3.1: Write failing test for `check_r2s_values`

- [ ] Add to `tedana/tests/test_utils.py` after `test_check_t2s_values`:

```python
def test_check_r2s_values(caplog):
    """check_r2s_values should return values in s⁻¹ and raise/warn on bad input."""
    # Valid R2* values (typical 10-100 s⁻¹)
    r2s_valid = np.array([14.3, 25.0, 33.3, 50.0, 66.7])
    result = utils.check_r2s_values(r2s_valid)
    np.testing.assert_array_equal(result, r2s_valid)

    # R2* with zeros (masked voxels) — should be returned unchanged
    r2s_with_zeros = np.array([0.0, 25.0, 33.3, 0.0, 50.0])
    result = utils.check_r2s_values(r2s_with_zeros)
    np.testing.assert_array_equal(result, r2s_with_zeros)

    # All zeros — should return as-is with warning
    r2s_all_zeros = np.array([0.0, 0.0, 0.0])
    result = utils.check_r2s_values(r2s_all_zeros)
    np.testing.assert_array_equal(result, r2s_all_zeros)

    # Values that look like T2* in seconds (too small for R2*) — should raise
    r2s_like_t2s_seconds = np.array([0.015, 0.025, 0.035])
    with pytest.raises(ValueError, match="too small"):
        utils.check_r2s_values(r2s_like_t2s_seconds)

    # Values that are absurdly large — should raise
    r2s_too_large = np.array([2000.0, 3000.0, 5000.0])
    with pytest.raises(ValueError, match="too large"):
        utils.check_r2s_values(r2s_too_large)
```

- [ ] **Run to confirm it fails:**
  ```bash
  python -m pytest tedana/tests/test_utils.py::test_check_r2s_values -v 2>&1 | tail -10
  ```
  Expected: `FAILED` — `AttributeError: module 'tedana.utils' has no attribute 'check_r2s_values'`

### Step 3.2: Implement `check_r2s_values` in `utils.py`

- [ ] Add after the existing `check_t2s_values` function in `tedana/utils.py`:

```python
def check_r2s_values(r2s_map):
    """Check R2* map values are in expected units (s⁻¹).

    Parameters
    ----------
    r2s_map : numpy.ndarray
        R2* map values to check. Expected to be in s⁻¹ per BIDS convention.

    Returns
    -------
    numpy.ndarray
        R2* map values in s⁻¹.

    Raises
    ------
    ValueError
        If R2* values appear to be outside the expected range.
    """
    positive_values = r2s_map[r2s_map > 0]
    if len(positive_values) == 0:
        LGR.warning("R2* map has no positive values.")
        return r2s_map

    median_r2s = np.median(positive_values)

    if median_r2s < 0.5:
        raise ValueError(
            f"R2* map median value is {median_r2s:.4f} s⁻¹, which is too small. "
            "R2* maps should be in s⁻¹ (typical values: 10-100 s⁻¹). "
            "If you have a T2* map in seconds, use --t2smap instead."
        )
    elif median_r2s > 1000:
        raise ValueError(
            f"R2* map median value is {median_r2s:.2f} s⁻¹, which is too large. "
            "R2* maps should be in s⁻¹ (typical values: 10-100 s⁻¹). "
            "Please check your R2* map units."
        )
    elif not (2 <= median_r2s <= 200):
        LGR.warning(
            f"R2* map median value is {median_r2s:.2f} s⁻¹, which is outside the typical range "
            "(2–200 s⁻¹). Please verify your R2* map units."
        )
    else:
        LGR.debug(f"R2* map values appear to be in s⁻¹ (median={median_r2s:.4f} s⁻¹).")

    return r2s_map
```

### Step 3.3: Run test and commit

- [ ] **Run:**
  ```bash
  python -m pytest tedana/tests/test_utils.py::test_check_r2s_values -v 2>&1 | tail -10
  ```
  Expected: `PASSED`

- [ ] **Commit:**
  ```bash
  git add tedana/utils.py tedana/tests/test_utils.py
  git commit -m "feat: add check_r2s_values to utils.py"
  ```

---

## Task 4: Update `utils.py` — `load_mask` returns R2*

**Files:**
- Modify: `tedana/utils.py`
- Test: `tedana/tests/test_utils.py`

### Step 4.1: Write failing tests for updated `load_mask`

- [ ] In `tedana/tests/test_utils.py`, update the existing `load_mask` tests and add a new `r2smap` test:

Replace `test_load_mask_t2smap_only_builds_mask_and_returns_seconds` with:
```python
def test_load_mask_t2smap_only_builds_mask_and_returns_r2s(tmp_path):
    """`load_mask` with t2smap should build a mask and return R2* (= 1/T2*)."""
    shape = (3, 3, 3)
    affine = np.eye(4)
    ref_img = nb.Nifti1Image(np.ones(shape, dtype=np.float32), affine)

    t2s_arr = np.zeros(shape, dtype=np.float32)
    t2s_arr[1, 1, 1] = 0.02  # T2* in seconds → R2* = 50 s⁻¹
    t2s_img = nb.Nifti1Image(t2s_arr, affine)
    t2s_path = tmp_path / "t2smap.nii.gz"
    t2s_img.to_filename(t2s_path)

    out_mask_img, out_r2s = utils.load_mask(ref_img, mask=None, t2smap=str(t2s_path))

    assert out_mask_img is not None
    assert out_r2s is not None
    assert np.asarray(out_r2s).shape == (1,)
    assert np.allclose(out_r2s[0], 50.0)  # 1/0.02 = 50 s⁻¹
```

Replace `test_load_mask_combines_mask_and_t2smap` with:
```python
def test_load_mask_combines_mask_and_t2smap(tmp_path):
    """`load_mask` with mask + t2smap should combine both and return R2*."""
    shape = (3, 3, 3)
    affine = np.eye(4)
    ref_img = nb.Nifti1Image(np.ones(shape, dtype=np.float32), affine)

    mask_arr = np.zeros(shape, dtype=np.uint8)
    mask_arr[1, 1, 1] = 1
    mask_img = nb.Nifti1Image(mask_arr, affine)
    mask_path = tmp_path / "mask.nii.gz"
    mask_img.to_filename(mask_path)

    t2s_arr = np.zeros(shape, dtype=np.float32)
    t2s_arr[1, 1, 1] = 0.03  # T2* = 30ms → R2* ≈ 33.33 s⁻¹
    t2s_img = nb.Nifti1Image(t2s_arr, affine)
    t2s_path = tmp_path / "t2smap.nii.gz"
    t2s_img.to_filename(t2s_path)

    out_mask_img, out_r2s = utils.load_mask(
        ref_img, mask=str(mask_path), t2smap=str(t2s_path)
    )

    assert out_mask_img is not None
    assert np.asarray(out_r2s).shape == (1,)
    assert np.allclose(out_r2s[0], 1 / 0.03)
```

Add a new test for `r2smap`:
```python
def test_load_mask_r2smap_only_builds_mask_and_returns_r2s(tmp_path):
    """`load_mask` with r2smap should build a mask and return the R2* values directly."""
    shape = (3, 3, 3)
    affine = np.eye(4)
    ref_img = nb.Nifti1Image(np.ones(shape, dtype=np.float32), affine)

    r2s_arr = np.zeros(shape, dtype=np.float32)
    r2s_arr[1, 1, 1] = 33.3  # R2* in s⁻¹
    r2s_img = nb.Nifti1Image(r2s_arr, affine)
    r2s_path = tmp_path / "r2smap.nii.gz"
    r2s_img.to_filename(r2s_path)

    out_mask_img, out_r2s = utils.load_mask(ref_img, mask=None, r2smap=str(r2s_path))

    assert out_mask_img is not None
    assert out_r2s is not None
    assert np.asarray(out_r2s).shape == (1,)
    assert np.allclose(out_r2s[0], 33.3, rtol=1e-4)
```

Update `test_load_mask_user_mask_converted_to_nifti1`:
```python
out_mask_img, out_r2s = utils.load_mask(ref_img, mask=str(mask_path), t2smap=None)
assert out_r2s is None
```

Update `test_load_mask_falls_back_to_compute_epi_mask`:
```python
out_mask_img, out_r2s = utils.load_mask(ref_img, mask=None, t2smap=None)
assert out_r2s is None
```

- [ ] **Run to confirm they fail:**
  ```bash
  python -m pytest tedana/tests/test_utils.py -k "load_mask" -v 2>&1 | tail -20
  ```

### Step 4.2: Update `load_mask` in `utils.py`

- [ ] Replace the entire `load_mask` function with:

```python
def load_mask(ref_img, mask=None, t2smap=None, r2smap=None):
    """Load mask from user-defined mask, T2* map (deprecated), or R2* map.

    Parameters
    ----------
    ref_img : nibabel.Nifti1Image
    mask : str or None
    t2smap : str or None
        Path to T2* map file (deprecated; use r2smap instead).
    r2smap : str or None
        Path to R2* map file in s⁻¹.

    Returns
    -------
    mask_img : nibabel.Nifti1Image
        Mask image.
    r2s : numpy.ndarray or None
        Masked R2* map in s⁻¹, or None if no map was provided.
    """
    import nibabel as nb
    from nilearn.masking import apply_mask, compute_epi_mask

    from tedana import io

    if t2smap is not None and r2smap is not None:
        raise ValueError("Only one of t2smap or r2smap can be provided.")

    if t2smap is not None:
        LGR.warning(
            "--t2smap is deprecated and will be removed in a future release. "
            "Please convert your T2* map to R2* (R2* = 1/T2*) and use --r2smap instead."
        )

    r2s = None
    if mask and not (t2smap or r2smap):
        LGR.info("Using user-defined mask")
        RepLGR.info("A user-defined mask was applied to the data.")
        mask_img = io._convert_to_nifti1(nb.load(mask), max_dim=3)
        mask_data = mask_img.get_fdata() > 0
        mask_img = nb.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
    elif t2smap and not mask:
        LGR.info("Assuming user-defined T2* map is masked and using it to generate mask")
        t2s_img = io._convert_to_nifti1(nb.load(t2smap), max_dim=3)
        t2s_loaded = t2s_img.get_fdata()
        mask_data = (t2s_loaded != 0).astype(np.uint8)
        mask_img = nb.Nifti1Image(mask_data, ref_img.affine)
        t2s = apply_mask(t2s_img, mask_img)
        t2s = check_t2s_values(t2s)
        r2s = 1.0 / t2s
    elif t2smap and mask:
        LGR.info("Combining user-defined mask and T2* map to generate mask")
        t2s_img = io._convert_to_nifti1(nb.load(t2smap), max_dim=3)
        t2s_loaded = t2s_img.get_fdata()
        mask_img = io._convert_to_nifti1(nb.load(mask), max_dim=3)
        mask_data = mask_img.get_fdata() > 0
        mask_data[t2s_loaded == 0] = 0
        mask_img = nb.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
        t2s = apply_mask(t2s_img, mask_img)
        t2s = check_t2s_values(t2s)
        r2s = 1.0 / t2s
    elif r2smap and not mask:
        LGR.info("Assuming user-defined R2* map is masked and using it to generate mask")
        r2s_img = io._convert_to_nifti1(nb.load(r2smap), max_dim=3)
        r2s_loaded = r2s_img.get_fdata()
        mask_data = (r2s_loaded != 0).astype(np.uint8)
        mask_img = nb.Nifti1Image(mask_data, ref_img.affine)
        r2s = apply_mask(r2s_img, mask_img)
        r2s = check_r2s_values(r2s)
    elif r2smap and mask:
        LGR.info("Combining user-defined mask and R2* map to generate mask")
        r2s_img = io._convert_to_nifti1(nb.load(r2smap), max_dim=3)
        r2s_loaded = r2s_img.get_fdata()
        mask_img = io._convert_to_nifti1(nb.load(mask), max_dim=3)
        mask_data = mask_img.get_fdata() > 0
        mask_data[r2s_loaded == 0] = 0
        mask_img = nb.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
        r2s = apply_mask(r2s_img, mask_img)
        r2s = check_r2s_values(r2s)
    else:
        LGR.warning(
            "Computing EPI mask from first echo using nilearn's compute_epi_mask function. "
            "Most external pipelines include more reliable masking functions. "
            "It is strongly recommended to provide an external mask, "
            "and to visually confirm that mask accurately conforms to data boundaries."
        )
        mask_img = compute_epi_mask(ref_img)
        RepLGR.info(
            "An initial mask was generated from the first echo using "
            "nilearn's compute_epi_mask function."
        )

    return mask_img, r2s
```

### Step 4.3: Run utils tests and commit

- [ ] **Run:**
  ```bash
  python -m pytest tedana/tests/test_utils.py -v 2>&1 | tail -30
  ```
  Expected: all tests pass.

- [ ] **Commit:**
  ```bash
  git add tedana/utils.py tedana/tests/test_utils.py
  git commit -m "refactor: load_mask returns R2*; add r2smap param and check_r2s_values"
  ```

---

## Task 5: Update `outputs.json` and `test_io.py`

**Files:**
- Modify: `tedana/resources/config/outputs.json`
- Test: `tedana/tests/test_io.py`

### Step 5.1: Write failing test for new output key name

- [ ] In `tedana/tests/test_io.py`, find the line with `"t2star img"` (line 351) and update it to:
  ```python
  fname = io_generator.save_file(data_1d, "r2star img")
  ```

- [ ] **Run to confirm it fails:**
  ```bash
  python -m pytest tedana/tests/test_io.py -k "t2star or r2star" -v 2>&1 | tail -10
  ```

### Step 5.2: Update `outputs.json`

- [ ] In `tedana/resources/config/outputs.json`, rename the four T2* output keys and their BIDS filenames:

```json
"r2star img": {
    "bidsv1.5.0": "R2starmap"
},
"r2star variance img": {
    "bidsv1.5.0": "stat-variance_desc-r2star_statmap"
},
"r2star-s0 covariance img": {
    "bidsv1.5.0": "stat-covariance_desc-r2star+s0_statmap"
},
"limited r2star img": {
    "bidsv1.5.0": "desc-limited_R2starmap"
},
```

### Step 5.3: Run io tests and commit

- [ ] **Run:**
  ```bash
  python -m pytest tedana/tests/test_io.py -v 2>&1 | tail -20
  ```
  Expected: all tests pass.

- [ ] **Commit:**
  ```bash
  git add tedana/resources/config/outputs.json tedana/tests/test_io.py
  git commit -m "refactor: rename T2starmap → R2starmap output keys in outputs.json"
  ```

---

## Task 6: Update `workflows/tedana.py`

**Files:**
- Modify: `tedana/workflows/tedana.py`

### Step 6.1: Add `--r2smap` CLI argument (mutually exclusive with `--t2smap`)

- [ ] In the argument parsing section (around lines 195–213), replace the standalone `--t2smap` argument with a mutually exclusive group:

```python
map_group = decay_args.add_mutually_exclusive_group()
map_group.add_argument(
    "--t2smap",
    dest="t2smap",
    metavar="FILE",
    type=lambda x: is_valid_file(parser, x),
    help=(
        "[DEPRECATED: use --r2smap] Precalculated T2* map in the same space as the input data. "
        "Values should be in seconds (per BIDS convention). "
        "Maps in milliseconds are auto-detected and handled with a warning."
    ),
    default=None,
)
map_group.add_argument(
    "--r2smap",
    dest="r2smap",
    metavar="FILE",
    type=lambda x: is_valid_file(parser, x),
    help=(
        "Precalculated R2* map in the same space as the input data. "
        "Values should be in s⁻¹ (per BIDS convention)."
    ),
    default=None,
)
```

### Step 6.2: Add `r2smap` parameter to `tedana_workflow` function signature

- [ ] Find the function signature of `tedana_workflow` (around line 433–452) and add `r2smap=None` alongside `t2smap=None`. Also add to the docstring:
  ```
  r2smap : :obj:`str`, optional
      Precalculated R2* map in s⁻¹.
  ```

### Step 6.3: Update the t2smap/r2smap file-copy block

- [ ] Replace the block at lines 670–677:
  ```python
  if t2smap is not None and op.isfile(t2smap):
      r2smap_file = io_generator.get_name("r2star img")
      t2smap = op.abspath(t2smap)
      if t2smap != r2smap_file:
          shutil.copyfile(t2smap, r2smap_file)
  elif t2smap is not None:
      raise OSError("Argument 't2smap' must be an existing file.")

  if r2smap is not None and op.isfile(r2smap):
      r2smap_file = io_generator.get_name("r2star img")
      r2smap = op.abspath(r2smap)
      if r2smap != r2smap_file:
          shutil.copyfile(r2smap, r2smap_file)
  elif r2smap is not None:
      raise OSError("Argument 'r2smap' must be an existing file.")
  ```

### Step 6.4: Update the `load_mask` call

- [ ] At line 684, update to pass both parameters:
  ```python
  mask_img, r2s_limited = utils.load_mask(ref_img, mask=mask, t2smap=t2smap, r2smap=r2smap)
  if r2s_limited is not None:
      r2s_full = r2s_limited.copy()
  ```

### Step 6.5: Rename all T2* data variables

- [ ] Rename throughout the rest of the workflow function body:
  - `t2s_limited` → `r2s_limited`
  - `t2s_full` → `r2s_full`
  - `t2s_var` → `r2s_var`
  - `t2s_s0_covar` → `r2s_s0_covar`
  - `"Computing T2* map"` → `"Computing R2* map"`

### Step 6.6: Update `decay.fit_decay` call and unpacking

- [ ] At line 783, the unpacked variables become:
  ```python
  r2s_full, s0_full, failures, r2s_var, s0_var, r2s_s0_covar = decay.fit_decay(...)
  ```

### Step 6.7: Update `save_file` calls and `modify_r2s_s0_maps` call

- [ ] Update lines 799–823:
  ```python
  io_generator.save_file(r2s_var, "r2star variance img", mask=mask_denoise)
  io_generator.save_file(s0_var, "s0 variance img", mask=mask_denoise)
  io_generator.save_file(r2s_s0_covar, "r2star-s0 covariance img", mask=mask_denoise)
  ```
  ```python
  r2s_full, s0_full, r2s_limited, s0_limited = decay.modify_r2s_s0_maps(
      r2s=r2s_full,
      s0=s0_full,
      adaptive_mask=masksum_masked,
      tes=tes,
  )
  ```
  ```python
  io_generator.save_file(r2s_full, "r2star img")
  io_generator.save_file(s0_full, "s0 img", mask=mask_denoise)
  if verbose:
      io_generator.save_file(r2s_limited, "limited r2star img")
      io_generator.save_file(s0_limited, "limited s0 img")
  ```
  ```python
  rmse_map, rmse_df = decay.rmse_of_fit_decay_ts(
      data=data_cat,
      tes=tes,
      adaptive_mask=masksum_denoise,
      r2s=r2s_limited,
      s0=s0_limited,
      fitmode="all",
  )
  ```

### Step 6.8: Update `make_optcom` call and `combmode` default

- [ ] Update the `make_optcom` call (line 840):
  ```python
  data_optcom = combine.make_optcom(
      data_cat,
      tes,
      masksum_denoise,
      r2s=r2s_full,
      combmode=combmode,
  )
  ```

- [ ] Update CLI `--combmode` argument choices and default: `choices=["t2s"]` → `choices=["r2s"]`; `default="t2s"` → `default="r2s"`.

- [ ] Update `tedana_workflow` function signature default: `combmode="t2s"` → `combmode="r2s"`.

### Step 6.9: Update reporting call and conditions

- [ ] At line 1183, update the static figure call:
  ```python
  reporting.static_figures.plot_r2star_and_s0(
      io_generator=io_generator,
      mask=mask_denoise_img,
  )
  ```

- [ ] At line 1187, update the condition that guards RMSE plotting (only runs when R2* was fit, not loaded from a file):
  ```python
  if t2smap is None and r2smap is None:
  ```

- [ ] At line 779, update the condition that guards R2* fitting:
  ```python
  if t2smap is None and r2smap is None:
  ```

- [ ] Update log message at line 780: `"Computing R2* map"` (already done in Step 6.5).

- [ ] **Run a quick import check:**
  ```bash
  python -c "from tedana.workflows.tedana import tedana_workflow; print('OK')"
  ```
  Expected: `OK`

- [ ] **Commit:**
  ```bash
  git add tedana/workflows/tedana.py
  git commit -m "refactor: rename T2* → R2* in workflows/tedana.py; add --r2smap CLI arg"
  ```

---

## Task 7: Update `workflows/t2smap.py`

**Files:**
- Modify: `tedana/workflows/t2smap.py`

Apply the same pattern as Task 6 to `t2smap.py`:

### Step 7.1: Update CLI `--combmode`

- [ ] Lines 175–176: `choices=["t2s"]` → `choices=["r2s"]`, default `"t2s"` → `"r2s"`.

### Step 7.2: Update `t2smap_workflow` signature default

- [ ] `combmode="t2s"` → `combmode="r2s"` in function signature (line 252).

### Step 7.3: Rename all T2* data variables

- [ ] Rename `t2s_full` → `r2s_full`, `t2s_limited` → `r2s_limited`, `t2s_var` → `r2s_var`, `t2s_s0_covar` → `r2s_s0_covar` throughout the function body.

### Step 7.4: Update `decay.fit_decay` / `fit_decay_ts` unpacking

- [ ] At line 462:
  ```python
  r2s_full, s0_full, failures, r2s_var, s0_var, r2s_s0_covar = decay_function(...)
  ```

### Step 7.5: Update `save_file` calls

- [ ] Lines 478–484:
  ```python
  io_generator.save_file(r2s_var, "r2star variance img", mask=mask_denoise)
  io_generator.save_file(s0_var, "s0 variance img", mask=mask_denoise)
  io_generator.save_file(r2s_s0_covar, "r2star-s0 covariance img", mask=mask_denoise)
  ```

### Step 7.6: Update `modify_r2s_s0_maps` call

- [ ] Lines 489–494:
  ```python
  r2s_full, s0_full, r2s_limited, s0_limited = decay.modify_r2s_s0_maps(
      r2s=r2s_full,
      s0=s0_full,
      adaptive_mask=masksum_masked,
      tes=tes,
  )
  ```

### Step 7.7: Update `rmse_of_fit_decay_ts` call

- [ ] Lines 506–513:
  ```python
  rmse_map, rmse_df = decay.rmse_of_fit_decay_ts(
      data=data_cat,
      tes=tes,
      adaptive_mask=masksum_denoise,
      r2s=r2s_limited,
      s0=s0_limited,
      fitmode=fitmode,
  )
  ```

### Step 7.8: Update `save_file` for limited and full maps

- [ ] Lines 516–531:
  ```python
  io_generator.save_file(s0_limited, "limited s0 img")
  del s0_limited
  io_generator.save_file(r2s_limited, "limited r2star img")
  del r2s_limited
  ```
  ```python
  data_optcom = combine.make_optcom(
      data_cat,
      tes,
      masksum_denoise,
      r2s=r2s_full,
      combmode=combmode,
  )
  io_generator.save_file(r2s_full, "r2star img")
  ```

### Step 7.9: Update log messages and docstrings

- [ ] `"Computing T2* map"` → `"Computing R2* map"` (line 458).
- [ ] Update the outputs docstring: `T2starmap.nii.gz` → `R2starmap.nii.gz`, `desc-limited_T2starmap.nii.gz` → `desc-limited_R2starmap.nii.gz`, update units from seconds to s⁻¹.

### Step 7.10: Verify and commit

- [ ] **Run:**
  ```bash
  python -c "from tedana.workflows.t2smap import t2smap_workflow; print('OK')"
  ```

- [ ] **Commit:**
  ```bash
  git add tedana/workflows/t2smap.py
  git commit -m "refactor: rename T2* → R2* in workflows/t2smap.py; add --r2smap CLI arg"
  ```

---

## Task 8: Update `reporting/`

**Files:**
- Modify: `tedana/reporting/static_figures.py`
- Modify: `tedana/reporting/html_report.py`
- Modify: `tedana/reporting/data/html/report_body_template.html`

### Step 8.1: Update `static_figures.py`

- [ ] Rename function `plot_t2star_and_s0` → `plot_r2star_and_s0`.

- [ ] Inside the function body, rename all local variables and filenames:
  - `t2star_img` → `r2star_img`
  - `io_generator.get_name("t2star img")` → `io_generator.get_name("r2star img")`
  - `t2star_histogram` → `r2star_histogram` (filename: `{prefix}r2star_histogram.svg`)
  - `t2star_plot` → `r2star_plot` (filename: `{prefix}r2star_brain.svg`)
  - `t2s_p02`, `t2s_p98` → `r2s_p02`, `r2s_p98`
  - `t2star_data` → `r2star_data`

- [ ] In `plot_rmse` (if it calls `plot_t2star_and_s0`): update call to `plot_r2star_and_s0`.

- [ ] In `generate_report_figures` (line 1183–1187), update the `imgs` list:
  ```python
  imgs = ["r2star variance img", "s0 variance img", "r2star-s0 covariance img"]
  ```
  And update the output filenames for the variance/covariance SVGs.

### Step 8.2: Update `html_report.py`

- [ ] Rename all `t2star_*` local variables to `r2star_*` (lines 169–213):
  - `t2star_brain_filename` → `r2star_brain_filename` (value: `f"{prefix}r2star_brain.svg"`)
  - `t2star_histogram_filename` → `r2star_histogram_filename` (value: `f"{prefix}r2star_histogram.svg"`)
  - `t2star_brain` → `r2star_brain`
  - `t2star_histogram` → `r2star_histogram`
  - `t2star_variance_filename` → `r2star_variance_filename` (value: `f"{prefix}stat-variance_desc-r2star_statmap.svg"`)
  - `t2star_variance` → `r2star_variance`
  - `t2s_s0_covariance_filename` → `r2s_s0_covariance_filename` (value: `f"{prefix}stat-covariance_desc-r2star+s0_statmap.svg"`)
  - `t2star_exists` → `r2star_exists`

- [ ] Update the `LGR.info` message: `"T2* files exist:"` → `"R2* files exist:"`.

- [ ] Update the Jinja template keyword arguments (lines 242–245):
  ```python
  r2starBrainPlot=r2star_brain,
  r2starHistogram=r2star_histogram,
  r2starExists=r2star_exists,
  r2starVariancePlot=r2star_variance,
  ```

- [ ] Find any call to `plot_t2star_and_s0` in `html_report.py` (if present) and update to `plot_r2star_and_s0`.

### Step 8.3: Update HTML template

- [ ] In `tedana/reporting/data/html/report_body_template.html` (lines 371–392), rename all Jinja variables:
  ```html
  {% if r2starExists %}
      <img id="r2starBrainPlot" src="{{ r2starBrainPlot }}" ...>
      <img id="r2starHistogram" src="{{ r2starHistogram }}" ...>
      <img id="r2starVariancePlot" src="{{ r2starVariancePlot }}" ...>
  ```

### Step 8.4: Verify and commit

- [ ] **Run import check:**
  ```bash
  python -c "from tedana.reporting import static_figures, html_report; print('OK')"
  ```

- [ ] **Commit:**
  ```bash
  git add tedana/reporting/static_figures.py tedana/reporting/html_report.py \
         tedana/reporting/data/html/report_body_template.html
  git commit -m "refactor: rename T2* → R2* in reporting module"
  ```

---

## Task 9: Update `rica.py` and `docs/outputs.rst`

**Files:**
- Modify: `tedana/rica.py`
- Modify: `docs/outputs.rst`

### Step 9.1: Update `rica.py`

- [ ] Find line 396 (`"T2starmap.nii"`) and change to `"R2starmap.nii"`.

### Step 9.2: Update `docs/outputs.rst`

- [ ] Update all references to `T2starmap.nii.gz` → `R2starmap.nii.gz`.
- [ ] Update all references to `desc-limited_T2starmap.nii.gz` → `desc-limited_R2starmap.nii.gz`.
- [ ] Update units from "seconds" → "s⁻¹" for the map descriptions.
- [ ] Update the `"t2star img"` key documentation entries to `"r2star img"` etc.
- [ ] Update prose descriptions: "T2* map" → "R2* map" where referring to the output file.

### Step 9.3: Commit

- [ ] **Commit:**
  ```bash
  git add tedana/rica.py docs/outputs.rst
  git commit -m "refactor: rename T2* → R2* in rica.py and docs/outputs.rst"
  ```

---

## Task 10: Update integration and workflow tests

**Files:**
- Modify: `tedana/tests/test_t2smap.py`
- Modify: `tedana/tests/test_integration.py`

### Step 10.1: Update `test_t2smap.py`

- [ ] Replace all `"T2starmap.nii.gz"` → `"R2starmap.nii.gz"` and `"desc-limited_T2starmap.nii.gz"` → `"desc-limited_R2starmap.nii.gz"`. Example (lines 33–37):
  ```python
  img = nb.load(op.join(out_dir, "R2starmap.nii.gz"))
  assert img.shape == (39, 38, 1)

  img = nb.load(op.join(out_dir, "desc-limited_R2starmap.nii.gz"))
  assert img.shape == (39, 38, 1)
  ```
  Apply the same pattern to all five test functions in the file.

### Step 10.2: Update `test_integration.py`

- [ ] At line 384, replace the `t2smap=` argument:
  ```python
  r2smap=os.path.join(out_dir, "R2starmap.nii.gz"),
  ```
  (The first workflow run in the same test already produces `R2starmap.nii.gz`; this second run consumes it as an R2* input.)

### Step 10.3: Run the full test suite (excluding integration tests)

- [ ] **Run:**
  ```bash
  python -m pytest tedana/tests/ -v --ignore=tedana/tests/test_integration.py 2>&1 | tail -40
  ```
  Expected: all tests pass (0 failures).

- [ ] **Commit:**
  ```bash
  git add tedana/tests/test_t2smap.py tedana/tests/test_integration.py
  git commit -m "test: update T2* → R2* in integration and workflow tests"
  ```

---

## Final verification

- [ ] **Run full unit test suite:**
  ```bash
  python -m pytest tedana/tests/ --ignore=tedana/tests/test_integration.py -v 2>&1 | tail -50
  ```
  Expected: all tests pass.

- [ ] **Run a quick import check of all changed modules:**
  ```bash
  python -c "
  from tedana import decay, combine, utils
  from tedana.workflows import tedana as tw, t2smap as ts
  from tedana.reporting import static_figures, html_report
  from tedana import rica
  print('All imports OK')
  "
  ```
  Expected: `All imports OK`

- [ ] **Check for any remaining stray T2* references in non-test Python files:**
  ```bash
  grep -rn "t2s\b\|t2star\b\|T2starmap\|modify_t2s\|_apply_t2s\|_combine_t2s" \
    tedana/ --include="*.py" | grep -v "__pycache__" | grep -v "test_" | \
    grep -v "check_t2s_values\|t2smap" | grep -v "\.pyc"
  ```
  Expected: only `check_t2s_values` (kept for deprecated `--t2smap` path) and any `t2smap` file-path string handling should remain. No `t2s` data-variable references.
