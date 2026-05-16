# Design: Accept and Prioritize R2\* over T2\*

**Date:** 2026-05-16
**Issue:** [ME-ICA/tedana#1424](https://github.com/ME-ICA/tedana/issues/1424)
**Branch:** `t2star-to-r2star`
**Labels:** breaking-change, enhancement

---

## Summary

Replace tedana's internal representation from T2\* (seconds) to R2\* (s⁻¹, where R2\* = 1/T2\*).
R2\* is preferable for group-level analyses because outlier voxels with very small T2\* values
inflate to extremely large T2\* values when projected back to the native unit, whereas the same
voxels are simply large (but finite) R2\* values.

Breaking changes:
- Output file `T2starmap.nii.gz` → `R2starmap.nii.gz` (and related variants), values in s⁻¹.
- `combmode="t2s"` → `combmode="r2s"` in `make_optcom` (and the CLI `--combmode` argument).

---

## Scope

### In scope

- Rename all `t2s`/`t2star` variables, function parameters, and function names to `r2s`/`r2star`
  throughout `decay.py`, `combine.py`, `utils.py`, both workflows, reporting, `rica.py`, tests,
  and docs.
- Invert math everywhere: `exp(-te / t2s)` → `exp(-te * r2s)`; `t2s = 1 / betas[1]` →
  `r2s = betas[1]` (log-linear model directly yields R2\*; inversion removed).
- Add `--r2smap` CLI parameter (mutually exclusive with `--t2smap`); deprecate `--t2smap`.
- Rename output files: `T2starmap.nii.gz` → `R2starmap.nii.gz` (and all related variants).
- Invert floor/cap logic in `modify_t2s_s0_maps` → `modify_r2s_s0_maps`.

### Out of scope (separate effort)

- Metric names `countsigFT2`, `dice_FT2`, `map FT2`, `map FS0`, etc. — these are
  classification-framework names used in decision tree JSON files and potentially external tools.
  Renaming them is a separate, larger breaking change.

---

## Data Flow

### Input

1. **`--r2smap FILE`** (new): load NIfTI directly as R2\* (s⁻¹); validate via new
   `check_r2s_values()`.
2. **`--t2smap FILE`** (deprecated): log `DeprecationWarning`, load as T2\*, run
   `check_t2s_values()` to normalise units (s vs ms), then compute `r2s = 1 / t2s` before
   returning. Callers receive R2\* regardless of which flag was used.
3. **Neither flag**: fit R2\* directly in `decay.fit_decay` (log-linear: `r2s = betas[1]`;
   monoexponential: fit with `r2star` parameter).

### Internal representation

R2\* in s⁻¹ everywhere from `load_mask` / `fit_decay` onward.

### Output

`R2starmap.nii.gz` (s⁻¹). The `t2smap` workflow also produces `R2starmap.nii.gz`.

---

## Module-by-module changes

### `decay.py`

| Old | New | Notes |
|-----|-----|-------|
| `_apply_t2s_floor(t2s, echo_times)` | `_apply_r2s_ceiling(r2s, echo_times)` | Prevent `exp(-te*r2s)` underflow; ceiling formula: `r2s[bad] = -log(eps) / max(te)` |
| `monoexponential(tes, s0, t2star)` | `monoexponential(tes, s0, r2star)` | `s0 * exp(-tes * r2star)` |
| `fit_loglinear` → `t2s = 1.0 / betas[1]` | `r2s = betas[1]` | Direct — log-linear model yields R2\*; inversion removed |
| `fit_monoexponential` — all `t2s_*` locals | `r2s_*` | Bounds unchanged (R2\* must be positive, same as T2\*) |
| `fit_decay` / `fit_decay_ts` — `t2s` | `r2s` | Rename only |
| `modify_t2s_s0_maps(t2s, …)` | `modify_r2s_s0_maps(r2s, …)` | Floor/cap inverted (see below) |

**Floor/cap inversion in `modify_r2s_s0_maps`:**

```python
# was: t2s[np.isinf(t2s)] = 500.0
# T2*→inf means R2*→0; clamp non-positive R2* to a small positive value
r2s[r2s <= 0] = 1 / 500.0

# was: t2s[t2s <= 0] = 1.0
# T2*=0 means R2*=inf; clamp infinite R2* to 1.0 s⁻¹
r2s[np.isinf(r2s)] = 1.0

# was: t2s = _apply_t2s_floor(t2s, tes)
r2s = _apply_r2s_ceiling(r2s, tes)

# was: cap at 10× 99.5th percentile of t2s_limited (remove anomalously large T2*)
# now: floor at 1/10th of 0.5th percentile of r2s_limited (remove anomalously small R2*)
floor_r2s = stats.scoreatpercentile(
    r2s_limited[r2s_limited > 0].flatten(), 0.5, interpolation_method="lower"
)
r2s_limited[r2s_limited < floor_r2s / 10] = floor_r2s
```

**`_apply_r2s_ceiling` ceiling formula:**

```python
# was: t2s_corrected[bad] = np.min(-echo_times) / np.log(eps)
#      = (-max_te) / (-|log_eps|) = max_te / |log_eps|   [T2* floor]
# new: r2s_corrected[bad] = -np.log(eps) / np.max(echo_times)
#      = |log_eps| / max_te = 1 / (max_te / |log_eps|)   [R2* ceiling = 1 / T2* floor]
r2s_corrected[bad] = -np.log(eps) / np.max(echo_times)
```

The R2\* ceiling is exactly the reciprocal of the T2\* floor, which is correct: if T2\* must be
at least `max_te / |log_eps|` to prevent underflow, then R2\* must be at most
`|log_eps| / max_te`.

---

### `combine.py`

| Old | New | Notes |
|-----|-----|-------|
| `_combine_t2s(data, tes, ft2s)` | `_combine_r2s(data, tes, fr2s)` | |
| `alpha = tes * np.exp(-tes / ft2s)` | `alpha = tes * np.exp(-tes * fr2s)` | |
| `make_optcom(..., t2s=None, combmode="t2s")` | `make_optcom(..., r2s=None, combmode="r2s")` | `combmode` string renamed; `t2s` parameter renamed to `r2s` |

---

### `utils.py`

- **Add** `check_r2s_values(r2s_map)`: validates R2\* in s⁻¹. Warns if median is outside the
  expected range (~2–200 s⁻¹ for typical brain tissue). Raises on obviously invalid values
  (e.g. all zeros, all negative).
- **Keep** `check_t2s_values` unchanged — used only inside the deprecated `--t2smap` code path.
- **`load_mask`** signature: add `r2smap=None` parameter alongside existing `t2smap=None`.
  Return type changes from `(mask_img, t2s | None)` to `(mask_img, r2s | None)`.
  - `t2smap` given: load → `check_t2s_values` → `r2s = 1 / t2s` → return `r2s`
  - `r2smap` given: load → `check_r2s_values` → return `r2s`

---

### `workflows/tedana.py` and `workflows/t2smap.py`

- Add `--r2smap` to an `argparse.mutually_exclusive_group` with `--t2smap`.
- When `--t2smap` is parsed: emit `DeprecationWarning` directing users to `--r2smap`.
- The `t2smap` / `r2smap` function parameters remain as file-path strings (not data arrays);
  both are threaded through to `load_mask`.
- Rename all internal data variables: `t2s_full → r2s_full`, `t2s_limited → r2s_limited`,
  `t2s_var → r2s_var`, `t2s_s0_covar → r2s_s0_covar`.
- Update all `save_file` / `get_name` keys (see `outputs.json` table below).
- Log message: `"Computing T2* map"` → `"Computing R2* map"`.
- `combmode` default: `"t2s"` → `"r2s"`; CLI `choices=["t2s"]` → `choices=["r2s"]`.
- `decay.modify_t2s_s0_maps(t2s=…)` → `decay.modify_r2s_s0_maps(r2s=…)`.
- `decay.rmse_of_fit_decay_ts(t2s=…)` → `(r2s=…)`.

---

### `resources/config/outputs.json`

| Old key | New key | Old filename stem | New filename stem |
|---------|---------|-------------------|------------------|
| `"t2star img"` | `"r2star img"` | `T2starmap` | `R2starmap` |
| `"limited t2star img"` | `"limited r2star img"` | `desc-limited_T2starmap` | `desc-limited_R2starmap` |
| `"t2star variance img"` | `"r2star variance img"` | `stat-variance_desc-t2star_statmap` | `stat-variance_desc-r2star_statmap` |
| `"t2star-s0 covariance img"` | `"r2star-s0 covariance img"` | `stat-covariance_desc-t2star+s0_statmap` | `stat-covariance_desc-r2star+s0_statmap` |

---

### `reporting/`

**`static_figures.py`:**
- `plot_t2star_and_s0` → `plot_r2star_and_s0`
- Figure filenames: `t2star_brain.svg` → `r2star_brain.svg`,
  `t2star_histogram.svg` → `r2star_histogram.svg`,
  `stat-variance_desc-t2star_statmap.svg` → `stat-variance_desc-r2star_statmap.svg`
- Output description keys in `imgs` list: `"t2star variance img"` → `"r2star variance img"`, etc.

**`html_report.py`:**
- Rename all `t2star_*` local variables to `r2star_*`.
- Update Jinja template keys: `t2starBrainPlot` → `r2starBrainPlot`,
  `t2starHistogram` → `r2starHistogram`, `t2starExists` → `r2starExists`,
  `t2starVariancePlot` → `r2starVariancePlot`.
- Update the HTML template file accordingly.

---

### `rica.py`

- Change `"T2starmap.nii"` → `"R2starmap.nii"`.

---

### `docs/outputs.rst`

- Update all T2\* output descriptions, units (seconds → s⁻¹), and file name examples to R2\*.

---

### Tests

| File | Change |
|------|--------|
| `test_t2smap.py` | `T2starmap.nii.gz` → `R2starmap.nii.gz`; `desc-limited_T2starmap.nii.gz` → `desc-limited_R2starmap.nii.gz` |
| `test_decay.py` | Rename `t2s` variables; invert numeric values (e.g. `t2s=0.030` → `r2s=33.3`) |
| `test_combine.py` | Rename `t2s` parameter; invert numeric values |
| `test_utils.py` | Update `check_t2s_values` tests; add `check_r2s_values` tests |
| `test_io.py` | `"t2star img"` → `"r2star img"` |
| `test_integration.py` | Switch `t2smap=` argument to `r2smap=` with inverted values; check `R2starmap.nii.gz` is produced |

---

## Error handling and edge cases

- **Negative R2\* from bad fit**: caught by `r2s[r2s <= 0] = 1/500.0` in `modify_r2s_s0_maps`.
- **Infinite R2\* from T2\*=0**: caught by `r2s[np.isinf(r2s)] = 1.0`.
- **Numerical underflow in `exp(-te * r2s)`**: caught by `_apply_r2s_ceiling`.
- **Anomalously small R2\* outliers** (very long apparent T2\*): caught by the 0.5th-percentile
  floor in `r2s_limited`.
- **`--t2smap` + `--r2smap` together**: `argparse` mutually exclusive group raises error at parse
  time with a clear message.
- **`--t2smap` used**: `DeprecationWarning` logged; data still processed correctly.

---

## Implementation order

Work module-by-module in dependency order so each file compiles cleanly before its callers are
updated. Suggested order:

1. `decay.py` (no external tedana dependencies in changed signatures)
2. `combine.py`
3. `utils.py` (add `check_r2s_values`, update `load_mask`)
4. `resources/config/outputs.json`
5. `workflows/tedana.py` and `workflows/t2smap.py`
6. `reporting/static_figures.py`, `reporting/html_report.py`, HTML template
7. `rica.py`
8. `docs/outputs.rst`
9. Tests (update after all production code is consistent)
