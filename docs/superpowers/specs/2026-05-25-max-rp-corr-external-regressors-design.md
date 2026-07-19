# Design: `max_RP_corr` via External Regressors

**Date:** 2026-05-25
**Branch:** aroma-metrics

## Summary

Make `max_RP_corr` calculable from columns in the external regressors DataFrame, as
declared in the decision tree configuration file. Remove the dedicated `motpars`
parameter pathway. Make the set of metrics calculable from external regressors easily
expandable via a dispatch dictionary.

---

## Background

`max_RP_corr` is an AROMA-derived metric: the mean, over 1000 random 90 %-subsamples,
of the maximum absolute Pearson correlation between each component time series and a
model built from motion parameters (raw params + derivatives + ±1 TR time shifts).

Currently the metric is computed in `collect.generate_metrics` via a dedicated
`motpars` argument (a hard-coded (T × 6) array). The decision tree `external_aroma.json`
already declares `"statistic": "max_rp_corr"` in its `external_regressor_config`, but
the code does not yet support this — the validator only accepts `"f"` and
`fit_regressors` raises on any other value.

---

## Goals

1. Remove the `motpars`-based `max_RP_corr` path entirely.
2. Support `"statistic": "max_rp_corr"` in `external_regressor_config`, using the named
   regressor columns from the external regressors DataFrame as motion parameters.
3. Generalize `calculate_max_rp_corr` to accept N columns (not just 6).
4. Honor the `detrend` field for `max_rp_corr` (default: no detrending).
5. Make adding future external-regressor statistic types require touching only one file
   (`external.py`) plus a one-line validator update.

---

## Architecture

Three concerns currently require separate edits when adding a new statistic type:

| Concern | File today |
|---|---|
| Validate statistic name | `selection/component_selector.py` |
| Register output metric names in dependency graph | `metrics/_utils.py` |
| Dispatch computation | `metrics/external.py` |

After this change the first two collapse into `external.py`. The validator imports
`STATISTIC_HANDLERS.keys()` from `external.py`, so it updates automatically. Only the
handler function and its registration need to be written when adding a new statistic.

---

## Detailed Changes

### `tedana/metrics/external.py`

**`calculate_max_rp_corr`**
- Rename parameter `motpars` → `regressors` (keyword-only).
- Remove the `rp6.shape[1] != 6` check; keep only the row-count match check.
- Algorithm generalizes: N input columns → 2N with derivatives → 6N with ±1 TR shifts.
  Output is unchanged: a (C,) array in [0, 1].

**`fit_max_rp_corr_to_regressors` (new)**
Signature matches all other handlers:
```
fit_max_rp_corr_to_regressors(
    component_table, external_regressors, config, mixing, detrend_regressors
) -> component_table
```
Steps:
1. Extract `config["regressors"]` columns from `external_regressors` → (T × N) array.
2. If `config["detrend"]` is `True` or a positive int: residualize both `mixing` and
   the regressor array against `detrend_regressors` using least-squares projection:
   `Y_resid = Y - L @ lstsq(L, Y)`, where L is the (T × P) Legendre basis matrix.
3. Call `calculate_max_rp_corr(mixing=..., regressors=...)`.
4. Store result as `"max_RP_corr {regress_ID} model"` column in `component_table`.

**`STATISTIC_HANDLERS` (new module-level dict)**
```python
STATISTIC_HANDLERS = {
    "f": fit_mixing_to_regressors,
    "max_rp_corr": fit_max_rp_corr_to_regressors,
}
```
All handler functions share the same 5-argument signature.

**`fit_regressors`**
Replace the `if statistic == "f" / else: raise` block with:
```python
handler = STATISTIC_HANDLERS.get(statistic)
if handler is None:
    raise ValueError(f"statistic '{statistic}' is not valid. ...")
component_table = handler(component_table, external_regressors, config, mixing, detrend_regressors)
```

**`add_external_dependencies` (moved from `_utils.py`)**
Add a branch for `"max_rp_corr"` that registers
`"max_RP_corr {regress_ID} model"` in `dependency_config["dependencies"]` with
`["external regressors"]` as its dependency.

---

### `tedana/metrics/_utils.py`

Remove `add_external_dependencies` (moved to `external.py`).

---

### `tedana/metrics/collect.py`

- Remove `motpars` from the `generate_metrics` signature and docstring.
- Remove the `if "max_RP_corr" in required_metrics:` block and the associated
  `motpars is None` guard.
- Remove `max_RP_corr` from `get_metadata`.
- Change import of `add_external_dependencies` from `_utils` to `external`
  (already imported as a module).

---

### `tedana/resources/config/metrics.json`

- Remove `"motpars"` from the top-level `"inputs"` list.
- Remove the `"max_RP_corr"` entry from `"dependencies"`.

---

### `tedana/selection/component_selector.py`

Replace:
```python
statistic_key_options = set("f")
```
with:
```python
from tedana.metrics.external import STATISTIC_HANDLERS
statistic_key_options = set(STATISTIC_HANDLERS.keys())
```

---

## Data Flow

```
decision tree JSON
  └─ external_regressor_config[i]["statistic"] = "max_rp_corr"
       └─ validate_tree (component_selector.py)
            statistic ∈ STATISTIC_HANDLERS.keys()  ✓
       └─ generate_metrics (collect.py)
            add_external_dependencies (external.py)
              → registers "max_RP_corr motion model" in dependency graph
            fit_regressors (external.py)
              ├─ builds detrend_regressors (Legendre basis)
              └─ STATISTIC_HANDLERS["max_rp_corr"](...)
                   ├─ extract config["regressors"] columns → (T × N) array
                   ├─ if detrend: residualize mixing + regressors vs. Legendre basis
                   └─ calculate_max_rp_corr(mixing, regressors)
                        builds (T × 6N) model, 1000-split correlation → (C,) array
                   stores "max_RP_corr {regress_ID} model" in component_table
```

---

## Error Handling

- `validate_tree` already rejects `partial_models` when `statistic != "f"` — no change.
- `validate_extern_regress` already expands regex patterns and confirms all named
  columns exist in the DataFrame — no change.
- `calculate_max_rp_corr` keeps the row-count mismatch check; removes 6-column check.
- `fit_regressors` raises a clear `ValueError` if the statistic is not in
  `STATISTIC_HANDLERS` (explicit guard before dict lookup).

---

## Testing

**`test_calculate_max_rp_corr`**
- Rename `motpars=` → `regressors=` in all existing calls.
- Remove the "wrong column count" test.
- Add tests for N ≠ 6 (e.g., N=3, N=12) to confirm generalization.

**`test_fit_max_rp_corr_to_regressors` (new)**
- `detrend=False`: raw inputs, output shape (C,) in component_table.
- `detrend=True`: residualized inputs, output differs from `detrend=False` result.
- Column name mismatch between `config["regressors"]` and DataFrame: should raise
  via existing `validate_extern_regress`.

**`test_fit_regressors`**
- Add a case with `statistic="max_rp_corr"` config to confirm dispatch end-to-end.

**`test_generate_metrics`**
- Remove `motpars=` argument cases.
- Confirm `"max_RP_corr"` no longer appears as a standalone column.

---

## Extending to New Statistic Types

To add a future statistic (e.g., `"cross_corr"`):

1. Write `fit_cross_corr_to_regressors(...)` in `external.py` with the standard 5-arg
   signature.
2. Add `"cross_corr": fit_cross_corr_to_regressors` to `STATISTIC_HANDLERS`.
3. Add the output metric name pattern to `add_external_dependencies` in `external.py`.

The validator in `component_selector.py` picks up the new name automatically.
No other files need to change.
