# `max_RP_corr` via External Regressors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded `motpars`-based `max_RP_corr` metric with a generalized external-regressor-driven version declared in the decision tree JSON config.

**Architecture:** `STATISTIC_HANDLERS` in `external.py` maps statistic names to handler functions with a shared 5-argument signature; `fit_regressors` dispatches through this dict; the validator in `component_selector.py` imports the dict's keys so new statistics are auto-discovered. `add_external_dependencies` (moved from `_utils.py` to `external.py`) registers the correct output metric names for each statistic type.

**Tech Stack:** Python, NumPy, pandas, pytest. Run tests with `micromamba run -n tedanapy pytest`.

---

## File Map

| File | Change |
|---|---|
| `tedana/metrics/external.py` | Generalize `calculate_max_rp_corr`; add `fit_max_rp_corr_to_regressors`; add `STATISTIC_HANDLERS`; update `fit_regressors` dispatch; receive `add_external_dependencies` moved from `_utils.py` |
| `tedana/metrics/_utils.py` | Remove `add_external_dependencies` |
| `tedana/metrics/collect.py` | Remove `motpars` param, `max_RP_corr` calculation block, and `max_RP_corr` metadata; update imports |
| `tedana/resources/config/metrics.json` | Remove `"motpars"` input and `"max_RP_corr"` dependency entry |
| `tedana/selection/component_selector.py` | Import `STATISTIC_HANDLERS`; replace hard-coded `statistic_key_options = set("f")` |
| `tedana/tests/test_external_metrics.py` | Update `TestCalculateMaxRpCorr`; add `TestFitMaxRpCorrToRegressors`; update `test_fit_regressors` |
| `tedana/tests/test_metrics_utils.py` | Update import of `add_external_dependencies`; add `max_rp_corr` sub-test |
| `tedana/tests/test_component_selector.py` | Update invalid-statistic match pattern |

---

### Task 1: Generalize `calculate_max_rp_corr`

Remove the hard-coded 6-column restriction and rename the `motpars` parameter to `regressors`. The algorithm generalizes naturally: N input columns → 2N with derivatives → 6N with ±1 TR shifts.

**Files:**
- Modify: `tedana/metrics/external.py:670-740`
- Test: `tedana/tests/test_external_metrics.py:784-847`

- [ ] **Step 1: Update `TestCalculateMaxRpCorr` in the test file**

Replace the entire `TestCalculateMaxRpCorr` class (lines 784–847) in `tedana/tests/test_external_metrics.py` with:

```python
class TestCalculateMaxRpCorr:
    """Tests for external.calculate_max_rp_corr."""

    def test_output_shape(self):
        """Returns a 1-D array of length n_components."""
        rng = np.random.default_rng(0)
        n_vols, n_components = 100, 5
        mixing = rng.standard_normal((n_vols, n_components))
        regressors = rng.standard_normal((n_vols, 6))
        np.random.seed(42)
        result = external.calculate_max_rp_corr(mixing=mixing, regressors=regressors)
        assert result.shape == (n_components,)

    def test_output_range(self):
        """All values are in [0, 1] (absolute Pearson correlation)."""
        rng = np.random.default_rng(1)
        mixing = rng.standard_normal((100, 4))
        regressors = rng.standard_normal((100, 6))
        np.random.seed(0)
        result = external.calculate_max_rp_corr(mixing=mixing, regressors=regressors)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_correlated_component_has_higher_score(self):
        """A component built from a regressor has a higher score than white noise."""
        rng = np.random.default_rng(2)
        n_vols = 150
        regressors = rng.standard_normal((n_vols, 6))
        correlated = regressors[:, 0:1] + 0.1 * rng.standard_normal((n_vols, 1))
        noise = rng.standard_normal((n_vols, 1))
        mixing = np.hstack([correlated, noise])
        np.random.seed(0)
        result = external.calculate_max_rp_corr(mixing=mixing, regressors=regressors)
        assert result[0] > result[1], (
            f"Correlated component score ({result[0]:.3f}) should exceed "
            f"noise component score ({result[1]:.3f})"
        )

    def test_generalized_n_columns(self):
        """Accepts any number of columns, not just 6."""
        rng = np.random.default_rng(6)
        n_vols, n_components = 100, 3
        mixing = rng.standard_normal((n_vols, n_components))
        for n_cols in [1, 3, 12]:
            regressors = rng.standard_normal((n_vols, n_cols))
            np.random.seed(0)
            result = external.calculate_max_rp_corr(mixing=mixing, regressors=regressors)
            assert result.shape == (n_components,)
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_wrong_regressors_shape_n_rows(self):
        """Raises ValueError when regressors row count does not match mixing."""
        rng = np.random.default_rng(4)
        mixing = rng.standard_normal((100, 3))
        regressors_bad = rng.standard_normal((80, 6))
        with pytest.raises(ValueError, match="Number of rows"):
            external.calculate_max_rp_corr(mixing=mixing, regressors=regressors_bad)

    def test_keyword_only_args(self):
        """calculate_max_rp_corr requires keyword arguments."""
        rng = np.random.default_rng(5)
        mixing = rng.standard_normal((100, 2))
        regressors = rng.standard_normal((100, 6))
        with pytest.raises(TypeError):
            external.calculate_max_rp_corr(mixing, regressors)  # positional not allowed
```

- [ ] **Step 2: Run tests to confirm failures**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_external_metrics.py::TestCalculateMaxRpCorr -v
```

Expected: several tests fail with `TypeError: calculate_max_rp_corr() got an unexpected keyword argument 'regressors'`.

- [ ] **Step 3: Replace `calculate_max_rp_corr` in `external.py`**

Replace lines 670–740 of `tedana/metrics/external.py` with:

```python
def calculate_max_rp_corr(*, mixing: np.ndarray, regressors: np.ndarray) -> np.ndarray:
    """Calculate the maximum regressor-correlation (max_RP_corr) for each component.

    Computes the mean, over 1000 random 90%-subsamples of timepoints, of the
    maximum absolute Pearson correlation between each component time series and
    a 6*N-regressor model built from N input regressors (raw N parameters, their
    derivatives, and both sets time-shifted ±1 TR).  Correlations are computed for
    the raw time series and their element-wise squares, giving 12*N total
    comparisons per split.

    Parameters
    ----------
    mixing : (T x C) array_like
        ICA mixing matrix where T is time points and C is components.
    regressors : (T x N) array_like
        Regressor time series with T timepoints and N columns.

    Returns
    -------
    max_rp_corr : (C,) numpy.ndarray
        Maximum regressor-correlation score for each component.
        Values are in [0, 1].

    Raises
    ------
    ValueError
        If ``regressors`` is not 2-D or its row count does not match ``mixing``.
    """
    mixing = np.asarray(mixing, dtype=float)
    rp = np.asarray(regressors, dtype=float)

    if rp.ndim != 2:
        raise ValueError(f"regressors must be a 2-D array, got shape {rp.shape}")

    if rp.shape[0] != mixing.shape[0]:
        raise ValueError(
            f"Number of rows in mixing ({mixing.shape[0]}) does not match "
            f"number of rows in regressors ({rp.shape[0]})."
        )

    _, n_params = rp.shape

    # Derivatives (zero-padded at t=0)
    rp_der = np.vstack((np.zeros(n_params), np.diff(rp, axis=0)))
    rp2 = np.hstack((rp, rp_der))

    # Time-shifted versions (±1 TR, zero-padded at boundaries)
    rp2_1fw = np.vstack((np.zeros(2 * n_params), rp2[:-1]))
    rp2_1bw = np.vstack((rp2[1:], np.zeros(2 * n_params)))
    rp_model = np.hstack((rp2, rp2_1fw, rp2_1bw))  # (T, 6 * n_params)

    n_volumes, n_components = mixing.shape
    n_rows_to_choose = int(round(0.9 * n_volumes))
    n_splits = 1000

    max_correlations = np.empty((n_splits, n_components))
    for i_split in range(n_splits):
        chosen_rows = np.random.choice(n_volumes, size=n_rows_to_choose, replace=False)

        correl_nonsquared = utils.cross_correlation(mixing[chosen_rows], rp_model[chosen_rows])
        correl_squared = utils.cross_correlation(
            mixing[chosen_rows] ** 2, rp_model[chosen_rows] ** 2
        )
        correl_both = np.hstack((correl_squared, correl_nonsquared))
        max_correlations[i_split] = np.nanmax(np.abs(correl_both), axis=1)

    max_rp_corr = np.nanmean(max_correlations, axis=0)
    return max_rp_corr
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_external_metrics.py::TestCalculateMaxRpCorr -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tedana/metrics/external.py tedana/tests/test_external_metrics.py
git commit -m "feat: generalize calculate_max_rp_corr to accept N regressor columns"
```

---

### Task 2: Add `fit_max_rp_corr_to_regressors`

This handler extracts the configured regressor columns, optionally residualizes them against the Legendre polynomial detrending basis, and stores `"max_RP_corr {regress_ID} model"` in the component table.

**Files:**
- Modify: `tedana/metrics/external.py` (append after `calculate_max_rp_corr`)
- Test: `tedana/tests/test_external_metrics.py` (add `TestFitMaxRpCorrToRegressors` before `TestCalculateMaxRpCorr`)

- [ ] **Step 1: Add `TestFitMaxRpCorrToRegressors` to the test file**

Insert the following class just before the `# calculate_max_rp_corr` comment block in `tedana/tests/test_external_metrics.py`:

```python
# fit_max_rp_corr_to_regressors
# -----------------------------


class TestFitMaxRpCorrToRegressors:
    """Tests for external.fit_max_rp_corr_to_regressors."""

    def _make_inputs(self, n_vols=100, n_components=5, n_regressors=6, seed=0):
        """Return (component_table, external_regressors, config, mixing, detrend_regressors)."""
        rng = np.random.default_rng(seed)
        mixing = rng.standard_normal((n_vols, n_components))
        regressor_names = [f"mot_{i}" for i in range(n_regressors)]
        ext_reg = pd.DataFrame(
            rng.standard_normal((n_vols, n_regressors)), columns=regressor_names
        )
        config = {
            "regress_ID": "motion",
            "regressors": regressor_names,
            "detrend": False,
            "statistic": "max_rp_corr",
        }
        import tedana.utils as tedana_utils
        legendre_arr = tedana_utils.create_legendre_polynomial_basis_set(n_vols, dtrank=None)
        detrend_regressors = pd.DataFrame(
            legendre_arr, columns=[f"baseline {i}" for i in range(legendre_arr.shape[1])]
        )
        component_table = pd.DataFrame(
            {"Component": [f"ICA_{i:02d}" for i in range(n_components)]}
        )
        return component_table, ext_reg, config, mixing, detrend_regressors

    def test_output_column_exists(self):
        """Adds 'max_RP_corr motion model' column with correct length."""
        component_table, ext_reg, config, mixing, detrend_regressors = self._make_inputs()
        np.random.seed(0)
        result = external.fit_max_rp_corr_to_regressors(
            component_table, ext_reg, config, mixing, detrend_regressors
        )
        assert "max_RP_corr motion model" in result.columns
        assert len(result["max_RP_corr motion model"]) == mixing.shape[1]
        assert np.all(result["max_RP_corr motion model"] >= 0)
        assert np.all(result["max_RP_corr motion model"] <= 1)

    def test_column_name_uses_regress_id(self):
        """Output column name is 'max_RP_corr {regress_ID} model'."""
        component_table, ext_reg, config, mixing, detrend_regressors = self._make_inputs()
        config["regress_ID"] = "physio"
        np.random.seed(0)
        result = external.fit_max_rp_corr_to_regressors(
            component_table, ext_reg, config, mixing, detrend_regressors
        )
        assert "max_RP_corr physio model" in result.columns

    def test_detrend_true_changes_result(self):
        """detrend=True produces a different result than detrend=False."""
        component_table, ext_reg, config, mixing, detrend_regressors = self._make_inputs(seed=1)
        np.random.seed(0)
        config["detrend"] = False
        result_no_detrend = external.fit_max_rp_corr_to_regressors(
            component_table.copy(), ext_reg, config, mixing, detrend_regressors
        )
        np.random.seed(0)
        config["detrend"] = True
        result_detrend = external.fit_max_rp_corr_to_regressors(
            component_table.copy(), ext_reg, config, mixing, detrend_regressors
        )
        assert not np.allclose(
            result_no_detrend["max_RP_corr motion model"].values,
            result_detrend["max_RP_corr motion model"].values,
        ), "detrend=True and detrend=False should produce different results"
```

- [ ] **Step 2: Run tests to confirm failures**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_external_metrics.py::TestFitMaxRpCorrToRegressors -v
```

Expected: 3 tests fail with `AttributeError: module 'tedana.metrics.external' has no attribute 'fit_max_rp_corr_to_regressors'`.

- [ ] **Step 3: Add `fit_max_rp_corr_to_regressors` to `external.py`**

Append the following after `calculate_max_rp_corr` (at the end of `tedana/metrics/external.py`):

```python


def fit_max_rp_corr_to_regressors(
    component_table: pd.DataFrame,
    external_regressors: pd.DataFrame,
    external_regressor_config: Dict,
    mixing: npt.NDArray,
    detrend_regressors: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate max_RP_corr from specified external regressor columns.

    Parameters
    ----------
    component_table : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component,
        with a column for each metric.
    external_regressors : (T x R) :obj:`pandas.DataFrame`
        External regressor time series.
    external_regressor_config : :obj:`dict`
        Single external regressor config entry. Required keys:
        ``regress_ID``, ``regressors``, ``detrend``, ``statistic``.
    mixing : (T x C) array_like
        ICA mixing matrix.
    detrend_regressors : (T x P) :obj:`pandas.DataFrame`
        Legendre polynomial basis for optional detrending.

    Returns
    -------
    component_table : (C x X) :obj:`pandas.DataFrame`
        Input table with ``"max_RP_corr {regress_ID} model"`` column added.
    """
    regress_id = external_regressor_config["regress_ID"]
    rp_arr = external_regressors[external_regressor_config["regressors"]].values

    apply_detrend = external_regressor_config["detrend"] is True or (
        isinstance(external_regressor_config["detrend"], int)
        and external_regressor_config["detrend"] > 0
    )

    if apply_detrend:
        L = detrend_regressors.values
        mixing_use = mixing - L @ np.linalg.lstsq(L, mixing, rcond=None)[0]
        rp_use = rp_arr - L @ np.linalg.lstsq(L, rp_arr, rcond=None)[0]
        LGR.info(
            f"max_RP_corr for {regress_id} detrended with "
            f"{L.shape[1]} Legendre Polynomial regressors."
        )
    else:
        mixing_use = mixing
        rp_use = rp_arr

    max_rp_corr = calculate_max_rp_corr(mixing=mixing_use, regressors=rp_use)
    component_table[f"max_RP_corr {regress_id} model"] = max_rp_corr
    return component_table
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_external_metrics.py::TestFitMaxRpCorrToRegressors -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tedana/metrics/external.py tedana/tests/test_external_metrics.py
git commit -m "feat: add fit_max_rp_corr_to_regressors handler"
```

---

### Task 3: Add `STATISTIC_HANDLERS` and update `fit_regressors` dispatch

Replace the `if statistic == "f" / else: raise` block in `fit_regressors` with a dispatch through `STATISTIC_HANDLERS`.

**Files:**
- Modify: `tedana/metrics/external.py:313-329` (dispatch block) and append `STATISTIC_HANDLERS` after `fit_max_rp_corr_to_regressors`
- Test: `tedana/tests/test_external_metrics.py` (update `test_fit_regressors`)

- [ ] **Step 1: Add a `max_rp_corr` dispatch test to `test_fit_regressors`**

In `tedana/tests/test_external_metrics.py`, add the following block at the end of `test_fit_regressors` (before the closing of the function, after line 528):

```python
    # max_rp_corr dispatch: six motion columns from external_regressors
    caplog.clear()
    mot_col_names = [
        "Mot_X", "Mot_Y", "Mot_Z", "Mot_Pitch", "Mot_Roll", "Mot_Yaw"
    ]
    maxrp_config = [
        {
            "regress_ID": "motion",
            "info": "max_RP_corr test",
            "report": "max_RP_corr report",
            "detrend": False,
            "statistic": "max_rp_corr",
            "regressors": mot_col_names,
        }
    ]
    # validate_extern_regress expands any regex; names here are literals so pass directly
    external_regressors_loaded, _ = sample_external_regressors()
    component_table_rp = sample_comptable(mixing.shape[1])
    np.random.seed(0)
    component_table_rp = external.fit_regressors(
        component_table_rp, external_regressors_loaded, maxrp_config, mixing
    )
    assert "max_RP_corr motion model" in component_table_rp.columns
    assert np.all(component_table_rp["max_RP_corr motion model"] >= 0)
    assert np.all(component_table_rp["max_RP_corr motion model"] <= 1)
```

- [ ] **Step 2: Run new test sub-case to confirm it fails**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_external_metrics.py::test_fit_regressors -v
```

Expected: test fails on the new max_rp_corr block with `ValueError: statistic for motion … is max_rp_corr, which is not valid`.

- [ ] **Step 3: Append `STATISTIC_HANDLERS` after `fit_max_rp_corr_to_regressors` in `external.py`**

At the very end of `tedana/metrics/external.py`, append:

```python


STATISTIC_HANDLERS = {
    "f": fit_mixing_to_regressors,
    "max_rp_corr": fit_max_rp_corr_to_regressors,
}
```

- [ ] **Step 4: Replace the dispatch block in `fit_regressors`**

In `tedana/metrics/external.py`, replace lines 313–329 (the `if statistic == "f" / else: raise` block) with:

```python
        statistic = external_regressor_config[config_idx]["statistic"].lower()
        handler = STATISTIC_HANDLERS.get(statistic)
        if handler is None:
            raise ValueError(
                f"statistic for {regress_id} external regressors in decision tree is "
                f"{statistic}, which is not valid."
            )
        component_table = handler(
            component_table,
            external_regressors,
            external_regressor_config[config_idx],
            mixing,
            detrend_regressors,
        )
```

- [ ] **Step 5: Run all `fit_regressors` tests to confirm they pass**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_external_metrics.py::test_fit_regressors -v
```

Expected: all sub-cases pass, including the existing "invalid statistic" case (error message wording is unchanged — the `else: raise` is replaced by `if handler is None: raise` with the same message).

- [ ] **Step 6: Commit**

```bash
git add tedana/metrics/external.py tedana/tests/test_external_metrics.py
git commit -m "feat: add STATISTIC_HANDLERS dispatch dict to fit_regressors"
```

---

### Task 4: Move `add_external_dependencies` to `external.py` and add `max_rp_corr` branch

**Files:**
- Modify: `tedana/metrics/external.py` (append function)
- Modify: `tedana/metrics/_utils.py` (remove function)
- Modify: `tedana/metrics/collect.py` (update import)
- Test: `tedana/tests/test_metrics_utils.py` (update import + add `max_rp_corr` sub-test)

- [ ] **Step 1: Update `test_metrics_utils.py`**

In `tedana/tests/test_metrics_utils.py`, change line 7:

```python
from tedana.metrics._utils import (
    add_external_dependencies,
    dependency_resolver,
    determine_signs,
    flip_components,
    get_value_thresholds,
)
```

to:

```python
from tedana.metrics._utils import (
    dependency_resolver,
    determine_signs,
    flip_components,
    get_value_thresholds,
)
from tedana.metrics.external import add_external_dependencies
```

Also add the following sub-test at the end of `test_add_external_dependencies` (after line 167):

```python
    # Test with max_rp_corr statistic
    external_regressor_config_rp = [
        {
            "regress_ID": "motion",
            "statistic": "max_rp_corr",
            "info": "Motion",
            "report": "Motion correlation",
        }
    ]
    updated_config_rp = add_external_dependencies(
        {"inputs": ["data"], "dependencies": {}}, external_regressor_config_rp
    )
    assert "max_RP_corr motion model" in updated_config_rp["dependencies"]
    assert updated_config_rp["dependencies"]["max_RP_corr motion model"] == [
        "external regressors"
    ]
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_metrics_utils.py::test_add_external_dependencies -v
```

Expected: `ImportError: cannot import name 'add_external_dependencies' from 'tedana.metrics.external'`.

- [ ] **Step 3: Append `add_external_dependencies` to `external.py`**

Append after `STATISTIC_HANDLERS` in `tedana/metrics/external.py`:

```python


def add_external_dependencies(
    dependency_config: Dict, external_regressor_config: List[Dict]
) -> Dict:
    """Add dependency information when external regressors are inputted.

    Parameters
    ----------
    dependency_config : :obj:`dict`
        A dictionary stored in ``./config/metrics.json`` with information on
        all internally defined metrics.
    external_regressor_config : :obj:`list[dict]`
        A list of dictionaries with info for fitting external regressors to
        component time series.

    Returns
    -------
    dependency_config : :obj:`dict`
        Updated dictionary with external-regressor metric dependencies added.
    """
    dependency_config["inputs"].append("external regressors")

    for config_idx in range(len(external_regressor_config)):
        regress_id = external_regressor_config[config_idx]["regress_ID"]
        statistic = external_regressor_config[config_idx]["statistic"].lower()

        if statistic == "f":
            model_names = [regress_id]
            if "partial_models" in set(external_regressor_config[config_idx].keys()):
                partial_keys = external_regressor_config[config_idx]["partial_models"].keys()
                for key_name in partial_keys:
                    model_names.append(f"{regress_id} {key_name} partial")
            for model_name in model_names:
                for stat_type in ["Fstat", "R2stat", "pval"]:
                    dependency_config["dependencies"][f"{stat_type} {model_name} model"] = [
                        "external regressors"
                    ]
        elif statistic == "max_rp_corr":
            dependency_config["dependencies"][f"max_RP_corr {regress_id} model"] = [
                "external regressors"
            ]

    return dependency_config
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_metrics_utils.py::test_add_external_dependencies -v
```

Expected: pass.

- [ ] **Step 5: Remove `add_external_dependencies` from `_utils.py`**

Delete lines 13–52 of `tedana/metrics/_utils.py` (the entire `add_external_dependencies` function and its docstring). The file should now start with `def dependency_resolver(`.

Keep the `from typing import Dict, List` import — `dependency_resolver`'s type annotations still use both.

- [ ] **Step 6: Update `collect.py` import**

In `tedana/metrics/collect.py`, change lines 14–19:

```python
from tedana.metrics._utils import (
    add_external_dependencies,
    dependency_resolver,
    determine_signs,
    flip_components,
)
```

to:

```python
from tedana.metrics._utils import (
    dependency_resolver,
    determine_signs,
    flip_components,
)
```

Then find the call to `add_external_dependencies` in `collect.py` (line ~110) and change it from:

```python
        dependency_config = add_external_dependencies(dependency_config, external_regressor_config)
```

to:

```python
        dependency_config = external.add_external_dependencies(dependency_config, external_regressor_config)
```

(`external` is already imported as a module via `from tedana.metrics import dependence, external`.)

Also remove the now-duplicate `dependency_config["inputs"].append("external regressors")` call on line ~111, since `add_external_dependencies` now handles that internally. Verify by checking that the two lines around it read:

```python
        dependency_config = external.add_external_dependencies(dependency_config, external_regressor_config)
```

(The old code called `add_external_dependencies` and then also called `.append("external regressors")` separately — the new `add_external_dependencies` in `external.py` already does the append, so remove any duplicate `.append` call in `collect.py`.)

- [ ] **Step 7: Run the full external metrics test suite**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_external_metrics.py tedana/tests/test_metrics_utils.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add tedana/metrics/external.py tedana/metrics/_utils.py tedana/metrics/collect.py tedana/tests/test_metrics_utils.py
git commit -m "refactor: move add_external_dependencies to external.py, add max_rp_corr branch"
```

---

### Task 5: Update `component_selector.py` validator

Import `STATISTIC_HANDLERS` so valid statistic names are auto-discovered instead of hard-coded.

**Files:**
- Modify: `tedana/selection/component_selector.py:3-6` (imports) and `component_selector.py:271`
- Test: `tedana/tests/test_component_selector.py:387-395`

- [ ] **Step 1: Update the invalid-statistic test to use a flexible match**

In `tedana/tests/test_component_selector.py`, replace the match string at lines 387–395:

```python
    with pytest.raises(
        component_selector.TreeError,
        match=(
            "statistic in external_regressor_config 1 is corr. It must be one of the following: "
            "{'f'}\nExternal regressor dictionary cannot include partial_models "
            "if statistic is not F"
        ),
    ):
        component_selector.validate_tree(dicts_to_test("external_invalid_statistic"))
```

with:

```python
    with pytest.raises(
        component_selector.TreeError,
        match=r"statistic in external_regressor_config 1 is corr\. It must be one of the following: ",
    ):
        component_selector.validate_tree(dicts_to_test("external_invalid_statistic"))
```

The relaxed pattern avoids matching the exact set contents (which changes when we add `max_rp_corr`).

- [ ] **Step 2: Run the test to confirm it still passes (pre-condition check)**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_component_selector.py -k "external_invalid_statistic or validate_tree" -v
```

Expected: passes (the relaxed pattern still matches the current error).

- [ ] **Step 3: Update `component_selector.py`**

Add the following import to `tedana/selection/component_selector.py` at line 12 (after the existing `from tedana.utils import get_resource_path` line):

```python
from tedana.metrics.external import STATISTIC_HANDLERS
```

Then replace line 271:

```python
            statistic_key_options = set("f")
```

with:

```python
            statistic_key_options = set(STATISTIC_HANDLERS.keys())
```

Also remove the now-stale comment on the preceding line:

```python
            # Right now, "f" is the only option, but this leaves open the possibility
            #  to have additional options
```

- [ ] **Step 4: Verify `external_aroma.json` now passes validation**

Write a quick inline check to confirm the `external_aroma.json` tree (which uses `"statistic": "max_rp_corr"`) no longer raises a `TreeError`:

```bash
micromamba run -n tedanapy python -c "
from tedana.selection.component_selector import validate_tree
from tedana.io import load_json
import tedana.utils as u, os
path = os.path.join(u.get_resource_path(), 'decision_trees', 'external_aroma.json')
tree = load_json(path)
validate_tree(tree)
print('external_aroma.json validated OK')
"
```

Expected: `external_aroma.json validated OK`.

- [ ] **Step 5: Run full component selector tests**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_component_selector.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tedana/selection/component_selector.py tedana/tests/test_component_selector.py
git commit -m "feat: auto-discover valid statistic names from STATISTIC_HANDLERS"
```

---

### Task 6: Remove `motpars` from `collect.py` and `metrics.json`

**Files:**
- Modify: `tedana/metrics/collect.py:28-44` (signature), `collect.py:73-77` (docstring), `collect.py:435-443` (calculation block), `collect.py:810-823` (metadata block)
- Modify: `tedana/resources/config/metrics.json`

- [ ] **Step 1: Remove `motpars` from `generate_metrics` signature and docstring**

In `tedana/metrics/collect.py`:

1. Remove `motpars: Union[npt.NDArray, None] = None,` from the function signature (line 39).

2. Remove the following docstring block (lines 73–77):

```
    motpars : (T x 6) array_like or None, optional
        Motion parameters (rotation in radians, translation in mm) with the
        same number of timepoints as the mixing matrix.  Required when
        ``metrics`` includes ``"max_RP_corr"``.  Default is None.
```

- [ ] **Step 2: Remove the `max_RP_corr` calculation block**

Remove lines 435–443 of `tedana/metrics/collect.py` (the entire block):

```python
    # AROMA-derived motion-correlation metric
    if "max_RP_corr" in required_metrics:
        if motpars is None:
            raise ValueError("motpars must be provided to compute the max_RP_corr metric.")
        LGR.info("Calculating maximum motion-parameter correlation (max_RP_corr)")
        component_table["max_RP_corr"] = external.calculate_max_rp_corr(
            mixing=mixing,
            motpars=motpars,
        )
```

- [ ] **Step 3: Remove `max_RP_corr` from `get_metadata`**

Remove lines 810–823 in `tedana/metrics/collect.py` (the block):

```python
    if "max_RP_corr" in component_table:
        metric_metadata["max_RP_corr"] = {
            "LongName": "Maximum motion-parameter correlation",
            "Description": (
                "The mean, over 1000 random 90 %-subsamples of timepoints, of the maximum "
                "absolute Pearson correlation between a component time series and a "
                "36-regressor motion-parameter model (raw 6 parameters, their derivatives, "
                "and both sets shifted ±1 tr). "
                "Correlations are computed for the raw and squared time series, giving "
                "72 comparisons per split. "
                "Values near 1 indicate strong coupling with head motion (likely noise)."
            ),
            "Units": "arbitrary",
        }
```

- [ ] **Step 4: Remove `motpars` and `max_RP_corr` from `metrics.json`**

In `tedana/resources/config/metrics.json`:

1. Remove `"motpars"` from the `"inputs"` list (line 10).

2. Remove the `"max_RP_corr"` entry from `"dependencies"` (lines 84–87):

```json
        "max_RP_corr": [
            "mixing",
            "motpars"
        ],
```

- [ ] **Step 5: Run the collect and external metrics tests**

```bash
micromamba run -n tedanapy pytest tedana/tests/test_external_metrics.py tedana/tests/test_metrics_utils.py -v
```

Expected: all pass.

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
micromamba run -n tedanapy pytest tedana/tests/ -v --ignore=tedana/tests/test_integration.py -x
```

Expected: all pass. (Integration tests are excluded because they require external data downloads.)

- [ ] **Step 7: Commit**

```bash
git add tedana/metrics/collect.py tedana/resources/config/metrics.json
git commit -m "feat: remove motpars-based max_RP_corr in favor of external regressors path"
```

---

## Final Verification

After all tasks are complete, run the full non-integration test suite one more time:

```bash
micromamba run -n tedanapy pytest tedana/tests/ -v --ignore=tedana/tests/test_integration.py
```

All tests should pass. The `external_aroma.json` decision tree's `"statistic": "max_rp_corr"` entry is now fully supported end-to-end.
