# Internal Seconds Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace milliseconds with seconds as the internal unit for echo times and T2* maps throughout tedana, eliminating the convert-in/convert-out dance at the input and output boundaries.

**Architecture:** Flip `check_te_values()` and `check_t2s_values()` to output seconds instead of milliseconds; delete the now-unnecessary `sec2millisec()`/`millisec2sec()` utility functions and their output-boundary call sites; update all docstrings to reflect the new unit; update test fixtures to use second-scale values.

**Tech Stack:** Python, NumPy, pytest (via `uv run python -m pytest`). No new dependencies.

---

## File Map

| File | Change type |
|------|-------------|
| `tedana/utils.py` | Flip `check_te_values()` and `check_t2s_values()` output; delete `sec2millisec()` and `millisec2sec()` |
| `tedana/workflows/tedana.py` | Remove `utils.millisec2sec()` wrappers at lines 817 and 822 |
| `tedana/workflows/t2smap.py` | Remove `utils.millisec2sec()` wrappers at lines 518 and 531 |
| `tedana/decay.py` | Docstrings only: 5 occurrences of "in milliseconds" → "in seconds" |
| `tedana/combine.py` | Docstrings only: 2 occurrences of "in milliseconds" → "in seconds" |
| `tedana/metrics/dependence.py` | Docstrings only: 1 occurrence of "in milliseconds" → "in seconds" |
| `tedana/tests/test_utils.py` | Delete tests for deleted functions; update assertions in `test_check_te_values` and `test_check_t2s_values` to expect seconds |
| `tedana/tests/test_decay.py` | Update fixture TE array from ms-scale to s-scale |
| `tedana/tests/test_combine.py` | Update all TE fixture arrays from ms-scale to s-scale |

---

### Task 1: Update `test_check_te_values` to expect seconds output

**Files:**
- Modify: `tedana/tests/test_utils.py:511-585`

- [ ] **Step 1: Write the failing assertions**

Replace the body of `test_check_te_values` in `tedana/tests/test_utils.py` with:

```python
def test_check_te_values(caplog):
    """Ensure that check_te_values returns values in seconds."""
    # Values in seconds (preferred per BIDS) - should be returned as-is
    assert utils.check_te_values([0.015, 0.039, 0.063]) == [0.015, 0.039, 0.063]
    assert utils.check_te_values([0.15, 0.35, 0.55]) == [0.15, 0.35, 0.55]

    # EPTI echo times (48 echoes)
    epti_te_ms = [
        6.70,
        7.63,
        8.56,
        9.49,
        10.42,
        11.35,
        12.28,
        13.21,
        14.14,
        15.07,
        16.00,
        16.93,
        17.86,
        18.79,
        19.72,
        20.65,
        21.58,
        22.51,
        23.44,
        24.37,
        25.30,
        26.23,
        27.16,
        28.09,
        29.02,
        29.95,
        30.88,
        31.81,
        32.74,
        33.67,
        34.60,
        35.53,
        36.46,
        37.39,
        38.32,
        39.25,
        40.18,
        41.11,
        42.04,
        42.97,
        43.90,
        44.83,
        45.76,
        46.69,
        47.62,
        48.55,
        49.48,
        50.41,
    ]
    epti_te_sec = [te / 1000 for te in epti_te_ms]
    # Seconds input returns seconds unchanged
    assert utils.check_te_values(epti_te_sec) == epti_te_sec

    # Values in milliseconds (deprecated) - should be converted to seconds with warning
    assert utils.check_te_values([15, 39, 63]) == [0.015, 0.039, 0.063]
    assert (
        "TE values appear to be in milliseconds. Per BIDS convention, echo times should "
        "be provided in seconds. Support for millisecond TE values is deprecated and will "
        "be removed in a future version. Please provide TE values in seconds."
    ) in caplog.text
    assert utils.check_te_values([2, 3, 4]) == [0.002, 0.003, 0.004]

    # EPTI echo times in milliseconds (deprecated) - should be converted to seconds
    result = utils.check_te_values(epti_te_ms)
    np.testing.assert_allclose(result, epti_te_sec)

    # Check that the error is raised when TE values are in mixed units
    with pytest.raises(ValueError):
        utils.check_te_values([0.5, 2, 3])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tedana/tests/test_utils.py::test_check_te_values -x -q 2>&1 | tail -15
```

Expected: FAIL — assertions like `== [0.015, 0.039, 0.063]` fail because the function still returns `[15, 39, 63]`.

---

### Task 2: Flip `check_te_values()` to output seconds

**Files:**
- Modify: `tedana/utils.py:790-801`

- [ ] **Step 1: Update the function body**

In `tedana/utils.py`, replace the body of `check_te_values()` (lines 789–806):

```python
    te_values = np.array(te_values)
    if all((te_values > 0) & (te_values < 1)):
        # Values appear to be in seconds (expected per BIDS) - return as-is
        LGR.debug("TE values appear to be in seconds.")
        return te_values.tolist()
    elif all(te_values >= 1):
        # Values appear to be in milliseconds (deprecated) - convert to seconds
        LGR.warning(
            "TE values appear to be in milliseconds. Per BIDS convention, echo times should "
            "be provided in seconds. Support for millisecond TE values is deprecated and will "
            "be removed in a future version. Please provide TE values in seconds."
        )
        return (te_values / 1000).tolist()
    else:
        raise ValueError(
            "TE values must be positive and either all in seconds (values < 1, preferred per "
            "BIDS convention) or all in milliseconds (values >= 1, deprecated)."
        )
```

Also update the docstring for `check_te_values()` (lines 757–786):

```python
def check_te_values(te_values):
    """Check and convert TE values to seconds for internal use.

    This function checks if TE values are provided in seconds (preferred per
    BIDS convention) or milliseconds. Echo times are returned in seconds
    for internal processing.

    Parameters
    ----------
    te_values : list
        TE values to check. Per BIDS convention, these should be in seconds.

    Returns
    -------
    list
        TE values in seconds for internal use.

    Raises
    ------
    ValueError
        If TE values are not positive or appear to be in unexpected units.

    Notes
    -----
    The heuristic used is:

    - If all TE values are between 0 and 1: values are assumed to be in seconds
      (correct per BIDS), returned as-is
    - If all TE values are >= 1: values are assumed to be in milliseconds, a
      deprecation warning is logged, and values are converted to seconds
    - Mixed values or negative values raise an error

    """
```

- [ ] **Step 2: Run test to verify it passes**

```bash
uv run python -m pytest tedana/tests/test_utils.py::test_check_te_values -x -q 2>&1 | tail -10
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tedana/utils.py tedana/tests/test_utils.py
git commit -m "refactor: flip check_te_values() to output seconds instead of milliseconds"
```

---

### Task 3: Update `test_check_t2s_values` to expect seconds output

**Files:**
- Modify: `tedana/tests/test_utils.py:588-619`

- [ ] **Step 1: Write the failing assertions**

Replace the body of `test_check_t2s_values` in `tedana/tests/test_utils.py` with:

```python
def test_check_t2s_values(caplog):
    """Ensure that check_t2s_values returns values in seconds."""
    # Values in seconds (expected per BIDS) - should be returned as-is
    t2s_sec = np.array([0.015, 0.025, 0.035, 0.045])
    result = utils.check_t2s_values(t2s_sec)
    np.testing.assert_array_equal(result, [0.015, 0.025, 0.035, 0.045])

    # Values in milliseconds (common mistake) - should be converted to seconds with warning
    t2s_ms = np.array([15.0, 25.0, 35.0, 45.0])
    result = utils.check_t2s_values(t2s_ms)
    np.testing.assert_allclose(result, [0.015, 0.025, 0.035, 0.045])
    assert "milliseconds rather than seconds" in caplog.text

    # Array with zeros (common in T2* maps for masked voxels) - values in seconds
    t2s_with_zeros = np.array([0, 0.020, 0.030, 0, 0.040])
    result = utils.check_t2s_values(t2s_with_zeros)
    np.testing.assert_array_equal(result, [0, 0.020, 0.030, 0, 0.040])

    # All zeros - should return as-is with warning
    t2s_all_zeros = np.array([0, 0, 0, 0])
    result = utils.check_t2s_values(t2s_all_zeros)
    np.testing.assert_array_equal(result, [0, 0, 0, 0])

    # Values that are too large - should raise ValueError
    t2s_invalid = np.array([1500, 2500, 3500])
    with pytest.raises(ValueError):
        utils.check_t2s_values(t2s_invalid)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tedana/tests/test_utils.py::test_check_t2s_values -x -q 2>&1 | tail -15
```

Expected: FAIL — `assert_array_equal(result, [0.015, ...])` fails because function still returns `[15, ...]`.

---

### Task 4: Flip `check_t2s_values()` to output seconds

**Files:**
- Modify: `tedana/utils.py:562-647`

- [ ] **Step 1: Update the docstring**

Replace the `check_t2s_values` docstring (lines 563–597) with:

```python
def check_t2s_values(t2s_map):
    """Check and convert T2* map values to seconds.

    This function checks if a precalculated T2* map is in seconds (expected
    per BIDS convention) or milliseconds. Typical brain T2* values at 3T are
    approximately 0.015-0.070 seconds (15-70 ms).

    Parameters
    ----------
    t2s_map : array_like
        T2* map values to check. Expected to be in seconds per BIDS convention.

    Returns
    -------
    array_like
        T2* map values in seconds for internal use.

    Raises
    ------
    ValueError
        If T2* values appear to be in unexpected units.

    Notes
    -----
    The heuristic used is:

    - If median non-zero T2* < 1: values are assumed to be in seconds (correct
      per BIDS), returned as-is
    - If median non-zero T2* >= 1 and < 1000: values are assumed to be in
      milliseconds, a deprecation warning is logged, and values are converted
      to seconds
    - If median non-zero T2* >= 1000: values are considered invalid

    This function is designed to handle the common case where users provide
    T2* maps in milliseconds rather than seconds, which can cause severely
    biased optimal combination weighting.
    """
```

- [ ] **Step 2: Update the branching logic**

Replace the three `if/elif/else` branches (lines 625–647) with:

```python
    if median_t2s < 1:
        # Values appear to be in seconds (expected per BIDS) - return as-is
        LGR.debug(
            f"T2* map values appear to be in seconds (median={median_t2s:.4f}s)."
        )
        return t2s_map
    elif median_t2s < 1000:
        # Values appear to be in milliseconds - convert to seconds
        LGR.warning(
            f"T2* map median value is {median_t2s:.2f}, which suggests values are in "
            "milliseconds rather than seconds. Per BIDS convention, T2* maps should be "
            "in seconds. Converting to seconds. Support for millisecond T2* values is "
            "deprecated and will be removed in a future version."
        )
        return t2s_map / 1000.0
    else:
        # Values are unexpectedly large
        raise ValueError(
            f"T2* map median value is {median_t2s:.2f}, which is outside the expected range. "
            "T2* maps should be in seconds (typical values: 0.01-0.1s per BIDS convention) "
            "or milliseconds (typical values: 10-100ms). Please check your T2* map units."
        )
```

- [ ] **Step 3: Run test to verify it passes**

```bash
uv run python -m pytest tedana/tests/test_utils.py::test_check_t2s_values -x -q 2>&1 | tail -10
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tedana/utils.py tedana/tests/test_utils.py
git commit -m "refactor: flip check_t2s_values() to output seconds instead of milliseconds"
```

---

### Task 5: Delete `sec2millisec()`, `millisec2sec()`, and their call sites

**Files:**
- Modify: `tedana/utils.py:528-559`
- Modify: `tedana/workflows/tedana.py:817,822`
- Modify: `tedana/workflows/t2smap.py:518,531`
- Modify: `tedana/tests/test_utils.py:499-508`
- Modify: `docs/api.rst:315-316`

- [ ] **Step 1: Delete the test functions for the removed utilities**

In `tedana/tests/test_utils.py`, delete the following two functions entirely (lines 499–508):

```python
def test_sec2millisec():
    """Ensure that sec2millisec returns 1000x the input values."""
    assert utils.sec2millisec(5) == 5000
    assert utils.sec2millisec(np.array([5])) == np.array([5000])


def test_millisec2sec():
    """Ensure that millisec2sec returns 1/1000x the input values."""
    assert utils.millisec2sec(5000) == 5
    assert utils.millisec2sec(np.array([5000])) == np.array([5])
```

- [ ] **Step 2: Remove call sites in workflows/tedana.py**

In `tedana/workflows/tedana.py`, change line 817:

```python
# Before:
io_generator.save_file(utils.millisec2sec(t2s_full), "t2star img")
# After:
io_generator.save_file(t2s_full, "t2star img")
```

And change line 822:

```python
# Before:
io_generator.save_file(utils.millisec2sec(t2s_limited), "limited t2star img")
# After:
io_generator.save_file(t2s_limited, "limited t2star img")
```

- [ ] **Step 3: Remove call sites in workflows/t2smap.py**

In `tedana/workflows/t2smap.py`, change line 518:

```python
# Before:
io_generator.save_file(utils.millisec2sec(t2s_limited), "limited t2star img")
# After:
io_generator.save_file(t2s_limited, "limited t2star img")
```

And change line 531:

```python
# Before:
io_generator.save_file(utils.millisec2sec(t2s_full), "t2star img")
# After:
io_generator.save_file(t2s_full, "t2star img")
```

- [ ] **Step 4: Delete the utility functions from utils.py**

In `tedana/utils.py`, delete the following two functions entirely (lines 528–559):

```python
def sec2millisec(arr):
    """..."""
    return arr * 1000


def millisec2sec(arr):
    """..."""
    return arr / 1000.0
```

- [ ] **Step 5: Remove the functions from docs/api.rst**

In `docs/api.rst`, delete the two lines (around line 315–316):

```
   sec2millisec
   millisec2sec
```

- [ ] **Step 6: Run the full test suite for affected files**

```bash
uv run python -m pytest tedana/tests/test_utils.py tedana/tests/test_decay.py tedana/tests/test_combine.py -x -q 2>&1 | tail -15
```

Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add tedana/utils.py tedana/workflows/tedana.py tedana/workflows/t2smap.py tedana/tests/test_utils.py docs/api.rst
git commit -m "refactor: delete sec2millisec/millisec2sec and remove output-boundary conversions"
```

---

### Task 6: Update docstrings in decay.py, combine.py, and metrics/dependence.py

**Files:**
- Modify: `tedana/decay.py` (5 locations)
- Modify: `tedana/combine.py` (2 locations)
- Modify: `tedana/metrics/dependence.py` (1 location)

These are docstring-only changes. No tests needed — the math is scale-invariant.

- [ ] **Step 1: Update decay.py**

In `tedana/decay.py`, change every occurrence of:
```
Echo times in milliseconds.
```
to:
```
Echo times in seconds.
```

The five locations are:
- Line 28 (`_apply_t2s_floor` parameter `echo_times`)
- Line 131 (`fit_monoexponential` parameter `echo_times`)
- Line 300 (`fit_loglinear` parameter `echo_times`)
- Line 395 (`fit_decay` parameter `tes`)
- Line 583 (`fit_decay_ts` parameter `tes`)

- [ ] **Step 2: Update combine.py**

In `tedana/combine.py`, change:
- Line 23 (`_combine_t2s` parameter `tes`): `Echo times in milliseconds.` → `Echo times in seconds.`
- Line 74 (`_combine_paid` parameter `tes`): `Echo times in milliseconds.` → `Echo times in seconds.`

(Line 118 in `make_optcom` already reads "Array of TEs, in seconds." — no change needed.)

- [ ] **Step 3: Update metrics/dependence.py**

In `tedana/metrics/dependence.py`, change line 159:
```
Echo times in milliseconds, in the same order as the echoes in data_cat.
```
to:
```
Echo times in seconds, in the same order as the echoes in data_cat.
```

- [ ] **Step 4: Commit**

```bash
git add tedana/decay.py tedana/combine.py tedana/metrics/dependence.py
git commit -m "docs: update docstrings from milliseconds to seconds for TE and T2* parameters"
```

---

### Task 7: Update test fixtures to second-scale values

**Files:**
- Modify: `tedana/tests/test_decay.py:17`
- Modify: `tedana/tests/test_combine.py:14,32,49,74`

These fixtures currently use millisecond-scale values. The math is scale-invariant so tests still pass, but the values are physically wrong for seconds.

- [ ] **Step 1: Update test_decay.py fixture**

In `tedana/tests/test_decay.py`, change line 17:

```python
# Before:
tes = np.array([14.5, 38.5, 62.5])
# After:
tes = np.array([0.0145, 0.0385, 0.0625])
```

- [ ] **Step 2: Update test_combine.py fixtures**

In `tedana/tests/test_combine.py`:

Line 14 (in `test__combine_t2s`):
```python
# Before:
tes = np.array([[10, 20, 30]])  # 1 x E
# After:
tes = np.array([[0.010, 0.020, 0.030]])  # 1 x E
```

Line 32 (in `test__combine_paid`):
```python
# Before:
tes = np.array([[10, 20, 30]])  # 1 x E
# After:
tes = np.array([[0.010, 0.020, 0.030]])  # 1 x E
```

Line 49 (in `test_make_optcom`):
```python
# Before:
tes = np.array([10, 20, 30])  # E
# After:
tes = np.array([0.010, 0.020, 0.030])  # E
```

Line 74 (in `test_make_optcom`, the bad_tes check):
```python
# Before:
bad_tes = np.array([10, 20])
# After:
bad_tes = np.array([0.010, 0.020])
```

- [ ] **Step 3: Run tests to confirm they still pass**

```bash
uv run python -m pytest tedana/tests/test_decay.py tedana/tests/test_combine.py -x -q 2>&1 | tail -10
```

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tedana/tests/test_decay.py tedana/tests/test_combine.py
git commit -m "test: update TE fixtures from millisecond-scale to second-scale values"
```

---

## Final Verification

After all tasks are complete, run the full test suite for all affected modules:

```bash
uv run python -m pytest tedana/tests/test_utils.py tedana/tests/test_decay.py tedana/tests/test_combine.py -q 2>&1 | tail -15
```

Expected: All pass, no references to `sec2millisec` or `millisec2sec` remain.
