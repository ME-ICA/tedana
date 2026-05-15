# Design: Use Seconds for Internal TE and T2* Processing

**Date:** 2026-05-15
**Branch:** internal-seconds
**Status:** Approved

## Motivation

tedana currently converts echo times (TEs) and T2* maps from seconds (BIDS standard) to milliseconds at the input boundary, processes everything internally in milliseconds, and converts back to seconds for BIDS-compliant output. This creates:

- **Unit confusion**: docstrings, comments, and variable names inconsistently describe units (e.g., `combine.py` line 118 claims TEs are "in seconds" while the function actually receives milliseconds)
- **Unnecessary conversion code**: `sec2millisec()` and `millisec2sec()` utility functions exist solely to undo the input conversion at output time
- **BIDS/NIfTI inconsistency**: BIDS specifies seconds; the internal representation requires two boundary conversions
- **Literature misalignment**: MRI decay equations are conventionally written with values in seconds

The goal is to eliminate the convert-in/convert-out dance by using seconds throughout.

## Scope

This change covers **echo times (TEs) and T2* maps** — the two quantities in the decay equation `exp(-TE/T2*)`. Both must be in the same unit; converting one without the other is not valid.

TR (repetition time) is already stored in seconds via the nibabel header and is unaffected. S0 (proton density) has no physical unit constraint here and is unaffected.

## Design

### Section 1: Input Boundary

**Files:** `tedana/utils.py` — `check_te_values()` and `check_t2s_values()`

Both functions flip their output unit from milliseconds to seconds:

- **Values < 1 (seconds, BIDS standard):** accepted as-is, returned unchanged, no warning.
- **Values ≥ 1 (milliseconds, deprecated):** divided by 1000, returned as seconds, with a `DeprecationWarning` instructing callers to pass values in seconds.

The detection heuristic is unchanged — only the output unit changes. Callers in `workflows/tedana.py` and `workflows/t2smap.py` pass user-provided values through these functions and require no changes.

### Section 2: Utility Functions

**Files:** `tedana/utils.py`, `tedana/workflows/tedana.py`, `tedana/workflows/t2smap.py`

`sec2millisec()` and `millisec2sec()` exist solely to convert internal millisecond T2* values to seconds at output. With internal values now already in seconds, these conversions are no-ops.

- `sec2millisec()` and `millisec2sec()` are **deleted** from `utils.py`.
- Their call sites in `workflows/tedana.py` (lines 817, 822) and `workflows/t2smap.py` (lines 518, 531) are **removed**; the T2* arrays are passed through directly.

### Section 3: Core Calculations

**Files:** `tedana/decay.py`, `tedana/combine.py`, `tedana/metrics/dependence.py`

No numerical changes are required. The decay equation `exp(-TE/T2*)` is scale-invariant: it produces identical results whether both values are in milliseconds or both are in seconds. After the boundary change, both quantities arrive in seconds, so correctness is preserved.

Changes are **docstrings only**: every parameter or variable described as "in milliseconds" is updated to "in seconds". This also resolves the existing mismatch in `combine.py` where the docstring already said "in seconds" but the code was receiving milliseconds.

### Section 4: Tests

**Files:** `tedana/tests/test_decay.py`, `tedana/tests/test_combine.py`, and any other test files constructing TE or T2* fixtures.

- All test fixture arrays using millisecond-scale values are updated to second-scale values:
  - `np.array([14.5, 38.5, 62.5])` → `np.array([0.0145, 0.0385, 0.0625])`
  - `np.array([10, 20, 30])` → `np.array([0.010, 0.020, 0.030])`
- Assertions on T2* output values that expect millisecond-scale numbers (15–70 range) are updated to second-scale (0.015–0.070).
- Tests for `check_te_values()` and `check_t2s_values()` — specifically the deprecated millisecond input path — are updated to:
  - Assert returned values are in seconds (divided by 1000)
  - Assert a `DeprecationWarning` is raised

## Change Inventory

| File | Change type |
|------|-------------|
| `tedana/utils.py` | Flip output of `check_te_values()` and `check_t2s_values()`; delete `sec2millisec()` and `millisec2sec()` |
| `tedana/workflows/tedana.py` | Remove `millisec2sec()` call sites at T2* output |
| `tedana/workflows/t2smap.py` | Remove `millisec2sec()` call sites at T2* output |
| `tedana/decay.py` | Docstring updates only |
| `tedana/combine.py` | Docstring updates only (fixes existing mismatch) |
| `tedana/metrics/dependence.py` | Docstring updates only |
| `tedana/tests/test_decay.py` | Update fixtures and assertions to seconds |
| `tedana/tests/test_combine.py` | Update fixtures and assertions to seconds |
| Other test files | Update fixtures and assertions as needed |

## Non-Goals

- No changes to TR handling (already in seconds)
- No changes to S0 handling
- No new dependencies
- No user-visible API changes (input still accepts both ms and s via auto-detection)
- No typed-unit wrappers or physical-quantity libraries
