# Interpolate Failing Voxels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--interpolate-failing-voxels` option that replaces T2*/S0 values for voxels whose curvefit failed with nearest-neighbor interpolated values from non-failing neighbors.

**Architecture:** A new `interpolate_masked_values` utility function in `utils.py` handles nearest-neighbor interpolation using `scipy.spatial.KDTree` in physical (mm) space. Both `tedana_workflow` and `t2smap_workflow` gain a new `interpolate_failing_voxels=False` parameter and call this function after `fit_decay`/`fit_decay_ts`, before `modify_t2s_s0_maps`. No changes to the decay module itself.

**Tech Stack:** numpy, scipy.spatial.KDTree, nibabel.affines.apply_affine (all existing dependencies).

---

## File Map

| File | Action | Change |
|---|---|---|
| `tedana/utils.py` | Modify | Add `_nn_replace` (private helper) + `interpolate_masked_values` (public); add 2 imports |
| `tedana/workflows/tedana.py` | Modify | CLI flag, workflow param + docstring, loglin warning, interpolation call |
| `tedana/workflows/t2smap.py` | Modify | CLI flag, workflow param + docstring, loglin warning, interpolation call |
| `tedana/tests/test_utils.py` | Modify | 5 new unit tests for `interpolate_masked_values` |
| `tedana/tests/test_t2smap.py` | Modify | 1 smoke test for t2smap workflow with interpolation |

---

## Task 1: Add `interpolate_masked_values` to `tedana/utils.py`

**Files:**
- Modify: `tedana/utils.py` (imports block at top, append new functions near bottom)
- Test: `tedana/tests/test_utils.py`

### Background

`interpolate_masked_values` accepts:
- `data`: `(M,)` or `(M, T)` ndarray of T2* or S0 values in denoising-mask space (M = `mask.sum()`)
- `failures`: `(M,)` or `(M, T)` bool ndarray — True where curvefit failed
- `img`: nibabel NIfTI brain-mask image — its `get_fdata()` gives the 3D brain mask; its `affine` converts voxel indices to mm
- `mask`: `(S,)` bool ndarray — `mask_denoise`, selecting M of S brain voxels

It reconstructs 3D physical coordinates for each masked voxel, then uses `scipy.spatial.KDTree` to find the nearest non-failing neighbor for each failing voxel. The private helper `_nn_replace` does the in-place replacement for a single 1D slice.

- [ ] **Step 1: Write 5 failing tests in `tedana/tests/test_utils.py`**

Add these tests at the bottom of the file (after the last test function):

```python
import nibabel as nib  # add to the imports block at the top of test_utils.py


def test_interpolate_masked_values_some_failures():
    # 5 voxels in a line (5x1x1). Affine is identity so physical == voxel coords.
    img = nib.Nifti1Image(np.ones((5, 1, 1), dtype=np.uint8), np.eye(4))
    mask = np.ones(5, dtype=bool)  # all 5 brain voxels in denoising mask

    data = np.array([99.0, 2.0, 3.0, 4.0, 99.0])
    failures = np.array([True, False, False, False, True])

    result = utils.interpolate_masked_values(data, failures, img, mask)

    # Voxel 0 at x=0: nearest non-failing voxel is voxel 1 at x=1 (val=2.0)
    assert result[0] == 2.0
    # Voxel 4 at x=4: nearest non-failing voxel is voxel 3 at x=3 (val=4.0)
    assert result[4] == 4.0
    # Non-failing voxels are unchanged
    np.testing.assert_array_equal(result[1:4], data[1:4])


def test_interpolate_masked_values_no_failures():
    img = nib.Nifti1Image(np.ones((3, 1, 1), dtype=np.uint8), np.eye(4))
    mask = np.ones(3, dtype=bool)
    data = np.array([1.0, 2.0, 3.0])
    failures = np.zeros(3, dtype=bool)

    result = utils.interpolate_masked_values(data, failures, img, mask)

    np.testing.assert_array_equal(result, data)


def test_interpolate_masked_values_all_failures(caplog):
    import logging

    img = nib.Nifti1Image(np.ones((3, 1, 1), dtype=np.uint8), np.eye(4))
    mask = np.ones(3, dtype=bool)
    data = np.array([1.0, 2.0, 3.0])
    failures = np.ones(3, dtype=bool)

    with caplog.at_level(logging.WARNING, logger="tedana.utils"):
        result = utils.interpolate_masked_values(data, failures, img, mask)

    assert "cannot interpolate" in caplog.text.lower()
    np.testing.assert_array_equal(result, data)


def test_interpolate_masked_values_timeseries():
    # 5 voxels, 2 timepoints; different failing voxel per timepoint
    img = nib.Nifti1Image(np.ones((5, 1, 1), dtype=np.uint8), np.eye(4))
    mask = np.ones(5, dtype=bool)

    data = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0])[:, np.newaxis], (1, 2))
    failures = np.zeros((5, 2), dtype=bool)
    failures[0, 0] = True   # t=0: voxel 0 fails
    failures[4, 1] = True   # t=1: voxel 4 fails

    result = utils.interpolate_masked_values(data, failures, img, mask)

    # t=0: voxel 0 gets value of nearest non-failing (voxel 1, val=2.0)
    assert result[0, 0] == 2.0
    # t=1: voxel 4 gets value of nearest non-failing (voxel 3, val=4.0)
    assert result[4, 1] == 4.0
    # Non-failing entries are unchanged
    assert result[4, 0] == 5.0
    assert result[0, 1] == 1.0


def test_interpolate_masked_values_shape_preservation():
    img = nib.Nifti1Image(np.ones((3, 1, 1), dtype=np.uint8), np.eye(4))
    mask = np.ones(3, dtype=bool)

    data_1d = np.array([1.0, 2.0, 3.0])
    failures_1d = np.array([True, False, False])
    result_1d = utils.interpolate_masked_values(data_1d, failures_1d, img, mask)
    assert result_1d.shape == data_1d.shape

    data_2d = np.ones((3, 5))
    failures_2d = np.zeros((3, 5), dtype=bool)
    failures_2d[0, :] = True
    result_2d = utils.interpolate_masked_values(data_2d, failures_2d, img, mask)
    assert result_2d.shape == data_2d.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/c/Users/tsalo/Documents/tsalo/tedana
python -m pytest tedana/tests/test_utils.py::test_interpolate_masked_values_some_failures \
    tedana/tests/test_utils.py::test_interpolate_masked_values_no_failures \
    tedana/tests/test_utils.py::test_interpolate_masked_values_all_failures \
    tedana/tests/test_utils.py::test_interpolate_masked_values_timeseries \
    tedana/tests/test_utils.py::test_interpolate_masked_values_shape_preservation \
    -v 2>&1 | tail -20
```

Expected: 5 FAILED (AttributeError: module 'tedana.utils' has no attribute 'interpolate_masked_values')

- [ ] **Step 3: Add imports to `tedana/utils.py`**

In `tedana/utils.py`, add two imports after the existing `from scipy import ndimage` line (around line 20):

```python
from nibabel.affines import apply_affine
from scipy.spatial import KDTree
```

- [ ] **Step 4: Add `_nn_replace` and `interpolate_masked_values` to `tedana/utils.py`**

Append both functions at the end of `tedana/utils.py` (after the last function in the file):

```python
def _nn_replace(data, failures, phys_coords):
    """Replace failing entries with nearest non-failing neighbor values (in-place).

    Parameters
    ----------
    data : (M,) :obj:`numpy.ndarray`
        Values to potentially replace, modified in-place.
    failures : (M,) :obj:`numpy.ndarray` of bool
        True where values should be replaced.
    phys_coords : (M, 3) :obj:`numpy.ndarray`
        Physical coordinates in mm for each voxel.
    """
    if not failures.any():
        return
    good = ~failures
    if not good.any():
        LGR.warning(
            "All voxels failed the monoexponential fit; cannot interpolate missing values."
        )
        return
    tree = KDTree(phys_coords[good])
    _, idx = tree.query(phys_coords[failures])
    data[failures] = data[good][idx]


def interpolate_masked_values(data, failures, img, mask):
    """Replace failing voxels with nearest-neighbor values from non-failing voxels.

    Parameters
    ----------
    data : (M,) or (M, T) :obj:`numpy.ndarray`
        T2* or S0 values for masked voxels, where M is the number of True
        values in ``mask``.
    failures : (M,) or (M, T) :obj:`numpy.ndarray` of bool
        True where the curvefit failed for the corresponding voxel
        (and timepoint for 2D input).
    img : :obj:`nibabel.nifti1.Nifti1Image`
        Brain mask image. Its data determines which voxels are in the brain
        mask and its affine converts voxel indices to physical (mm) coordinates.
    mask : (S,) :obj:`numpy.ndarray` of bool
        Denoising mask (``mask_denoise``), where S is the number of brain
        voxels in ``img`` and M is the number of True values in this array.

    Returns
    -------
    result : (M,) or (M, T) :obj:`numpy.ndarray`
        Copy of ``data`` with failing voxels replaced by the value of the
        nearest non-failing voxel in physical space.
    """
    brain_mask = img.get_fdata().astype(bool)
    brain_coords = np.argwhere(brain_mask)  # (S, 3) voxel indices
    voxel_coords = brain_coords[mask]  # (M, 3)
    phys_coords = apply_affine(img.affine, voxel_coords)  # (M, 3) in mm

    result = data.copy()
    if failures.ndim == 1:
        _nn_replace(result, failures, phys_coords)
    else:
        for t in range(failures.shape[1]):
            _nn_replace(result[:, t], failures[:, t], phys_coords)
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tedana/tests/test_utils.py::test_interpolate_masked_values_some_failures \
    tedana/tests/test_utils.py::test_interpolate_masked_values_no_failures \
    tedana/tests/test_utils.py::test_interpolate_masked_values_all_failures \
    tedana/tests/test_utils.py::test_interpolate_masked_values_timeseries \
    tedana/tests/test_utils.py::test_interpolate_masked_values_shape_preservation \
    -v 2>&1 | tail -20
```

Expected: 5 PASSED

- [ ] **Step 6: Run full test_utils.py to check for regressions**

```bash
python -m pytest tedana/tests/test_utils.py -v 2>&1 | tail -20
```

Expected: All existing tests PASSED, 5 new tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add tedana/utils.py tedana/tests/test_utils.py
git commit -m "$(cat <<'EOF'
feat: add interpolate_masked_values utility for curvefit failure recovery

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `--interpolate-failing-voxels` to `tedana_workflow`

**Files:**
- Modify: `tedana/workflows/tedana.py`

### Context

- The argument parser function `_get_parser()` has a `decay_args` group (line 182). The last `decay_args.add_argument` call ends around line 213.
- `tedana_workflow` is defined at line 423. The `fittype` parameter is at line 432 in the signature and documented around line 493 in the docstring.
- The `fit_decay` call is around line 783. The `if fittype == "curvefit":` block saving failures runs through line 801. `del failures, ...` is at line 803. `modify_t2s_s0_maps` follows at line 805.
- `mask_img` is in scope (defined at line 684). `mask_denoise` is in scope (defined at line 756).
- `_main()` at line 1250 uses `**kwargs` from `vars(options)` so the new CLI flag automatically flows to `tedana_workflow` once both the parser and function signature are updated.

- [ ] **Step 1: Add CLI flag to `_get_parser()` in `tedana/workflows/tedana.py`**

Locate the existing `--t2smap` argument (around line 202). Add the new flag immediately after the closing parenthesis of that argument block (after line 213 or wherever `decay_args.add_argument("--t2smap", ...)` ends):

```python
    decay_args.add_argument(
        "--interpolate-failing-voxels",
        dest="interpolate_failing_voxels",
        action="store_true",
        help=(
            "If fittype='curvefit', replace T2*/S0 values for voxels where the "
            "monoexponential fit failed with nearest-neighbor interpolated values "
            "from non-failing neighbors. Ignored if fittype='loglin'."
        ),
        default=False,
    )
```

- [ ] **Step 2: Add `interpolate_failing_voxels=False` to `tedana_workflow` signature**

In `tedana_workflow`'s parameter list (around line 432), add the new parameter after `fittype="loglin",`:

```python
    fittype="loglin",
    interpolate_failing_voxels=False,
```

- [ ] **Step 3: Add docstring entry for `interpolate_failing_voxels`**

In the `tedana_workflow` docstring, after the `fittype` entry (around line 497), add:

```
    interpolate_failing_voxels : :obj:`bool`, optional
        If ``True`` and ``fittype='curvefit'``, replace T2*/S0 values for
        voxels where the monoexponential fit failed with nearest-neighbor
        interpolated values from non-failing neighbors.
        Ignored if ``fittype='loglin'``. Default is ``False``.
```

- [ ] **Step 4: Add loglin warning before the T2* map section**

Locate the block `if t2smap is None:` (around line 779). Add the warning immediately before it:

```python
    if interpolate_failing_voxels and fittype != "curvefit":
        LGR.warning(
            "interpolate_failing_voxels is set but fittype is not 'curvefit'; "
            "interpolation will be skipped."
        )
```

- [ ] **Step 5: Add interpolation call inside the `if fittype == "curvefit":` block**

Locate the `if fittype == "curvefit":` block that saves failures (around line 792). After the `if verbose:` sub-block (after the last `io_generator.save_file` call inside it), and before `del failures, ...`, add:

```python
        if interpolate_failing_voxels:
            if failures.any():
                t2s_full = utils.interpolate_masked_values(
                    t2s_full, failures, mask_img, mask_denoise
                )
                s0_full = utils.interpolate_masked_values(
                    s0_full, failures, mask_img, mask_denoise
                )
            else:
                LGR.info("No curvefit failures found; skipping interpolation.")
```

The result (with surrounding context) should look like:

```python
        if fittype == "curvefit":
            io_generator.save_file(
                failures.astype(np.uint8),
                "fit failures img",
                mask=mask_denoise,
            )
            if verbose:
                io_generator.save_file(t2s_var, "t2star variance img", mask=mask_denoise)
                io_generator.save_file(s0_var, "s0 variance img", mask=mask_denoise)
                io_generator.save_file(t2s_s0_covar, "t2star-s0 covariance img", mask=mask_denoise)

            if interpolate_failing_voxels:
                if failures.any():
                    t2s_full = utils.interpolate_masked_values(
                        t2s_full, failures, mask_img, mask_denoise
                    )
                    s0_full = utils.interpolate_masked_values(
                        s0_full, failures, mask_img, mask_denoise
                    )
                else:
                    LGR.info("No curvefit failures found; skipping interpolation.")

        del failures, t2s_var, s0_var, t2s_s0_covar
```

- [ ] **Step 6: Run existing tedana tests to check for regressions**

```bash
python -m pytest tedana/tests/test_decay.py tedana/tests/test_utils.py -v 2>&1 | tail -20
```

Expected: All PASSED.

- [ ] **Step 7: Commit**

```bash
git add tedana/workflows/tedana.py
git commit -m "$(cat <<'EOF'
feat: add --interpolate-failing-voxels to tedana workflow

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `--interpolate-failing-voxels` to `t2smap_workflow`

**Files:**
- Modify: `tedana/workflows/t2smap.py`
- Test: `tedana/tests/test_t2smap.py`

### Context

- `_get_parser()` has a `decay_args` group (line 150). The last `decay_args.add_argument` call ends at line 180 (the `--combmode` argument).
- `t2smap_workflow` is defined at line 241. `fittype="loglin"` is at line 250.
- The `decay_function(...)` call is around line 462 and its `if fittype == "curvefit":` block runs through line 487. `del failures, ...` is at line 487.
- `mask_img` is in scope (defined at line 411). `mask_denoise` is in scope (defined at line 449).
- `_main()` uses the same `**kwargs` forwarding pattern as `tedana.py`.
- The existing test file `tedana/tests/test_t2smap.py` already imports nibabel as `nb` and uses `get_test_data_path()` to find `echo1.nii.gz`, `echo2.nii.gz`, `echo3.nii.gz` in `tedana/tests/data/`.

- [ ] **Step 1: Add CLI flag to `_get_parser()` in `tedana/workflows/t2smap.py`**

Locate the `--combmode` argument block in `decay_args` (around line 174). Add the new flag immediately after its closing parenthesis:

```python
    decay_args.add_argument(
        "--interpolate-failing-voxels",
        dest="interpolate_failing_voxels",
        action="store_true",
        help=(
            "If fittype='curvefit', replace T2*/S0 values for voxels where the "
            "monoexponential fit failed with nearest-neighbor interpolated values "
            "from non-failing neighbors. Ignored if fittype='loglin'."
        ),
        default=False,
    )
```

- [ ] **Step 2: Add `interpolate_failing_voxels=False` to `t2smap_workflow` signature**

In `t2smap_workflow`'s parameter list (around line 250), add after `fittype="loglin",`:

```python
    fittype="loglin",
    interpolate_failing_voxels=False,
```

- [ ] **Step 3: Add docstring entry**

In the `t2smap_workflow` docstring, after the `fittype` entry (around line 298), add:

```
    interpolate_failing_voxels : :obj:`bool`, optional
        If ``True`` and ``fittype='curvefit'``, replace T2*/S0 values for
        voxels where the monoexponential fit failed with nearest-neighbor
        interpolated values from non-failing neighbors.
        Ignored if ``fittype='loglin'``. Default is ``False``.
```

- [ ] **Step 4: Add loglin warning**

Before the `LGR.info("Computing T2* map")` line (around line 458), add:

```python
    if interpolate_failing_voxels and fittype != "curvefit":
        LGR.warning(
            "interpolate_failing_voxels is set but fittype is not 'curvefit'; "
            "interpolation will be skipped."
        )
```

- [ ] **Step 5: Add interpolation call inside the `if fittype == "curvefit":` block**

Locate the `if fittype == "curvefit":` block after the `decay_function(...)` call (around line 471). After the `if verbose:` sub-block and before `del failures, ...`, add:

```python
        if interpolate_failing_voxels:
            if failures.any():
                t2s_full = utils.interpolate_masked_values(
                    t2s_full, failures, mask_img, mask_denoise
                )
                s0_full = utils.interpolate_masked_values(
                    s0_full, failures, mask_img, mask_denoise
                )
            else:
                LGR.info("No curvefit failures found; skipping interpolation.")
```

The result should look like:

```python
    if fittype == "curvefit":
        io_generator.save_file(
            failures.astype(np.uint8),
            "fit failures img",
            mask=mask_denoise,
        )
        if verbose:
            io_generator.save_file(t2s_var, "t2star variance img", mask=mask_denoise)
            io_generator.save_file(s0_var, "s0 variance img", mask=mask_denoise)
            io_generator.save_file(
                t2s_s0_covar,
                "t2star-s0 covariance img",
                mask=mask_denoise,
            )

        if interpolate_failing_voxels:
            if failures.any():
                t2s_full = utils.interpolate_masked_values(
                    t2s_full, failures, mask_img, mask_denoise
                )
                s0_full = utils.interpolate_masked_values(
                    s0_full, failures, mask_img, mask_denoise
                )
            else:
                LGR.info("No curvefit failures found; skipping interpolation.")

    # Delete unused variables
    del failures, t2s_var, s0_var, t2s_s0_covar
```

- [ ] **Step 6: Write a smoke test in `tedana/tests/test_t2smap.py`**

Add the following test to the `TestT2smap` class (after `test_basic_t2smap2` or whichever is last):

```python
    def test_interpolate_failing_voxels_curvefit(self, tmp_path):
        """Smoke test: t2smap_workflow with interpolate_failing_voxels=True completes."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = str(tmp_path / "output")
        workflows.t2smap_workflow(
            data,
            [14.5, 38.5, 62.5],
            combmode="t2s",
            fitmode="all",
            fittype="curvefit",
            interpolate_failing_voxels=True,
            out_dir=out_dir,
        )
        assert op.isfile(op.join(out_dir, "T2starmap.nii.gz"))
        assert op.isfile(op.join(out_dir, "S0map.nii.gz"))
```

- [ ] **Step 7: Run the smoke test**

```bash
python -m pytest tedana/tests/test_t2smap.py::TestT2smap::test_interpolate_failing_voxels_curvefit -v 2>&1 | tail -20
```

Expected: PASSED

- [ ] **Step 8: Run all t2smap tests to check for regressions**

```bash
python -m pytest tedana/tests/test_t2smap.py -v 2>&1 | tail -30
```

Expected: All PASSED.

- [ ] **Step 9: Commit**

```bash
git add tedana/workflows/t2smap.py tedana/tests/test_t2smap.py
git commit -m "$(cat <<'EOF'
feat: add --interpolate-failing-voxels to t2smap workflow

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Final verification

- [ ] **Step 1: Run all unit tests**

```bash
python -m pytest tedana/tests/test_utils.py tedana/tests/test_decay.py tedana/tests/test_t2smap.py -v 2>&1 | tail -30
```

Expected: All PASSED.

- [ ] **Step 2: Verify the CLI flag appears in --help**

```bash
python -m tedana.workflows.tedana --help 2>&1 | grep -A3 "interpolate"
python -m tedana.workflows.t2smap --help 2>&1 | grep -A3 "interpolate"
```

Expected: Both show `--interpolate-failing-voxels` with its help text.

- [ ] **Step 3: Run pre-commit checks**

```bash
pre-commit run --all-files 2>&1 | tail -30
```

Expected: All checks pass (fix any auto-fixable issues pre-commit reports).

- [ ] **Step 4: Commit any pre-commit fixes**

If pre-commit auto-fixed anything:

```bash
git add -u
git commit -m "$(cat <<'EOF'
style: apply pre-commit fixes

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
