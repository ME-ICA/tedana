# Design: Interpolate Failing Voxels for Curvefit T2*/S0 Maps

**Date**: 2026-05-15
**Branch**: `interpolate-failing-voxels`

## Summary

When `fittype="curvefit"` is used, some voxels fail the monoexponential fit and currently fall back to the log-linear estimate. This design adds an opt-in option (`--interpolate-failing-voxels`) that replaces T2* and S0 values for those failing voxels with nearest-neighbor interpolated values from neighboring, non-failing voxels instead.

---

## 1. New Utility Function — `utils.interpolate_masked_values`

**File**: `tedana/utils.py`

### Signature

```python
def interpolate_masked_values(data, failures, img, mask):
```

### Parameters

- `data`: `(M,)` or `(M, T)` ndarray — T2* or S0 values for masked voxels (in denoising-mask space, where M = `mask.sum()`)
- `failures`: `(M,)` or `(M, T)` bool ndarray — True where the curvefit failed
- `img`: nibabel NIfTI image — provides affine and 3D shape to reconstruct physical coordinates
- `mask`: `(S,)` bool ndarray — the denoising mask (`mask_denoise`) mapping M voxels back to the full brain-mask space of S voxels

### Returns

`(M,)` or `(M, T)` ndarray — data with failing voxels replaced by nearest-neighbor values.

### Logic

1. Reconstruct 3D voxel indices from the mask:
   ```python
   coords = np.argwhere(mask.reshape(img.shape[:3]))  # (M, 3)
   ```
2. Convert to physical (mm) space using the image affine:
   ```python
   phys_coords = nib.affines.apply_affine(img.affine, coords)  # (M, 3)
   ```
3. **Static case** (`failures.ndim == 1`):
   - If no failures: return `data` unchanged.
   - If all voxels fail: log a warning and return `data` unchanged.
   - Otherwise: build `scipy.spatial.KDTree(phys_coords[~failures])`, query `phys_coords[failures]` for the index of the nearest non-failing neighbor, and assign `data[failures] = data[~failures][nn_indices]`.
4. **Time-series case** (`failures.ndim == 2`, shape `(M, T)`):
   - Compute `phys_coords` once.
   - For each timepoint `t`, apply the static-case logic to `data[:, t]` and `failures[:, t]`.
5. Return the corrected array.

### Dependencies

- `scipy.spatial.KDTree` — scipy is already a required dependency.
- `nibabel.affines.apply_affine` — nibabel is already a required dependency.

---

## 2. CLI and Workflow Parameter

### New parameter

| Layer | Name |
|---|---|
| CLI flag | `--interpolate-failing-voxels` |
| Python argument | `interpolate_failing_voxels` (default: `False`) |

### Changes to `tedana.py`

Add to the argument parser:
```
--interpolate-failing-voxels
    action="store_true"
    help="If fittype='curvefit', replace T2*/S0 values for voxels where the
         monoexponential fit failed with nearest-neighbor interpolated values
         from non-failing neighbors. Ignored if fittype='loglin'."
```

Add `interpolate_failing_voxels=False` to `tedana_workflow` signature and docstring.

### Changes to `t2smap.py`

Identical additions: CLI flag and `t2smap_workflow` parameter.

### Behavior when `fittype='loglin'`

Emit `LGR.warning("interpolate_failing_voxels is ignored when fittype='loglin'.")` and proceed normally. No error raised.

---

## 3. Workflow Integration

### `tedana_workflow` (`tedana/workflows/tedana.py`)

After `decay.fit_decay(...)` and before `decay.modify_t2s_s0_maps(...)`:

```python
if fittype == "curvefit":
    io_generator.save_file(failures.astype(np.uint8), "fit failures img", mask=mask_denoise)
    if verbose:
        ...  # existing variance/covariance saves

    if interpolate_failing_voxels:
        if failures.any():
            t2s_full = utils.interpolate_masked_values(t2s_full, failures, ref_img, mask_denoise)
            s0_full = utils.interpolate_masked_values(s0_full, failures, ref_img, mask_denoise)
        else:
            LGR.info("No curvefit failures found; skipping interpolation.")

del failures, t2s_var, s0_var, t2s_s0_covar
# modify_t2s_s0_maps continues as before
```

`ref_img` and `mask_denoise` are both already in scope at this point in the workflow.

### `t2smap_workflow` (`tedana/workflows/t2smap.py`)

Same pattern after `decay_function(...)`. When `fitmode="ts"`, `failures` is `(Md, T)` and `interpolate_masked_values` handles it via the time-series branch automatically — no special-casing required in the workflow.

### No changes to decay module

`fit_decay`, `fit_decay_ts`, and `fit_monoexponential` are unchanged. Interpolation is purely a post-processing step applied in the workflow.

---

## 4. Testing

### New tests in `tedana/tests/test_utils.py`

1. **Static, some failures**: Synthetic `(M,)` data + failures mask with known geometry; assert failing voxels receive the value of their geometrically nearest non-failing neighbor.
2. **Static, no failures**: Assert function returns data unchanged.
3. **Static, all failures**: Assert a warning is logged and data is returned unchanged.
4. **Time-series**: `(M, T)` data and `(M, T)` failures with different patterns per timepoint; assert each timepoint is corrected independently.
5. **Shape preservation**: Assert output shape always matches input shape for both 1D and 2D inputs.

### Workflow smoke test

Extend the existing integration test suite with a single test that runs `tedana_workflow` (or a minimal mock) with `fittype="curvefit"` and `interpolate_failing_voxels=True` and asserts it completes without error.

### Unchanged

`tedana/tests/test_decay.py` needs no changes — interpolation is not inside `fit_decay`.

---

## Out of Scope

- Distance-weighted or kernel-based interpolation methods (nearest-neighbor only for now).
- Exposing interpolation as a new `fittype` value.
- Always-on interpolation (opt-in via flag only).
