"""Spatial noise metrics.

These metrics are independent of the TE-dependence model.
They identify acquisition/motion artifacts that mimic mixed TE-dependence:

- ``compute_slice_banding`` detects multiband/slice-leakage structure in a
  component's spatial weight map.
"""

import logging

import nibabel as nb
import numpy as np
from nilearn import masking

LGR = logging.getLogger("GENERAL")


def compute_slice_banding(*, weight_maps: np.ndarray, mask_img: nb.Nifti1Image) -> np.ndarray:
    """Quantify slice-banding (multiband leakage) in component weight maps.

    For each spatial axis, ``bandR2`` is the fraction of weight variance
    explained by the per-slice mean (banding magnitude) and ``bandMB`` is the
    fraction of the linearly-detrended slice-profile power spectrum in its upper
    half (the alternating-slice / multiband-leakage signature, which separates
    leakage from smooth anatomical gradients). The metric is ``max over axes of
    bandR2 * bandMB``.

    Parameters
    ----------
    weight_maps : (M_s x C) array_like
        Component weight maps in strict-mask voxel space.
    mask_img : img_like
        Strict mask used to unmask ``weight_maps`` to the image grid.

    Returns
    -------
    (C,) :obj:`numpy.ndarray`
        Slice-banding index in ``[0, 1]`` per component.
    """
    mask_bool = np.asanyarray(mask_img.dataobj).astype(bool)
    img = masking.unmask(weight_maps.T, mask_img)
    vols = np.asanyarray(img.dataobj)  # (X, Y, Z, C)
    n_comp = vols.shape[-1]
    out = np.zeros(n_comp)
    for c in range(n_comp):
        vol = vols[..., c]
        best = 0.0
        for axis in range(3):
            other = tuple(a for a in range(3) if a != axis)
            n_per_slice = np.maximum(mask_bool.sum(axis=other), 1)
            profile = (vol * mask_bool).sum(axis=other) / n_per_slice
            vox = vol[mask_bool]
            coords = np.indices(vol.shape)[axis][mask_bool]
            pred = profile[coords]
            ss_tot = np.sum((vox - vox.mean()) ** 2)
            if ss_tot <= 0:
                continue
            band_r2 = 1.0 - np.sum((vox - pred) ** 2) / ss_tot
            centered = profile - profile.mean()
            ss_before = np.dot(centered, centered)
            t = np.arange(len(centered), dtype=float) - (len(centered) - 1) / 2.0
            denom = np.dot(t, t)
            if denom > 0:
                centered = centered - t * (np.dot(t, centered) / denom)
            ss_after = np.dot(centered, centered)
            if ss_before > 0 and ss_after > ss_before * 1e-10:
                spec = np.abs(np.fft.rfft(centered)) ** 2
                band_mb = spec[len(spec) // 2 :].sum() / spec.sum() if spec.sum() > 0 else 0.0
            else:
                band_mb = 0.0
            best = max(best, band_r2 * band_mb)
        out[c] = best
    return out


def _slice_axis_from_affine(affine: np.ndarray) -> int:
    """Through-plane (slice) axis = array axis with the largest voxel spacing.

    Warns when the largest two spacings are near-equal (isotropic voxels), since
    the slice axis is then ambiguous and ``--metadata`` should be preferred.
    """
    voxel_sizes = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
    order = np.argsort(voxel_sizes)
    axis = int(order[-1])
    largest, second = voxel_sizes[order[-1]], voxel_sizes[order[-2]]
    if largest - second <= 1e-3 * max(largest, 1.0):
        LGR.warning(
            "Slice axis is ambiguous from the affine (near-isotropic voxels); "
            "defaulting to axis %d. Provide --metadata for the true slice axis.",
            axis,
        )
    return axis


def _candidate_group_spacings(n_slices: int, mb_factor):
    """Aliasing slice spacings to test.

    With a known multiband factor that divides ``n_slices``, the only spacing is
    ``n_slices / mb_factor``. Otherwise scan integer divisors ``g`` in
    ``[2, n_slices // 2]`` (equal-sized aliasing groups).
    """
    if mb_factor is not None and mb_factor >= 2 and n_slices % mb_factor == 0:
        return [n_slices // mb_factor]
    return [g for g in range(2, n_slices // 2 + 1) if n_slices % g == 0]


def _slice_inplane_and_profile(vol, mask_bool, axis):
    """Per-slice flattened (values, mask) and the in-mask mean-energy profile."""
    n = vol.shape[axis]
    planes = []
    profile = np.zeros(n)
    for s in range(n):
        values = np.take(vol, s, axis=axis).ravel()
        mask = np.take(mask_bool, s, axis=axis).ravel()
        planes.append((values, mask))
        profile[s] = values[mask].mean() if mask.any() else 0.0
    return planes, profile


def _pairwise_slice_corr(planes):
    """(S x S) Pearson correlation between in-plane slice images over common in-mask voxels."""
    n = len(planes)
    corr = np.full((n, n), np.nan)
    for a in range(n):
        va, ma = planes[a]
        for b in range(a + 1, n):
            vb, mb_mask = planes[b]
            common = ma & mb_mask
            if common.sum() < 3:
                continue
            xa = va[common]
            xb = vb[common]
            if xa.std() <= 0 or xb.std() <= 0:
                continue
            r = float(np.corrcoef(xa, xb)[0, 1])
            corr[a, b] = corr[b, a] = r
    return corr


def _aliasing_stat(corr, order, g):
    """Mean within-aliasing-group correlation; group p = positions {p, p+g, ...}.

    ``order`` maps slice-position -> slice index (identity = observed; a
    permutation = a null draw).
    """
    n = len(order)
    group_means = []
    for p in range(g):
        slices = [order[pos] for pos in range(p, n, g)]
        vals = []
        for i in range(len(slices)):
            for j in range(i + 1, len(slices)):
                c = corr[slices[i], slices[j]]
                if not np.isnan(c):
                    vals.append(c)
        if vals:
            group_means.append(np.mean(vals))
    return float(np.mean(group_means)) if group_means else 0.0


def _periodicity_stat(profile, g):
    """Fraction of profile variance explained by period-g group means (R²).

    Computes the one-way-ANOVA R² of the energy profile partitioned into
    ``g`` equally-spaced groups of positions ``{p, p+g, p+2g, ...}``.  A
    periodic profile with period ``g`` has R² ≈ 1; a randomly ordered
    profile has R² near ``(g-1) / (n-1)`` on average.  This formulation is
    waveform-shape agnostic: it captures periodicity from both sinusoidal and
    non-sinusoidal (e.g. sawtooth energy-envelope) profiles without requiring
    power to concentrate at a single FFT bin.
    """
    x = np.asarray(profile, dtype=float)
    n = x.size
    grand_mean = x.mean()
    ss_tot = np.dot(x - grand_mean, x - grand_mean)
    if ss_tot <= 0:
        return 0.0
    ss_model = 0.0
    for p in range(g):
        group = x[p::g]
        group_mean = group.mean()
        ss_model += len(group) * (group_mean - grand_mean) ** 2
    return float(np.clip(ss_model / ss_tot, 0.0, 1.0))


def _zscore(obs, null):
    null = np.asarray(null, dtype=float)
    sd = null.std()
    if sd <= 0:
        return 0.0
    return float((obs - null.mean()) / sd)


def compute_slice_leakage(
    *,
    weight_maps: np.ndarray,
    mask_img: nb.Nifti1Image,
    slice_axis: int = None,
    mb_factor: int = None,
    n_permutations: int = 256,
    seed: int = 0,
) -> dict:
    """Metadata-aware, null-calibrated SMS slice-leakage detector.

    For each component (squared weight map), two statistics are computed and
    calibrated against a slice-index permutation null:

    - ``aliasing_z``: excess Pearson correlation among aliasing-partner slices
      (separated by ``n_slices / mb_factor``) versus randomly grouped slices.
    - ``periodicity_z``: fraction of the per-slice energy profile's variance
      explained by period-g group means (ANOVA R^2), z-scored against
      permuted-order profiles.

    ``slice_leakage = min(aliasing_z, periodicity_z)`` (both signatures
    required). When ``mb_factor`` is unknown, candidate spacings are scanned and
    the null applies the identical scan, so the z-scores stay calibrated.

    Parameters
    ----------
    weight_maps : (M_s x C) array_like
        Squared component weight maps in strict-mask voxel space.
    mask_img : img_like
        Strict mask used to unmask ``weight_maps``.
    slice_axis : int, optional
        Array axis of the slice direction. If None, inferred from the affine.
    mb_factor : int, optional
        Multiband factor. If None, candidate spacings are scanned. If 1, the
        metric is not applicable and all outputs are 0.
    n_permutations : int
        Permutation-null draws (default 256).
    seed : int
        RNG seed for reproducibility (default 0).

    Returns
    -------
    dict
        ``{"slice_leakage": (C,), "aliasing_z": (C,), "periodicity_z": (C,)}``.
    """
    n_comp = weight_maps.shape[1]
    leakage = np.zeros(n_comp)
    aliasing_z = np.zeros(n_comp)
    periodicity_z = np.zeros(n_comp)

    if mb_factor is not None and mb_factor < 2:
        # No simultaneous-multi-slice acquisition -> no leakage to detect.
        LGR.info("Multiband factor < 2; slice_leakage is not applicable (returning zeros).")
        return {
            "slice_leakage": leakage,
            "aliasing_z": aliasing_z,
            "periodicity_z": periodicity_z,
        }

    mask_bool = np.asanyarray(mask_img.dataobj).astype(bool)
    img = masking.unmask(weight_maps.T, mask_img)
    vols = np.asanyarray(img.dataobj)  # (X, Y, Z, C)

    axis = slice_axis if slice_axis is not None else _slice_axis_from_affine(mask_img.affine)
    rng = np.random.default_rng(seed)

    for c in range(n_comp):
        vol = vols[..., c].astype(np.float64)
        n_slices = vol.shape[axis]
        spacings = _candidate_group_spacings(n_slices, mb_factor)
        if not spacings or n_slices < 4:
            continue

        planes, profile = _slice_inplane_and_profile(vol, mask_bool, axis)
        corr = _pairwise_slice_corr(planes)
        identity = np.arange(n_slices)

        stat1_obs = max(_aliasing_stat(corr, identity, g) for g in spacings)
        stat2_obs = max(_periodicity_stat(profile, g) for g in spacings)

        null1 = np.empty(n_permutations)
        null2 = np.empty(n_permutations)
        for i in range(n_permutations):
            order = rng.permutation(n_slices)
            null1[i] = max(_aliasing_stat(corr, order, g) for g in spacings)
            permuted_profile = profile[order]
            null2[i] = max(_periodicity_stat(permuted_profile, g) for g in spacings)

        z1 = _zscore(stat1_obs, null1)
        z2 = _zscore(stat2_obs, null2)
        aliasing_z[c] = z1
        periodicity_z[c] = z2
        leakage[c] = min(z1, z2)

    return {
        "slice_leakage": leakage,
        "aliasing_z": aliasing_z,
        "periodicity_z": periodicity_z,
    }
