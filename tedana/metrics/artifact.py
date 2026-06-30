"""Artifact metrics: spatial (weight-map) and temporal (mixing) noise signatures.

These metrics are independent of the TE-dependence model. They identify
acquisition/motion artifacts that mimic mixed TE-dependence:

- ``compute_slice_banding`` detects multiband/slice-leakage structure in a
  component's spatial weight map.
- ``compute_spike`` detects isolated transients in a component's time series.
"""

import logging

import nibabel as nb
import numpy as np
from nilearn import masking
from scipy.signal import detrend
from scipy.stats import kurtosis

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


def compute_spike(*, mixing: np.ndarray) -> np.ndarray:
    """Temporal kurtosis of each component time series (transient-spike index).

    Each column is linearly detrended (removing drift without inflating the
    statistic for slow/task structure), then its Fisher kurtosis is taken. A
    component dominated by an isolated transient has high kurtosis; smooth or
    oscillatory signal stays low. Sign-invariant.

    Parameters
    ----------
    mixing : (T x C) array_like
        Component mixing matrix (time series per component).

    Returns
    -------
    (C,) :obj:`numpy.ndarray`
        Fisher temporal kurtosis per component.
    """
    detrended = detrend(np.asarray(mixing, dtype=np.float64), axis=0, type="linear")
    return kurtosis(detrended, axis=0, fisher=True)
