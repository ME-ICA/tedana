"""Expand mixing matrices fit on retained volumes to the full, censored time series."""

import logging

import numpy as np

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def project_censored_mixing(data, mixing_retained, sample_mask, cond_warning=1e6):
    """Estimate component time series at censored volumes by spatial projection.

    tedana uses spatial ICA: voxels are the samples and volumes are the features, so the
    ICA cannot be applied to a volume it was not fit on. Instead, each censored volume
    is regressed onto the component maps estimated from the retained volumes, which is
    analogous to the second stage of dual regression. The resulting amplitudes are
    valid under that fixed spatial model; they are not values the ICA itself produced.

    Parameters
    ----------
    data : (S x T) array_like
        Optimally combined data, within the spatial mask used for ICA, for every
        (post-dummy) volume.
    mixing_retained : (R x C) array_like
        Mixing matrix fit to the retained volumes, where ``R = sample_mask.sum()``.
    sample_mask : (T,) array_like of bool
        Temporal censoring mask (True = retained).
    cond_warning : float, optional
        Warn if the condition number of the component maps exceeds this value.
        Default is 1e6.

    Returns
    -------
    mixing_full : (T x C) :obj:`numpy.ndarray`
        Full-length mixing matrix. Retained rows are exactly ``mixing_retained``;
        censored rows are the projected amplitudes.
    diagnostics : :obj:`dict`
        Conditioning of the projection, and per-component agreement between
        ``mixing_retained`` and the projection of the retained volumes
        (``RetainedProjectionCorrelation``, ``RetainedProjectionScaleRatio``).
        The two are not expected to be identical, because the maps are estimated by
        regression and the component maps need not be orthogonal.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, if there are too few retained volumes for the
        number of components, or if the component maps are rank-deficient (so the
        censored amplitudes are not identifiable).
    """
    sample_mask = np.asarray(sample_mask, dtype=bool)
    data = np.asarray(data, dtype=float)
    mixing_retained = np.asarray(mixing_retained, dtype=float)
    n_vols = sample_mask.size
    n_retained = int(sample_mask.sum())
    n_components = mixing_retained.shape[1]

    if data.ndim != 2 or data.shape[1] != n_vols:
        raise ValueError(f"data must be (samples x {n_vols} volumes), not shape {data.shape}.")

    if mixing_retained.shape[0] != n_retained:
        raise ValueError(
            f"mixing_retained has {mixing_retained.shape[0]} rows, but the sample mask "
            f"retains {n_retained} volumes."
        )

    if n_retained <= n_components:
        raise ValueError(
            f"Only {n_retained} volumes are retained, which is not enough to estimate "
            f"maps for {n_components} components."
        )

    # Center every volume using the retained-volume mean of each voxel
    voxel_means = data[:, sample_mask].mean(axis=1, keepdims=True)
    data_centered = data - voxel_means

    # Spatial maps from the retained volumes (same regression as the component maps)
    maps = np.linalg.lstsq(mixing_retained, data_centered[:, sample_mask].T, rcond=None)[0].T

    singular_values = np.linalg.svd(maps, compute_uv=False)
    tol = max(maps.shape) * np.finfo(float).eps * singular_values[0]
    rank = int((singular_values > tol).sum())
    cond = float(singular_values[0] / singular_values[-1]) if rank == n_components else np.inf
    if rank < n_components:
        raise ValueError(
            f"The component maps are rank-deficient (rank {rank} < {n_components} "
            "components), so component amplitudes at censored volumes cannot be "
            "identified."
        )

    if cond > cond_warning:
        LGR.warning(
            f"The component maps are poorly conditioned (condition number {cond:.3g}), "
            "so projected amplitudes at censored volumes may be unstable."
        )

    mixing_full = np.empty((n_vols, n_components))
    mixing_full[sample_mask] = mixing_retained
    if n_retained < n_vols:
        mixing_full[~sample_mask] = np.linalg.lstsq(
            maps,
            data_centered[:, ~sample_mask],
            rcond=None,
        )[0].T

    # Diagnostics: how close is the spatial projection to the ICA on retained volumes?
    projected_retained = np.linalg.lstsq(maps, data_centered[:, sample_mask], rcond=None)[0].T
    correlations, scale_ratios = [], []
    for i_comp in range(n_components):
        ica_ts, proj_ts = mixing_retained[:, i_comp], projected_retained[:, i_comp]
        ica_std, proj_std = ica_ts.std(), proj_ts.std()
        if ica_std == 0 or proj_std == 0:
            correlations.append(np.nan)
            scale_ratios.append(np.nan)
            continue

        correlations.append(float(np.corrcoef(ica_ts, proj_ts)[0, 1]))
        scale_ratios.append(float(proj_std / ica_std))

    diagnostics = {
        "ProjectionMethod": (
            "Censored volumes were regressed onto the component maps estimated from the "
            "retained volumes (analogous to the second stage of dual regression). "
            "Retained rows of the mixing matrix are the fitted ICA values; censored rows "
            "are projected estimates."
        ),
        "ProjectionMapsConditionNumber": cond,
        "ProjectionMapsRank": rank,
        "RetainedProjectionCorrelation": correlations,
        "RetainedProjectionScaleRatio": scale_ratios,
    }
    LGR.info(
        "Projected component amplitudes for "
        f"{n_vols - n_retained} censored volumes (map condition number {cond:.3g}; "
        f"median correlation between ICA and projected retained time series "
        f"{np.nanmedian(correlations):.3f})."
    )
    return mixing_full, diagnostics
