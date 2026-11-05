"""File to assess quality metrics for components."""

import numpy as np
import pandas as pd

from tedana.stats import fit_model, get_coeffs


def calculate_rejected_components_impact(selector, mixing):
    """Calculate the % variance explained by the rejected components for accepted components.

    The final metric is the weighted sum across accepted components of
    the total variance explained by each accepted component times
    the varianced explained (100*R^2) of each accepted component by the rejected comp time series.
    This quantifies the impact of rejected components on the overall variance explained
    by accepted components.

    Parameters
    ----------
    selector : :obj: tedana.selection.component_selector.ComponentSelector
        Uses `variance explained` and `classification` columns in ``selector.component_table_``.
    mixing : (T [x C]) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is time

    Returns
    -------
    None
        Updates elements the selector object:
        ``selector.component_table_`` has an added column for "Var Exp of rejected to accepted"
        ``selector.cross_component_metrics_`` has an added value
        `total_var_exp_rejected_components_on_accepted`
        `variance explained` is a percentage ranging from 0-100% and
        `total_var_exp_rejected_components_on_accepted` is also a percent with the same range.
    """
    component_table = selector.component_table_

    rej = component_table.loc[component_table["classification"] == "rejected"].index
    acc = component_table.loc[component_table["classification"] == "accepted"].index

    if len(rej) == 0 or len(acc) == 0:
        component_table["Var Exp of rejected to accepted"] = np.nan
        selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"] = np.nan
        return

    rej_arrs, acc_arrs = mixing[:, rej], mixing[:, acc]

    # Calculate the R-squared values for each accepted component
    _, sse, _ = fit_model(rej_arrs, acc_arrs)
    # Note: Since the mixing matrix time series are typically mean centered with stdev 1,
    #    ss_total will be the number of time points (+ rounding errors)
    ss_total = np.sum(
        (acc_arrs - np.tile(np.mean(acc_arrs, axis=0), [acc_arrs.shape[0], 1])) ** 2, axis=0
    )
    r2 = 1 - (sse / ss_total)

    if "Var Exp of rejected to accepted" in component_table.columns:
        # Reset to NaN before calculating for accepted components
        component_table["Var Exp of rejected to accepted"] = np.nan
    else:
        # initialize new column in component table that's left of the classification & tag columns
        num_columns = len(component_table.columns)
        component_table.insert(num_columns - 2, "Var Exp of rejected to accepted", np.nan)

    # Calculated r2 is assigned to accepted components and the rest remain NaN
    component_table.loc[acc, "Var Exp of rejected to accepted"] = 100 * r2

    # Update selector with the overall impact measure
    selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"] = (
        np.sum(
            component_table["Var Exp of rejected to accepted"][acc]
            * component_table["variance explained"][acc]
        )
        / 100
    )


def calculate_variance_summary(selector, data_optcom_masked, mixing):
    """Store aggregate variance QC scalars in ``selector.cross_component_metrics_``.

    Adds, as percentages:

    - ``accepted_variance``, ``rejected_variance``: sums of per-component
      ``"variance explained"`` grouped by classification. These are relative to the
      variance modeled by the ICA decomposition and together sum to ~100%.
    - ``unmodeled_variance`` = ``100 - total_r2``, where ``total_r2`` is the variance of
      the raw optimally-combined data explained by the full decomposition.
    - ``retained_variance``: variance of the denoised data (rejected components removed)
      relative to the raw optimally-combined data.

    Parameters
    ----------
    selector : :obj:`tedana.selection.component_selector.ComponentSelector`
    data_optcom_masked : (S x T) array_like
        Optimally-combined data restricted to the classification mask.
    mixing : (T x C) array_like
        ICA mixing matrix.
    """
    component_table = selector.component_table_

    for label in ("accepted", "rejected"):
        label_mask = component_table["classification"] == label
        selector.cross_component_metrics_[f"{label}_variance"] = float(
            component_table.loc[label_mask, "variance explained"].sum()
        )

    # Variance relative to the raw optimally-combined data (mirrors io.denoise_ts).
    dmdata = data_optcom_masked.T - data_optcom_masked.T.mean(axis=0)
    betas = get_coeffs(dmdata.T, mixing)
    sst = (dmdata.T**2).sum()
    reconstruction = betas.dot(mixing.T)
    total_r2 = (1 - ((dmdata.T - reconstruction) ** 2).sum() / sst) * 100
    selector.cross_component_metrics_["unmodeled_variance"] = float(100 - total_r2)

    rej = component_table[component_table["classification"] == "rejected"].index.values
    rejected_reconstruction = betas[:, rej].dot(mixing.T[rej, :])
    denoised = dmdata.T - rejected_reconstruction
    selector.cross_component_metrics_["retained_variance"] = float((denoised**2).sum() / sst * 100)


def _percent_of_variance(sum_squares, sst):
    """Express sums of squares as a percentage of each voxel's total sum of squares.

    Voxels with no variance are assigned 0.
    """
    out = np.zeros(sst.shape, dtype=float)
    valid = sst > 0
    out[valid] = 100 * sum_squares[valid] / sst[valid]
    return out


def calculate_variance_explained_maps(data, mixing, component_table=None):
    """Calculate voxel-wise maps of the percent variance explained by a decomposition.

    The data are regressed on the component time series in each voxel, after removing the
    mean over time, mirroring :func:`tedana.io.denoise_ts`.

    Parameters
    ----------
    data : (S x T) array_like
        Data to describe, such as optimally-combined data restricted to a mask.
    mixing : (T x C) array_like
        Component time series.
    component_table : :obj:`pandas.DataFrame` or None, optional
        Component table with a ``"classification"`` column.
        If provided, maps for the accepted and rejected components are also calculated.
        Default is None.

    Returns
    -------
    maps : dict of (S,) :obj:`numpy.ndarray`
        Percent of each voxel's variance, in the range 0-100.
        ``"total"`` is the variance explained by all components (R-squared x 100).
        When ``component_table`` is provided, ``"unmodeled"`` (``100 - total``), and
        ``"accepted"`` and ``"rejected"`` are added.
        The latter two are the variance of the data reconstructed from only the accepted or
        rejected components, using the betas from the full model.
        Because components are not necessarily orthogonal, the accepted, rejected, and
        unmodeled maps do not necessarily sum to 100.
    """
    data = np.asarray(data)
    mixing = np.asarray(mixing)
    dmdata = data - data.mean(axis=1, keepdims=True)
    sst = (dmdata**2).sum(axis=1)

    betas = get_coeffs(dmdata, mixing)
    sse = ((dmdata - betas.dot(mixing.T)) ** 2).sum(axis=1)
    unmodeled = _percent_of_variance(sse, sst)
    maps = {"total": np.where(sst > 0, 100 - unmodeled, 0)}

    if component_table is not None:
        maps["unmodeled"] = unmodeled
        for label in ("accepted", "rejected"):
            idx = component_table.loc[component_table["classification"] == label].index.values
            reconstruction = betas[:, idx].dot(mixing[:, idx].T)
            maps[label] = _percent_of_variance((reconstruction**2).sum(axis=1), sst)

    return maps


def write_variance_explained_maps(
    *,
    data_optcom,
    mixing,
    component_table,
    mask,
    io_generator,
    include_pca=True,
):
    """Write voxel-wise maps of the percent variance explained by the decompositions.

    Parameters
    ----------
    data_optcom : (Mb x T) array_like
        Optimally-combined data, in the space of the base mask.
    mixing : (T x C) array_like
        ICA mixing matrix used for denoising.
    component_table : :obj:`pandas.DataFrame`
        ICA component table with a ``"classification"`` column.
    mask : (Mb,) :obj:`numpy.ndarray` of bool
        Mask of voxels to describe.
    io_generator : :obj:`~tedana.io.OutputGenerator`
        The output generator for this workflow.
    include_pca : bool, optional
        Whether to also write the PCA map, which requires the "PCA mixing tsv" and
        "PCA metrics tsv" outputs. Default is True.

    Notes
    -----
    Writes the "ICA varexp img", "ICA accepted varexp img", "ICA rejected varexp img",
    "ICA unmodeled varexp img", and, if ``include_pca`` is True, "PCA varexp img" outputs.
    The PCA map uses the PCA components that were retained for ICA.
    """
    data = data_optcom[mask, :]

    ica_maps = calculate_variance_explained_maps(data, mixing, component_table)
    keys = {
        "total": "ICA varexp img",
        "accepted": "ICA accepted varexp img",
        "rejected": "ICA rejected varexp img",
        "unmodeled": "ICA unmodeled varexp img",
    }
    for label, key in keys.items():
        io_generator.save_file(ica_maps[label], key, mask=mask)

    if include_pca:
        pca_mixing = pd.read_table(io_generator.get_name("PCA mixing tsv")).to_numpy()
        pca_table = pd.read_table(io_generator.get_name("PCA metrics tsv"))
        retained = (pca_table["classification"] == "accepted").to_numpy()
        pca_maps = calculate_variance_explained_maps(data, pca_mixing[:, retained])
        io_generator.save_file(pca_maps["total"], "PCA varexp img", mask=mask)
