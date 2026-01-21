"""Metrics evaluating component TE-dependence or -independence."""

import logging
import typing

import nibabel as nb
import numpy as np
from scipy import stats

from tedana import io, utils
from tedana.stats import computefeats2, get_coeffs, t_to_z

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def calculate_weights(
    *,
    data_optcom: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    """Calculate standardized parameter estimates between data and mixing matrix.

    Parameters
    ----------
    data_optcom : (M x T) array_like
        Optimally combined data, already masked.
    mixing : (T x C) array_like
        Mixing matrix

    Returns
    -------
    weights : (M x C) array_like
        Standardized parameter estimates for optimally combined data against
        the mixing matrix.
    """
    assert data_optcom.shape[1] == mixing.shape[0]
    mixing_z = stats.zscore(mixing, axis=0)
    # compute un-normalized weight dataset (features)
    weights = computefeats2(data_optcom, mixing_z, normalize=False)
    return weights


def calculate_betas(
    *,
    data: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    """Calculate unstandardized parameter estimates between data and mixing matrix.

    Parameters
    ----------
    data : (M [x E] x T) array_like
        Data to calculate betas for
    mixing : (T x C) array_like
        Mixing matrix

    Returns
    -------
    betas : (M [x E] x C) array_like
        Unstandardized parameter estimates
    """
    if data.ndim == 2:
        data_optcom = data
        assert data_optcom.shape[1] == mixing.shape[0]
        # mean-center optimally-combined data
        data_optcom_dm = data_optcom - data_optcom.mean(axis=-1, keepdims=True)
        # betas are from a normal OLS fit of the mixing matrix against the mean-centered data
        betas = get_coeffs(data_optcom_dm, mixing)

    else:
        betas = np.zeros([data.shape[0], data.shape[1], mixing.shape[1]])
        for n_echo in range(data.shape[1]):
            betas[:, n_echo, :] = get_coeffs(data[:, n_echo, :], mixing)

    return betas


def calculate_psc(
    *,
    data_optcom: np.ndarray,
    optcom_betas: np.ndarray,
) -> np.ndarray:
    """Calculate percent signal change maps for components against optimally-combined data.

    Parameters
    ----------
    data_optcom : (M x T) array_like
        Optimally combined data, already masked.
    optcom_betas : (M x C) array_like
        Component-wise, unstandardized parameter estimates from the regression
        of the optimally combined data against component time series.

    Returns
    -------
    psc : (M x C) array_like
        Component-wise percent signal change maps.
    """
    assert data_optcom.shape[0] == optcom_betas.shape[0]
    psc = 100 * optcom_betas / data_optcom.mean(axis=-1, keepdims=True)
    return psc


def calculate_z_maps(
    *,
    weights: np.ndarray,
    z_max: float = 8,
) -> np.ndarray:
    """Calculate component-wise z-statistic maps.

    This is done by z-scoring standardized parameter estimate maps and cropping extreme values.

    Parameters
    ----------
    weights : (M x C) array_like
        Standardized parameter estimate maps for components.
    z_max : float, optional
        Maximum z-statistic, used to crop extreme values. Values in the
        z-statistic maps greater than this value are set to it.

    Returns
    -------
    z_maps : (M x C) array_like
        Z-statistic maps for components, reflecting voxel-wise component loadings.
    """
    z_maps = stats.zscore(weights, axis=0)
    extreme_idx = np.abs(z_maps) > z_max
    z_maps[extreme_idx] = z_max * np.sign(z_maps[extreme_idx])
    return z_maps


def calculate_f_maps(
    *,
    data_cat: np.ndarray,
    z_maps: np.ndarray,
    mixing: np.ndarray,
    adaptive_mask: np.ndarray,
    tes: np.ndarray,
    n_independent_echos=None,
    f_max: float = 500,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate pseudo-F-statistic maps for TE-dependence and -independence models.

    Parameters
    ----------
    data_cat : (M x E x T) array_like
        Multi-echo data, already masked.
    z_maps : (M x C) array_like
        Z-statistic maps for components, reflecting voxel-wise component loadings.
    mixing : (T x C) array_like
        Mixing matrix
    adaptive_mask : (M) array_like
        Adaptive mask, where each voxel's value is the number of echoes with
        "good signal". Limited to masked voxels.
    tes : (E) array_like
        Echo times in milliseconds, in the same order as the echoes in data_cat.
    n_independent_echos : int
        Number of independent echoes to use in goodness of fit metrics (fstat).
        Primarily used for EPTI acquisitions.
        If None, number of echoes will be used. Default is None.
    f_max : float, optional
        Maximum F-statistic, used to crop extreme values. Values in the
        F-statistic maps greater than this value are set to it.

    Returns
    -------
    f_t2_maps, f_s0_maps, pred_t2_maps, pred_s0_maps : (M x C) array_like
        Pseudo-F-statistic maps for TE-dependence and -independence models,
        respectively.
    """
    assert data_cat.shape[0] == z_maps.shape[0] == adaptive_mask.shape[0]
    assert data_cat.shape[1] == tes.shape[0]
    assert data_cat.shape[2] == mixing.shape[0]
    assert z_maps.shape[1] == mixing.shape[1]

    # TODO: Remove mask arg from get_coeffs
    me_betas = get_coeffs(data_cat, mixing, mask=np.ones(data_cat.shape[:2], bool), add_const=True)
    n_voxels, n_echos, n_components = me_betas.shape
    mu = data_cat.mean(axis=-1, dtype=float)
    tes = np.reshape(tes, (n_echos, 1))

    # set up Xmats
    x1 = mu.T  # Model 1
    x2 = np.tile(tes, (1, n_voxels)) * mu.T  # Model 2

    f_t2_maps = np.zeros([n_voxels, n_components])
    f_s0_maps = np.zeros([n_voxels, n_components])
    pred_t2_maps = np.zeros([n_voxels, len(tes), n_components])
    pred_s0_maps = np.zeros([n_voxels, len(tes), n_components])

    for i_comp in range(n_components):
        # size of comp_betas is (n_echoes, n_samples)
        comp_betas = np.atleast_3d(me_betas)[:, :, i_comp].T
        alpha = (np.abs(comp_betas) ** 2).sum(axis=0)

        # Only analyze good echoes at each voxel
        for j_echo in np.unique(adaptive_mask[adaptive_mask >= 3]):
            mask_idx = adaptive_mask == j_echo
            alpha = (np.abs(comp_betas[:j_echo]) ** 2).sum(axis=0)

            # S0 Model
            # (S,) model coefficient map
            coeffs_s0 = (comp_betas[:j_echo] * x1[:j_echo, :]).sum(axis=0) / (
                x1[:j_echo, :] ** 2
            ).sum(axis=0)
            pred_s0 = x1[:j_echo, :] * np.tile(coeffs_s0, (j_echo, 1))
            sse_s0 = (comp_betas[:j_echo] - pred_s0) ** 2
            sse_s0 = sse_s0.sum(axis=0)  # (S,) prediction error map
            if n_independent_echos is None or n_independent_echos >= j_echo:
                f_s0 = (alpha - sse_s0) * (j_echo - 1) / (sse_s0)
            else:
                f_s0 = (alpha - sse_s0) * (n_independent_echos - 1) / (sse_s0)
            f_s0[f_s0 > f_max] = f_max
            f_s0_maps[mask_idx, i_comp] = f_s0[mask_idx]

            # T2 Model
            coeffs_t2 = (comp_betas[:j_echo] * x2[:j_echo, :]).sum(axis=0) / (
                x2[:j_echo, :] ** 2
            ).sum(axis=0)
            pred_t2 = x2[:j_echo] * np.tile(coeffs_t2, (j_echo, 1))
            sse_t2 = (comp_betas[:j_echo] - pred_t2) ** 2
            sse_t2 = sse_t2.sum(axis=0)
            if n_independent_echos is None or n_independent_echos >= j_echo:
                f_t2 = (alpha - sse_t2) * (j_echo - 1) / (sse_t2)
            else:
                f_t2 = (alpha - sse_t2) * (n_independent_echos - 1) / (sse_t2)
            f_t2[f_t2 > f_max] = f_max
            f_t2_maps[mask_idx, i_comp] = f_t2[mask_idx]

            pred_s0_maps[mask_idx, :j_echo, i_comp] = pred_s0.T[mask_idx, :]
            pred_t2_maps[mask_idx, :j_echo, i_comp] = pred_t2.T[mask_idx, :]

    return f_t2_maps, f_s0_maps, pred_t2_maps, pred_s0_maps


def threshold_map(
    *,
    maps: np.ndarray,
    mask: np.ndarray,
    ref_img: nb.Nifti1Image,
    threshold: float,
    csize: typing.Union[int, None] = None,
) -> np.ndarray:
    """Perform cluster-extent thresholding.

    Parameters
    ----------
    maps : (M x C) array_like
        Statistical maps to be thresholded.
    mask : (S) array_like
        Binary mask.
    ref_img : img_like
        Reference image to convert to niimgs with.
    threshold : :obj:`float`
        Value threshold to apply to maps.
    csize : :obj:`int` or :obj:`None`, optional
        Minimum cluster size. If None, standard thresholding (non-cluster-extent) will be done.
        Default is None.

    Returns
    -------
    maps_thresh : (M x C) array_like
    """
    n_voxels, n_components = maps.shape
    maps_thresh = np.zeros([n_voxels, n_components], bool)
    if csize is None:
        csize = np.max([int(n_voxels * 0.0005) + 5, 20])
    else:
        csize = int(csize)

    for i_comp in range(n_components):
        # Cluster-extent threshold and binarize F-maps
        ccimg = io.new_nii_like(ref_img, np.squeeze(utils.unmask(maps[:, i_comp], mask)))

        maps_thresh[:, i_comp] = utils.threshold_map(
            ccimg, min_cluster_size=csize, threshold=threshold, mask=mask, binarize=True
        )
    return maps_thresh


def threshold_to_match(
    *,
    maps: np.ndarray,
    n_sig_voxels: np.ndarray,
    mask: np.ndarray,
    ref_img: nb.Nifti1Image,
    csize: typing.Union[int, None] = None,
) -> np.ndarray:
    """Cluster-extent threshold a map to target number of significant voxels.

    Resulting maps have roughly the requested number of significant voxels, after cluster-extent
    thresholding.

    Parameters
    ----------
    maps : (M x C) array_like
        Statistical maps to be thresholded.
    n_sig_voxels : (C) array_like
        Number of significant voxels to threshold to, for each map in maps.
    mask : (S) array_like
        Binary mask.
    ref_img : img_like
        Reference image to convert to niimgs with.
    csize : :obj:`int` or :obj:`None`, optional
        Minimum cluster size. If None, standard thresholding (non-cluster-extent) will be done.
        Default is None.

    Returns
    -------
    clmaps : (S x C) array_like
        Cluster-extent thresholded and binarized maps.
    """
    assert maps.shape[1] == n_sig_voxels.shape[0]

    n_voxels, n_components = maps.shape
    abs_maps = np.abs(maps)
    if csize is None:
        csize = np.max([int(n_voxels * 0.0005) + 5, 20])
    else:
        csize = int(csize)

    clmaps = np.zeros([n_voxels, n_components], bool)
    for i_comp in range(n_components):
        # Initial cluster-defining threshold is defined based on the number
        # of significant voxels from the F-statistic maps. This threshold
        # will be relaxed until the number of significant voxels from both
        # maps is roughly equal.
        ccimg = io.new_nii_like(ref_img, utils.unmask(stats.rankdata(abs_maps[:, i_comp]), mask))
        step = int(n_sig_voxels[i_comp] / 10)
        rank_thresh = n_voxels - n_sig_voxels[i_comp]

        while True:
            clmap = utils.threshold_map(
                ccimg,
                min_cluster_size=csize,
                threshold=rank_thresh,
                mask=mask,
                binarize=True,
            )
            if rank_thresh <= 0:  # all voxels significant
                break

            diff = n_sig_voxels[i_comp] - clmap.sum()
            if diff < 0 or clmap.sum() == 0:
                rank_thresh += step
                clmap = utils.threshold_map(
                    ccimg,
                    min_cluster_size=csize,
                    threshold=rank_thresh,
                    mask=mask,
                    binarize=True,
                )
                break
            else:
                rank_thresh -= step
        clmaps[:, i_comp] = clmap
    return clmaps


def calculate_dependence_metrics(
    *,
    f_t2_maps: np.ndarray,
    f_s0_maps: np.ndarray,
    z_maps: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Calculate Kappa and Rho metrics from F-statistic maps.

    Just a weighted average over voxels.

    Parameters
    ----------
    f_t2_maps, f_s0_maps : (S x C) array_like
        Pseudo-F-statistic maps for TE-dependence and -independence models,
        respectively.
    z_maps : (S x C) array_like
        Z-statistic maps for components, reflecting voxel-wise component loadings.

    Returns
    -------
    kappas, rhos : (C) array_like
        Averaged pseudo-F-statistics for TE-dependence and -independence
        models, respectively.
    """
    assert f_t2_maps.shape == f_s0_maps.shape == z_maps.shape

    RepLGR.info(
        "Kappa (kappa) and Rho (rho) were calculated as measures of "
        "TE-dependence and TE-independence, respectively."
    )

    weight_maps = z_maps**2.0
    n_components = z_maps.shape[1]
    kappas, rhos = np.zeros(n_components), np.zeros(n_components)
    for i_comp in range(n_components):
        kappas[i_comp] = np.average(f_t2_maps[:, i_comp], weights=weight_maps[:, i_comp])
        rhos[i_comp] = np.average(f_s0_maps[:, i_comp], weights=weight_maps[:, i_comp])
    return kappas, rhos


def calculate_varex(
    *,
    component_maps: np.ndarray,
) -> np.ndarray:
    """Calculate relative coefficient energy from parameter estimate maps.

    This measure indicates the relative coefficient magnitude across voxels for each component.

    Parameters
    ----------
    component_maps : (S x C) array_like
        Component-wise parameter estimates from the regression
        of the optimally combined data against component time series.

    Returns
    -------
    varex : (C) array_like
        Average (across voxels) relative coefficient energy for each component,
        on a scale from 0 to 100.
    """
    compvar = (component_maps**2).sum(axis=0)
    varex = 100 * (compvar / compvar.sum())
    return varex


def calculate_relative_varex(
    *,
    data_optcom: np.ndarray,
    component_maps: np.ndarray,
) -> np.ndarray:
    """Calculate relative component-wise contribution scaled by total DV variance.

    This estimates component-wise explained variance by fitting a single multivariate
    least-squares model and computing the variance of each regressor's fitted contribution to the
    signal. Because regressors may be correlated, these values reflect model-based variance
    attribution rather than unique or partial explained variance.

    This is not a common measure of variance explained, since any shared variance between
    regressors will be split between them.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined data.
    component_maps : (S x C) array_like
        Component-wise parameter estimates from the regression
        of the optimally combined data against component time series.

    Returns
    -------
    relative_varex : (C) array_like
        Component-wise contribution values, scaled by total DV variance.
        These do not sum to 100 unless the model explains all variance.
    """
    if data_optcom.shape[0] != component_maps.shape[0]:
        raise ValueError(
            f"First dimension (number of voxels) of data ({data_optcom.shape[0]}) "
            f"does not match first dimension of component maps ({component_maps.shape[0]})."
        )

    # XXX: This requires the same scaling as used to calculate the component maps.
    data_dm = data_optcom - data_optcom.mean(axis=0, keepdims=True)
    coeff_energy = np.sum(component_maps**2, axis=0)
    total_var = np.sum(data_dm**2)
    relative_varex = 100 * (coeff_energy / total_var)
    return relative_varex


def calculate_marginal_r2(
    *,
    data_optcom: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    """Calculate mean voxel-wise marginal R-squared for each component against the data.

    This is equivalent to the variance explained by each component without controlling
    for other components. Mathematically, it is equivalent to 100 * the squared correlation
    between the component time series and the data, averaged over voxels.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined data.
    mixing : (T x C) array_like
        Mixing matrix.

    Returns
    -------
    marginal_r2 : (C) array_like
        Average (across voxels) marginal R-squared for each component, on a scale from 0 to 100.
    """
    if data_optcom.shape[1] != mixing.shape[0]:
        raise ValueError(
            f"Second dimensions (number of volumes) of data ({data_optcom.shape[1]}) "
            f"and mixing ({mixing.shape[0]}) do not match."
        )

    mixing = stats.zscore(mixing, axis=0)
    data_optcom = stats.zscore(data_optcom, axis=1)
    total_var = (data_optcom**2).sum()

    marginal_r2 = np.zeros(mixing.shape[1])
    for i_comp in range(mixing.shape[1]):
        # Separate out each component's time series from the mixing matrix and calculate the
        # variance explained for that component.
        ts = mixing[:, i_comp][:, None]
        beta = np.linalg.lstsq(ts, data_optcom.T, rcond=None)[0].T
        marginal_r2[i_comp] = 100 * (1 - ((data_optcom - beta.dot(ts.T)) ** 2.0).sum() / total_var)

    return marginal_r2


def calculate_marginal_r2_2(
    *,
    data_optcom: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    """Calculate mean voxel-wise marginal R-squared for each component against the data.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined data.
    mixing : (T x C) array_like
        Mixing matrix.

    Returns
    -------
    marginal_r2 : (C) array_like
        Average (across voxels) marginal R-squared for each component, on a scale from 0 to 100.
    """
    n_vols = mixing.shape[0]

    # z-score voxels (timewise)
    mixing = stats.zscore(mixing, axis=0)
    data_optcom = stats.zscore(data_optcom, axis=1)

    # S x C correlation matrix of each component with the data
    # correlation = (1/(T-1)) * dat_z @ mix_z
    # XXX: We use population scaling here to match the behavior of np.corrcoef.
    correlations = data_optcom @ mixing / n_vols
    correlations = correlations ** 2
    marginal_r2 = correlations.mean(axis=0)
    return 100 * marginal_r2


def calculate_partial_r2(
    *,
    data_optcom: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    """Calculate mean voxelwise partial R-squared for each regressor.

    This is equivalent to the variance explained by each component after regressing the other
    components out of the data *and* the component itself. It is a conditional effect size.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined data.
    mixing : (T x C) array_like
        Mixing matrix.

    Returns
    -------
    r2_partial : (C) array_like
        Average (across voxels) partial R-squared for each regressor, on a scale from 0 to 100.
    """
    if data_optcom.shape[1] != mixing.shape[0]:
        raise ValueError(
            f"Second dimension (number of volumes) of data ({data_optcom.shape[1]}) "
            f"does not match first dimension of mixing ({mixing.shape[0]})."
        )

    n_components = mixing.shape[1]

    r2_partial = np.zeros(n_components)
    data_optcom_t = data_optcom.T  # shape (T, S)

    for i_comp in range(n_components):
        x_others = np.delete(mixing, i_comp, axis=1)
        x_r = mixing[:, i_comp]

        # Residualize all voxel time series with respect to the other components at once.
        beta_others, *_ = np.linalg.lstsq(x_others, data_optcom_t, rcond=None)
        y_res_all = data_optcom_t - x_others @ beta_others  # (T, S)

        # Residualize the target regressor with respect to the other components.
        x_res = _residualize(x_r, x_others)  # (T,)

        denom = np.dot(x_res, x_res)
        numer = x_res @ y_res_all  # (S,)
        beta = numer / denom  # (S,)

        resid = y_res_all - x_res[:, None] * beta  # (T, S)

        rss = np.sum(resid**2, axis=0)
        tss = np.sum(y_res_all**2, axis=0)
        r2_vox = 1 - rss / tss

        r2_partial[i_comp] = r2_vox.mean()

    return r2_partial


def calculate_semi_partial_r2(
    *,
    data_optcom: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    """Calculate mean voxelwise semi-partial R-squared for each regressor.

    Semi-partial R^2 is the incremental variance explained by adding a
    regressor to a model that already contains all other regressors.

    TODO: Simplify this by (1) orthogonalizing each component w.r.t. the other components,
    then (2) calculating the R-squared for each component against the data.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined data.
    mixing : (T x C) array_like
        Mixing matrix.

    Returns
    -------
    semi_partial_r2 : (C) array_like
        Average (across voxels) semi-partial R-squared for each regressor,
        on a scale from 0 to 100.
    """
    if data_optcom.shape[1] != mixing.shape[0]:
        raise ValueError(
            f"Second dimension (number of volumes) of data ({data_optcom.shape[1]}) "
            f"does not match first dimension of mixing ({mixing.shape[0]})."
        )

    # Z-score
    mixing = stats.zscore(mixing, axis=0)
    data_optcom = stats.zscore(data_optcom, axis=1)

    # Full model
    beta_full, *_ = np.linalg.lstsq(mixing, data_optcom.T, rcond=None)
    resid_full = data_optcom.T - mixing @ beta_full
    rss_full = np.sum(resid_full**2, axis=0)  # (S,)

    # Total sum of squares (per voxel)
    tss = np.sum(data_optcom**2, axis=1)  # (S,)

    n_components = mixing.shape[1]
    r2_semi = np.zeros(n_components)

    for i_comp in range(n_components):
        x_red = np.delete(mixing, i_comp, axis=1)

        beta_red, *_ = np.linalg.lstsq(x_red, data_optcom.T, rcond=None)
        resid_red = data_optcom.T - x_red @ beta_red
        rss_red = np.sum(resid_red**2, axis=0)

        r2_vox = (rss_red - rss_full) / tss
        r2_semi[i_comp] = r2_vox.mean()

    return 100 * r2_semi


def calculate_semipartial_r2(
    *,
    data_optcom: np.ndarray,
    mixing: np.ndarray,
) -> np.ndarray:
    """Calculate mean voxelwise semi-partial R-squared for each regressor.

    Semi-partial R^2 is the incremental variance explained by adding a
    regressor to a model that already contains all other regressors.

    TODO: Simplify this by (1) orthogonalizing each component w.r.t. the other components,
    then (2) calculating the R-squared for each component against the data.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined data.
    mixing : (T x C) array_like
        Mixing matrix.

    Returns
    -------
    semi_partial_r2 : (C) array_like
        Average (across voxels) semi-partial R-squared for each regressor,
        on a scale from 0 to 100.
    """
    if data_optcom.shape[1] != mixing.shape[0]:
        raise ValueError(
            f"Second dimension (number of volumes) of data ({data_optcom.shape[1]}) "
            f"does not match first dimension of mixing ({mixing.shape[0]})."
        )

    # Z-score
    mixing = stats.zscore(mixing, axis=0)
    data_optcom = stats.zscore(data_optcom, axis=1)

    # Orthogonalize each component with respect to the other components
    mixing = orthogonalize_by_others(arr=mixing)

    # S x C correlation matrix of each component with the data
    correlations = np.corrcoef(mixing.T, data_optcom)[mixing.shape[1]:, :mixing.shape[1]]
    correlations = correlations ** 2
    r2_semi = correlations.mean(axis=0)

    return 100 * r2_semi


def orthogonalize_by_others(*, arr: np.ndarray) -> np.ndarray:
    """Orthogonalize each column of the input array with respect to the other columns.

    Parameters
    ----------
    arr : (T x C) array_like
        Array to orthogonalize.

    Returns
    -------
    out : (T x C) array_like
        Orthogonalized array.
    """
    arr = np.asarray(arr, float)
    n_components = arr.shape[1]
    out = np.empty_like(arr)

    for j_comp in range(n_components):
        others = np.delete(arr, j_comp, axis=1)
        # coefficients for projecting column j onto the span of the other columns
        coef = np.linalg.lstsq(others, arr[:, j_comp], rcond=None)[0]
        proj = others @ coef
        out[:, j_comp] = arr[:, j_comp] - proj

    return out


def compute_dice(
    *,
    clmaps1: np.ndarray,
    clmaps2: np.ndarray,
    axis: typing.Union[int, None] = 0,
) -> np.ndarray:
    """Compute the Dice similarity index between two thresholded and binarized maps.

    NaNs are converted automatically to zeroes.

    Parameters
    ----------
    clmaps1, clmaps2 : (S x C) array_like
        Thresholded and binarized arrays.
    axis : int or None, optional
        Axis along which to calculate DSI. Default is 0.

    Returns
    -------
    dice_values : array_like
        DSI values.
    """
    assert clmaps1.shape == clmaps2.shape

    dice_values = utils.dice(clmaps1, clmaps2, axis=axis)
    dice_values = np.nan_to_num(dice_values, 0)
    return dice_values


def compute_signal_minus_noise_z(
    *,
    z_maps: np.ndarray,
    z_clmaps: np.ndarray,
    f_t2_maps: np.ndarray,
    z_thresh: float = 1.95,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compare signal and noise z-statistic distributions with a two-sample t-test.

    Divide voxel-level thresholded F-statistic maps into distributions of
    signal (voxels in significant clusters) and noise (voxels from
    non-significant clusters) statistics, then compare these distributions
    with a two-sample t-test. Convert the resulting t-statistics (per map)
    to normally distributed z-statistics.

    Parameters
    ----------
    z_maps : (S x C) array_like
        Z-statistic maps for components, reflecting voxel-wise component loadings.
    z_clmaps : (S x C) array_like
        Cluster-extent thresholded Z-statistic maps for components.
    f_t2_maps : (S x C) array_like
        Pseudo-F-statistic maps for components from TE-dependence models.
        Each voxel reflects the model fit for the component weights to the
        TE-dependence model across echoes.
    z_thresh : float, optional
        Z-statistic threshold for voxel-wise significance. Default is 1.95.

    Returns
    -------
    signal_minus_noise_z : (C) array_like
        Z-statistics from component-wise signal > noise paired t-tests.
    signal_minus_noise_p : (C) array_like
        P-values from component-wise signal > noise paired t-tests.
    """
    assert z_maps.shape == z_clmaps.shape == f_t2_maps.shape

    n_components = z_maps.shape[1]
    signal_minus_noise_z = np.zeros(n_components)
    signal_minus_noise_p = np.zeros(n_components)
    noise_idx = (np.abs(z_maps) > z_thresh) & (z_clmaps == 0)
    countnoise = noise_idx.sum(axis=0)
    countsignal = z_clmaps.sum(axis=0)
    for i_comp in range(n_components):
        noise_ft2_z = 0.5 * np.log(f_t2_maps[noise_idx[:, i_comp], i_comp])
        signal_ft2_z = 0.5 * np.log(f_t2_maps[z_clmaps[:, i_comp] == 1, i_comp])
        n_noise_dupls = noise_ft2_z.size - np.unique(noise_ft2_z).size
        if n_noise_dupls:
            LGR.debug(
                f"For component {i_comp}, {n_noise_dupls} duplicate noise F-values detected."
            )
        n_signal_dupls = signal_ft2_z.size - np.unique(signal_ft2_z).size
        if n_signal_dupls:
            LGR.debug(
                f"For component {i_comp}, {n_signal_dupls} duplicate signal F-values detected."
            )
        dof = countnoise[i_comp] + countsignal[i_comp] - 2

        t_value, signal_minus_noise_p[i_comp] = stats.ttest_ind(
            signal_ft2_z, noise_ft2_z, equal_var=False
        )
        signal_minus_noise_z[i_comp] = t_to_z(t_value, dof)

    signal_minus_noise_z = np.nan_to_num(signal_minus_noise_z, 0)
    signal_minus_noise_p = np.nan_to_num(signal_minus_noise_p, 0)
    return signal_minus_noise_z, signal_minus_noise_p


def compute_signal_minus_noise_t(
    *,
    z_maps: np.ndarray,
    z_clmaps: np.ndarray,
    f_t2_maps: np.ndarray,
    z_thresh: float = 1.95,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compare signal and noise t-statistic distributions with a two-sample t-test.

    Divide voxel-level thresholded F-statistic maps into distributions of
    signal (voxels in significant clusters) and noise (voxels from
    non-significant clusters) statistics, then compare these distributions
    with a two-sample t-test.

    Parameters
    ----------
    z_maps : (S x C) array_like
        Z-statistic maps for components, reflecting voxel-wise component loadings.
    z_clmaps : (S x C) array_like
        Cluster-extent thresholded Z-statistic maps for components.
    f_t2_maps : (S x C) array_like
        Pseudo-F-statistic maps for components from TE-dependence models.
        Each voxel reflects the model fit for the component weights to the
        TE-dependence model across echoes.
    z_thresh : float, optional
        Z-statistic threshold for voxel-wise significance. Default is 1.95.

    Returns
    -------
    signal_minus_noise_t : (C) array_like
        T-statistics from component-wise signal > noise paired t-tests.
    signal_minus_noise_p : (C) array_like
        P-values from component-wise signal > noise paired t-tests.
    """
    assert z_maps.shape == z_clmaps.shape == f_t2_maps.shape

    n_components = z_maps.shape[1]
    signal_minus_noise_t = np.zeros(n_components)
    signal_minus_noise_p = np.zeros(n_components)
    noise_idx = (np.abs(z_maps) > z_thresh) & (z_clmaps == 0)
    for i_comp in range(n_components):
        # NOTE: Why only compare distributions of *unique* F-statistics?
        noise_ft2_z = np.log10(np.unique(f_t2_maps[noise_idx[:, i_comp], i_comp]))
        signal_ft2_z = np.log10(np.unique(f_t2_maps[z_clmaps[:, i_comp] == 1, i_comp]))
        (signal_minus_noise_t[i_comp], signal_minus_noise_p[i_comp]) = stats.ttest_ind(
            signal_ft2_z, noise_ft2_z, equal_var=False
        )

    signal_minus_noise_t = np.nan_to_num(signal_minus_noise_t, 0)
    signal_minus_noise_p = np.nan_to_num(signal_minus_noise_p, 0)
    return signal_minus_noise_t, signal_minus_noise_p


def compute_countsignal(
    *,
    stat_cl_maps: np.ndarray,
) -> np.ndarray:
    """Count the number of significant voxels in a set of cluster-extent thresholded maps.

    Parameters
    ----------
    stat_cl_maps : (S x C) array_like
        Statistical map after cluster-extent thresholding and binarization.

    Returns
    -------
    countsignal : (C) array_like
        Number of significant (non-zero) voxels for each map in cl_arr.
    """
    countsignal = stat_cl_maps.sum(axis=0)
    return countsignal


def compute_countnoise(
    *,
    stat_maps: np.ndarray,
    stat_cl_maps: np.ndarray,
    stat_thresh: float = 1.95,
) -> np.ndarray:
    """Count the number of significant voxels from non-significant clusters.

    This is done after application of a cluster-defining threshold, but compared against results
    from cluster-extent thresholding.

    Parameters
    ----------
    stat_maps : (S x C) array_like
        Unthresholded statistical maps.
    stat_cl_maps : (S x C) array_like
        Cluster-extent thresholded and binarized version of stat_maps.
    stat_thresh : float, optional
        Statistical threshold. Default is 1.95 (Z-statistic threshold
        corresponding to p<X one-sided).

    Returns
    -------
    countnoise : (C) array_like
        Numbers of significant non-cluster voxels from the statistical maps.
    """
    assert stat_maps.shape == stat_cl_maps.shape

    noise_idx = (np.abs(stat_maps) > stat_thresh) & (stat_cl_maps == 0)
    countnoise = noise_idx.sum(axis=0)
    return countnoise


def generate_decision_table_score(
    *,
    kappa: np.ndarray,
    dice_ft2: np.ndarray,
    signal_minus_noise_t: np.ndarray,
    countnoise: np.ndarray,
    countsig_ft2: np.ndarray,
) -> np.ndarray:
    """Generate a five-metric decision table.

    Metrics are ranked in either descending or ascending order if they measure TE-dependence or
    -independence, respectively, and are then averaged for each component.

    Parameters
    ----------
    kappa : (C) array_like
        Pseudo-F-statistics for TE-dependence model.
    dice_ft2 : (C) array_like
        Dice similarity index for cluster-extent thresholded beta maps and
        cluster-extent thresholded TE-dependence F-statistic maps.
    signal_minus_noise_t : (C) array_like
        Signal-noise t-statistic metrics.
    countnoise : (C) array_like
        Numbers of significant non-cluster voxels from the thresholded beta
        maps.
    countsig_ft2 : (C) array_like
        Numbers of significant voxels from clusters from the thresholded
        TE-dependence F-statistic maps.

    Returns
    -------
    d_table_score : (C) array_like
        Decision table metric scores.
    """
    assert (
        kappa.shape
        == dice_ft2.shape
        == signal_minus_noise_t.shape
        == countnoise.shape
        == countsig_ft2.shape
    )

    d_table_rank = np.vstack(
        [
            len(kappa) - stats.rankdata(kappa),
            len(kappa) - stats.rankdata(dice_ft2),
            len(kappa) - stats.rankdata(signal_minus_noise_t),
            stats.rankdata(countnoise),
            len(kappa) - stats.rankdata(countsig_ft2),
        ]
    ).T
    d_table_score = d_table_rank.mean(axis=1)
    return d_table_score


def compute_kappa_rho_difference(*, kappa: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Compute the proportion of pseudo-F-statistics that is dominated by either kappa or rho.

    Parameters
    ----------
    kappa : (C) array_like
        Kappa values.
    rho : (C) array_like
        Rho values.

    Returns
    -------
    kappa_rho_difference : (C) array_like
        Proportion of pseudo-F-statistics that is dominated by either kappa or rho.
        Higher values indicate that either kappa or rho is dominating the component.
        Lower values indicate that both kappa and rho are contributing to the component.
    """
    assert kappa.shape == rho.shape

    return np.abs(kappa - rho) / (kappa + rho)


def _residualize(y, x):
    """Residualize y with respect to x.

    Parameters
    ----------
    y : (T,) array
        Dependent variable.
    x : (T, K) array
        Independent variables.

    Returns
    -------
    y_res : (T,) array
        Residuals of y with respect to x.
    """
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta
