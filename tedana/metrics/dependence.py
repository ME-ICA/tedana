"""Metrics evaluating component TE-dependence or -independence."""

import logging
import typing

import nibabel as nb
import numpy as np
from scipy import stats

from tedana import io, utils
from tedana.metrics._utils import get_value_thresholds
from tedana.stats import get_coeffs, t_to_z

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
    mixing = stats.zscore(mixing, axis=0)
    data_optcom = stats.zscore(data_optcom, axis=-1)
    # compute standardized parameter estimates
    weights = get_coeffs(data_optcom, mixing)
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
    assert data_cat.shape[0] == adaptive_mask.shape[0]
    assert data_cat.shape[1] == tes.shape[0]
    assert data_cat.shape[2] == mixing.shape[0]

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
    proportion_threshold: float = None,
    value_threshold: float = None,
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
    proportion_threshold : :obj:`float`
        Proportion threshold to apply to maps. Values between 0 and 100.
    value_threshold : float, optional
        Value threshold to apply to maps. Default is None.
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

    value_threshold = get_value_thresholds(
        maps=maps,
        proportion_threshold=proportion_threshold,
        value_threshold=value_threshold,
    )

    for i_comp in range(n_components):
        # Cluster-extent threshold and binarize F-maps
        ccimg = io.new_nii_like(ref_img, np.squeeze(utils.unmask(maps[:, i_comp], mask)))

        maps_thresh[:, i_comp] = utils.threshold_map(
            ccimg,
            min_cluster_size=csize,
            threshold=value_threshold[i_comp],
            mask=mask,
            binarize=True,
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
    """Calculate variance explained from parameter estimate maps.

    Parameters
    ----------
    component_maps : (S x C) array_like
        Component-wise parameter estimates from the regression
        of the optimally combined data against component time series.

    Returns
    -------
    varex : (C) array_like
        Variance explained for each component, on a scale from 0 to 100.
    """
    compvar = (component_maps**2).sum(axis=0)
    varex = 100 * (compvar / compvar.sum())
    return varex


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
    value_threshold: float = None,
    proportion_threshold: float = None,
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
    value_threshold : float, optional
        Threshold for voxel-wise significance in input ``z_maps``. Default is None.
    proportion_threshold : float, optional
        Proportion threshold for voxel-wise significance in input ``z_maps``.
        Values between 0 and 100.
        Default is None.

    Returns
    -------
    signal_minus_noise_z : (C) array_like
        Z-statistics from component-wise signal > noise paired t-tests.
    signal_minus_noise_p : (C) array_like
        P-values from component-wise signal > noise paired t-tests.
    """
    assert z_maps.shape == z_clmaps.shape == f_t2_maps.shape

    value_threshold = get_value_thresholds(
        maps=z_maps,
        proportion_threshold=proportion_threshold,
        value_threshold=value_threshold,
    )

    n_components = z_maps.shape[1]
    signal_minus_noise_z = np.zeros(n_components)
    signal_minus_noise_p = np.zeros(n_components)
    noise_idx = (np.abs(z_maps) > value_threshold) & (z_clmaps == 0)
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
    value_threshold: float = None,
    proportion_threshold: float = None,
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
    value_threshold : float, optional
        Threshold for voxel-wise significance in input ``z_maps``. Default is None.
    proportion_threshold : float, optional
        Proportion threshold for voxel-wise significance in input ``z_maps``.
        Values between 0 and 100.
        Default is None.

    Returns
    -------
    signal_minus_noise_t : (C) array_like
        T-statistics from component-wise signal > noise paired t-tests.
    signal_minus_noise_p : (C) array_like
        P-values from component-wise signal > noise paired t-tests.
    """
    assert z_maps.shape == z_clmaps.shape == f_t2_maps.shape

    value_threshold = get_value_thresholds(
        maps=z_maps,
        proportion_threshold=proportion_threshold,
        value_threshold=value_threshold,
    )

    n_components = z_maps.shape[1]
    signal_minus_noise_t = np.zeros(n_components)
    signal_minus_noise_p = np.zeros(n_components)
    noise_idx = (np.abs(z_maps) > value_threshold) & (z_clmaps == 0)
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
    value_threshold: float = None,
    proportion_threshold: float = None,
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
    value_threshold : float, optional
        Threshold for voxel-wise significance in input ``stat_maps``. Default is None.
    proportion_threshold : float, optional
        Proportion threshold for voxel-wise significance in input ``stat_maps``.
        Values between 0 and 100.
        Default is None.

    Returns
    -------
    countnoise : (C) array_like
        Numbers of significant non-cluster voxels from the statistical maps.
    """
    assert stat_maps.shape == stat_cl_maps.shape
    value_threshold = get_value_thresholds(
        maps=stat_maps,
        proportion_threshold=proportion_threshold,
        value_threshold=value_threshold,
    )

    noise_idx = (np.abs(stat_maps) > value_threshold) & (stat_cl_maps == 0)
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


def component_te_variance_tests_voxelwise(
    echowise_pes: np.ndarray,
    tes: np.ndarray,
    s0_hat: np.ndarray,
    t2s_hat: np.ndarray,
    adaptive_mask: np.ndarray,
):
    """Calculate component-wise S0/T2* contributions to explainable variance.

    Perform voxel-wise variance decomposition of multi-echo ICA components
    into S0-driven and T2*-driven contributions using physically motivated,
    nested linear models.

    This function evaluates, at each voxel and for each ICA component,
    how much of the echo-wise component parameter estimates can be attributed
    to changes in the monoexponential signal intercept (S0) versus changes in
    the decay constant (T2*).

    The method proceeds as follows:

    1. For each voxel, construct basis functions corresponding to the
       first-order linearization of the monoexponential signal model
       around the voxel's baseline S0 and T2* estimates:
           - phi_s0: sensitivity to S0 fluctuations
           - phi_t2: sensitivity to T2* fluctuations

    2. Fit three nested linear models to the echo-wise component parameter
       estimates:
           - Full model:     [phi_s0, phi_t2]
           - S0-only model:  [phi_s0]
           - T2*-only model: [phi_t2]

    3. Use partial F-tests to quantify whether adding the T2* term (or S0 term)
       significantly improves model fit beyond the reduced model.

    4. Decompose the explainable variance into S0 and T2* components and
       compute voxel-wise variance fractions:
           - kappa_star: fraction of explainable variance attributable to T2*
           - rho_star:   fraction of explainable variance attributable to S0

    Echo-wise fits and degrees of freedom are adjusted per voxel using the
    adaptive_mask, which specifies the number of valid echoes at each voxel.

    Parameters
    ----------
    echowise_pes : (n_voxels, n_echos, n_comps) array
        Echo-wise *unstandardized* ICA component parameter estimates.
        These should reflect the physical scaling of signal across echoes.
    tes : (n_echos,) array
        Echo times in milliseconds.
    s0_hat : (n_voxels,) or (n_voxels, n_vols) array
        Voxel-wise (or voxel- and volume-wise) baseline S0 estimates from
        monoexponential fitting.
    t2s_hat : (n_voxels,) or (n_voxels, n_vols) array
        Voxel-wise (or voxel- and volume-wise) baseline T2* estimates from
        monoexponential fitting.
    adaptive_mask : (n_voxels,) array of int
        Number of valid echoes per voxel (used to truncate echo-wise fits
        and compute voxel-wise degrees of freedom).

    Returns
    -------
    f_t2star : (n_voxels, n_comps) array
        Partial F-statistics testing the contribution of the T2* term
        conditional on the S0 term.
    f_s0 : (n_voxels, n_comps) array
        Partial F-statistics testing the contribution of the S0 term
        conditional on the T2* term.
    ss_t2 : (n_voxels, n_comps) array
        Unique sum of squares attributable to T2*-driven effects (Type III SS).
        This is the reduction in SSE when adding the T2* term to a model
        that already contains the S0 term.
    ss_s0 : (n_voxels, n_comps) array
        Unique sum of squares attributable to S0-driven effects (Type III SS).
        This is the reduction in SSE when adding the S0 term to a model
        that already contains the T2* term.

    Notes
    -----
    - Component-level variance fractions (kappa_star, rho_star) should be
      computed by aggregating ss_t2 and ss_s0 across voxels before taking
      ratios, rather than averaging voxel-wise ratios. This preserves the
      variance decomposition interpretation at the component level.
    """
    if not (
        echowise_pes.shape[0]
        == s0_hat.shape[0]
        == t2s_hat.shape[0]
        == adaptive_mask.shape[0]
    ):
        raise ValueError(
            "echowise_pes, s0_hat, t2s_hat, and adaptive_mask must have the same number of voxels"
        )
    if echowise_pes.shape[1] != len(tes):
        raise ValueError("echowise_pes and tes must have the same number of echoes")

    n_voxels, n_echos, n_comps = echowise_pes.shape
    tes = np.asarray(tes, dtype=np.float64)

    # Detect whether baseline estimates are voxel-wise or voxel+volume-wise
    volume_wise = s0_hat.ndim == 2

    # Initialize output arrays
    f_t2star = np.full((n_voxels, n_comps), np.nan)
    f_s0 = np.full((n_voxels, n_comps), np.nan)
    ss_t2_out = np.full((n_voxels, n_comps), np.nan)
    ss_s0_out = np.full((n_voxels, n_comps), np.nan)

    # Process voxels grouped by number of valid echoes for efficiency
    unique_n_echoes = np.unique(adaptive_mask[adaptive_mask >= 3])

    for n_e in unique_n_echoes:
        df_den = n_e - 2
        if df_den <= 0:
            continue

        # Get voxel indices with this number of echoes
        voxel_mask = adaptive_mask == n_e
        voxel_indices = np.where(voxel_mask)[0]

        # Filter for valid baseline estimates
        if volume_wise:
            valid_baseline = np.all((s0_hat[voxel_indices] > 0) & (t2s_hat[voxel_indices] > 0), axis=1)
        else:
            valid_baseline = (s0_hat[voxel_indices] > 0) & (t2s_hat[voxel_indices] > 0)

        voxel_indices = voxel_indices[valid_baseline]
        if len(voxel_indices) == 0:
            continue

        tes_v = tes[:n_e]

        # Compute basis functions for all voxels in batch
        if volume_wise:
            # (n_vox_batch, n_vols)
            s0_batch = s0_hat[voxel_indices]
            t2s_batch = t2s_hat[voxel_indices]

            # phi_s0: (n_vox_batch, n_e) - averaged over volumes
            # exp(-tes / t2s) for each voxel/volume, then mean over volumes
            phi_s0 = np.mean(
                np.exp(-tes_v[None, :, None] / t2s_batch[:, None, :]),
                axis=2,
            )

            # phi_t2: (n_vox_batch, n_e)
            phi_t2 = np.mean(
                s0_batch[:, None, :]
                * np.exp(-tes_v[None, :, None] / t2s_batch[:, None, :])
                * tes_v[None, :, None]
                / (t2s_batch[:, None, :] ** 2),
                axis=2,
            )
        else:
            # (n_vox_batch,) baseline estimates
            s0_batch = s0_hat[voxel_indices]
            t2s_batch = t2s_hat[voxel_indices]

            # phi_s0: (n_vox_batch, n_e)
            phi_s0 = np.exp(-tes_v[None, :] / t2s_batch[:, None])

            # phi_t2: (n_vox_batch, n_e)
            phi_t2 = (
                s0_batch[:, None]
                * np.exp(-tes_v[None, :] / t2s_batch[:, None])
                * tes_v[None, :]
                / (t2s_batch[:, None] ** 2)
            )

        # Extract echo-wise parameter estimates for this batch: (n_vox_batch, n_e, n_comps)
        Y = echowise_pes[voxel_indices, :n_e, :]

        # Compute SSE for single-predictor models using closed-form OLS
        # For y = x*b + e, b = (x'y)/(x'x), SSE = y'y - (x'y)^2/(x'x)

        # y'y for all voxels/components: (n_vox_batch, n_comps)
        yty = np.sum(Y ** 2, axis=1)

        # S0-only model: SSE_s0 = y'y - (phi_s0'y)^2 / (phi_s0'phi_s0)
        phi_s0_norm_sq = np.sum(phi_s0 ** 2, axis=1, keepdims=True)  # (n_vox_batch, 1)
        phi_s0_dot_y = np.einsum("ve,vec->vc", phi_s0, Y)  # (n_vox_batch, n_comps)
        sse_s0 = yty - (phi_s0_dot_y ** 2) / phi_s0_norm_sq

        # T2*-only model: SSE_t2 = y'y - (phi_t2'y)^2 / (phi_t2'phi_t2)
        phi_t2_norm_sq = np.sum(phi_t2 ** 2, axis=1, keepdims=True)  # (n_vox_batch, 1)
        phi_t2_dot_y = np.einsum("ve,vec->vc", phi_t2, Y)  # (n_vox_batch, n_comps)
        sse_t2 = yty - (phi_t2_dot_y ** 2) / phi_t2_norm_sq

        # Full model: need to solve 2x2 normal equations per voxel
        # X = [phi_s0, phi_t2], X'X is 2x2, X'Y is 2 x n_comps
        # SSE_full = y'y - y'X(X'X)^{-1}X'y

        # Compute X'X elements: (n_vox_batch,) each
        xtx_00 = phi_s0_norm_sq.squeeze()  # phi_s0' phi_s0
        xtx_11 = phi_t2_norm_sq.squeeze()  # phi_t2' phi_t2
        xtx_01 = np.sum(phi_s0 * phi_t2, axis=1)  # phi_s0' phi_t2

        # Determinant of X'X: (n_vox_batch,)
        det = xtx_00 * xtx_11 - xtx_01 ** 2

        # Handle singular/near-singular cases
        valid_det = det > 1e-10 * xtx_00 * xtx_11
        if not np.any(valid_det):
            continue

        # Inverse of X'X (only for valid determinants)
        # (X'X)^{-1} = (1/det) * [[xtx_11, -xtx_01], [-xtx_01, xtx_00]]
        # Use np.divide with where parameter to avoid divide-by-zero warnings
        inv_00 = np.zeros_like(det)
        inv_11 = np.zeros_like(det)
        inv_01 = np.zeros_like(det)
        np.divide(xtx_11, det, out=inv_00, where=valid_det)
        np.divide(xtx_00, det, out=inv_11, where=valid_det)
        np.divide(-xtx_01, det, out=inv_01, where=valid_det)

        # X'Y: (n_vox_batch, 2, n_comps)
        xty_0 = phi_s0_dot_y  # (n_vox_batch, n_comps)
        xty_1 = phi_t2_dot_y  # (n_vox_batch, n_comps)

        # y'X(X'X)^{-1}X'y = xty' @ inv @ xty (per voxel)
        # = inv_00 * xty_0^2 + inv_11 * xty_1^2 + 2 * inv_01 * xty_0 * xty_1
        quadform = (
            inv_00[:, None] * xty_0 ** 2
            + inv_11[:, None] * xty_1 ** 2
            + 2 * inv_01[:, None] * xty_0 * xty_1
        )

        sse_full = yty - quadform

        # Type III sums of squares
        ss_t2 = sse_s0 - sse_full
        ss_s0 = sse_t2 - sse_full

        # Compute F-statistics (only where SSE_full > 0 and determinant is valid)
        valid = valid_det[:, None] & (sse_full > 0)

        # F = (SS / df_num) / (SSE_full / df_den), with df_num = 1
        f_t2_batch = np.where(valid, (ss_t2 * df_den) / sse_full, np.nan)
        f_s0_batch = np.where(valid, (ss_s0 * df_den) / sse_full, np.nan)

        # Store results (mask out invalid entries)
        ss_t2 = np.where(valid, ss_t2, np.nan)
        ss_s0 = np.where(valid, ss_s0, np.nan)

        f_t2star[voxel_indices] = f_t2_batch
        f_s0[voxel_indices] = f_s0_batch
        ss_t2_out[voxel_indices] = ss_t2
        ss_s0_out[voxel_indices] = ss_s0

    return f_t2star, f_s0, ss_t2_out, ss_s0_out
