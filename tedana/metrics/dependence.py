"""Metrics evaluating component TE-dependence or -independence."""

import logging
import typing

import nibabel as nb
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm, trange

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
        signal_minus_noise_t[i_comp], signal_minus_noise_p[i_comp] = stats.ttest_ind(
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


def compute_te_variance(
    echowise_pes: np.ndarray,
    tes: np.ndarray,
    s0_hat: np.ndarray,
    t2s_hat: np.ndarray,
    adaptive_mask: np.ndarray,
    spatial_weights: np.ndarray | None = None,
    t2s_min: float = 5.0,
    t2s_max: float = 500.0,
):
    """Compute descriptive variance fractions (kappa_star, rho_star) for components.

    Decompose the echo-wise variance of each ICA component into S0-driven and
    T2*-driven contributions using a linearized monoexponential signal model.
    Returns descriptive metrics (kappa_star, rho_star) indicating what fraction
    of each component's explainable variance is attributable to T2* versus S0
    effects.

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
    spatial_weights : (n_voxels, n_comps) array, optional
        Voxel weights for aggregation (e.g., component loadings squared).
        If None, equal weighting is used.
    t2s_min : float, optional
        Minimum valid T2* value in milliseconds. Voxels with T2* below this
        are excluded as unrealistic. Default is 5.0 ms.
    t2s_max : float, optional
        Maximum valid T2* value in milliseconds. Voxels with T2* above this
        are excluded as unrealistic. Default is 500.0 ms.

    Returns
    -------
    kappa_star : (n_comps,) array
        Component-level T2* variance fraction (descriptive). Values range
        from 0 to 1, where higher values indicate more T2*-driven variance.
    rho_star : (n_comps,) array
        Component-level S0 variance fraction (descriptive). Values range
        from 0 to 1, where higher values indicate more S0-driven variance.
        Note: kappa_star + rho_star = 1 by construction.
    f_t2star : (n_comps,) array
        Weighted average partial F-statistics for T2* term (descriptive).
    f_s0 : (n_comps,) array
        Weighted average partial F-statistics for S0 term (descriptive).

    Notes
    -----
    **Method overview**

    For each voxel and component, this function:

    1. Constructs voxel-specific basis functions from the linearized
       monoexponential signal model S(TE) = S0·exp(-TE/T2*):

       - φ_S0 = ∂S/∂S0 = exp(-TE/T2*)
       - φ_T2* = ∂S/∂T2* = S0·exp(-TE/T2*)·TE/T2*²

    2. Fits three nested linear models to the echo-wise component betas:

       - Full model: Y ~ φ_S0 + φ_T2*
       - S0-only model: Y ~ φ_S0
       - T2*-only model: Y ~ φ_T2*

    3. Computes Type III (unique) sums of squares:

       - SS_T2* = SSE(S0-only) - SSE(Full)
       - SS_S0 = SSE(T2*-only) - SSE(Full)

    4. Aggregates across voxels using spatial weights and computes:

       - kappa_star = Σ(w·SS_T2*) / Σ(w·(SS_T2* + SS_S0))
       - rho_star = Σ(w·SS_S0) / Σ(w·(SS_T2* + SS_S0))

    **Strengths**

    - *Physically motivated*: Basis functions derived from the actual signal
      model, using voxel-specific T2* values for accurate basis shapes.

    - *Principled aggregation*: Uses sum-of-sums rather than averaging ratios,
      preserving variance decomposition interpretation at component level.

    - *Type III SS*: Properly partitions variance when basis functions are
      non-orthogonal (which they typically are).

    - *Fast computation*: Vectorized implementation, no permutations needed.

    - *Interpretable output*: kappa_star and rho_star sum to 1 and directly
      indicate the relative T2*/S0 contribution.

    **Weaknesses and limitations**

    - *Descriptive only*: kappa_star and rho_star are variance fractions,
      not p-values. They do not provide statistical inference about whether
      observed values are significantly different from chance.

    - *No spatial specificity test*: Does not test whether T2*-dependence
      specifically matches *local* T2* sensitivity (the BOLD signature).
      A component could have high kappa_star due to global T2*-like structure
      without being spatially specific.

    - *Model dependence*: Results depend on the linearization approximation
      δS ≈ (∂S/∂S0)δS0 + (∂S/∂T2*)δT2*. If fluctuations are large or the
      monoexponential model is poor, interpretation weakens.

    - *Map quality dependence*: Accuracy of kappa_star/rho_star depends on
      the quality of T2* and S0 maps. Noisy maps lead to noisier estimates.

    - *F-statistics are averaged*: While kappa_star/rho_star use principled
      sum-of-sums aggregation, f_t2star/f_s0 are still weighted averages of
      voxel-wise F-statistics, which is statistically suboptimal.

    **Interpretation guidelines**

    - kappa_star ≈ 1: Component variance is predominantly T2*-driven
      (consistent with BOLD-like signal, but not proof of BOLD)

    - kappa_star ≈ 0: Component variance is predominantly S0-driven
      (consistent with S0-driven noise like respiration/cardiac)

    - kappa_star ≈ 0.5: Mixed T2*/S0 contributions

    **Important**: High kappa_star indicates T2*-like echo-wise structure
    but does NOT prove the component is neuronal BOLD signal. Non-neuronal
    effects (e.g., motion-correlated susceptibility) can also produce
    T2*-like structure.

    **Comparison with compute_te_variance_permutation**

    +---------------------------+----------------------------+----------------------------+
    | Aspect                    | compute_te_variance| compute_te_variance_permutation|
    +===========================+============================+============================+
    | Output type               | Descriptive (variance      | Inferential (p-values)     |
    |                           | fractions)                 |                            |
    +---------------------------+----------------------------+----------------------------+
    | Statistical inference     | No                         | Yes (permutation-based)    |
    +---------------------------+----------------------------+----------------------------+
    | Tests spatial specificity | No                         | Yes                        |
    +---------------------------+----------------------------+----------------------------+
    | Computation time          | Fast (no permutations)     | Slower (n_perm iterations) |
    +---------------------------+----------------------------+----------------------------+
    | Use case                  | Quick descriptive summary  | Formal hypothesis testing  |
    +---------------------------+----------------------------+----------------------------+

    **Recommended workflow**:

    1. Use ``compute_te_variance`` for fast descriptive metrics
       (kappa_star, rho_star) to get a quick overview of component character.

    2. Use ``compute_te_variance_permutation`` for formal statistical testing
       (p_t2, p_s0) when you need to assess significance of spatial specificity.

    3. Combine both: Use kappa_star to characterize *how much* variance is
       T2*-driven, and p_t2 to assess *whether* that T2*-dependence is
       spatially specific (consistent with BOLD).

    See Also
    --------
    compute_te_variance_permutation : Permutation test for spatial specificity
        of TE-dependence. Provides p-values for statistical inference.
    calculate_dependence_metrics : Traditional kappa/rho calculation using
        simpler basis functions (µ and TE×µ) and averaged F-statistics.
    """
    if not (
        echowise_pes.shape[0] == s0_hat.shape[0] == t2s_hat.shape[0] == adaptive_mask.shape[0]
    ):
        raise ValueError(
            "echowise_pes, s0_hat, t2s_hat, and adaptive_mask must have the same number of voxels"
        )
    if not s0_hat.ndim == t2s_hat.ndim:
        raise ValueError("s0_hat and t2s_hat must have the same number of dimensions")
    if echowise_pes.shape[1] != len(tes):
        raise ValueError("echowise_pes and tes must have the same number of echoes")

    n_voxels, n_echos, n_comps = echowise_pes.shape
    tes = np.asarray(tes, dtype=np.float64)

    # Handle spatial weights
    if spatial_weights is None:
        spatial_weights = np.ones((n_voxels, n_comps))
    elif spatial_weights.shape != (n_voxels, n_comps):
        raise ValueError(
            f"spatial_weights shape {spatial_weights.shape} must match "
            f"(n_voxels, n_comps) = ({n_voxels}, {n_comps})"
        )

    # Detect whether baseline estimates are voxel-wise or voxel+volume-wise
    volume_wise = s0_hat.ndim == 2

    # Initialize voxel-wise output arrays (will be aggregated at the end)
    f_t2star_vox = np.full((n_voxels, n_comps), np.nan)
    f_s0_vox = np.full((n_voxels, n_comps), np.nan)
    ss_t2_vox = np.full((n_voxels, n_comps), np.nan)
    ss_s0_vox = np.full((n_voxels, n_comps), np.nan)

    # Process voxels grouped by number of valid echoes for efficiency
    unique_n_echoes = np.unique(adaptive_mask[adaptive_mask >= 3])

    for n_e in unique_n_echoes:
        df_den = n_e - 2
        if df_den <= 0:
            continue

        # Get voxel indices with this number of echoes
        voxel_mask = adaptive_mask == n_e
        voxel_indices = np.where(voxel_mask)[0]

        # Filter for valid baseline estimates (positive S0 and T2* within bounds)
        if volume_wise:
            valid_baseline = np.all(
                (s0_hat[voxel_indices] > 0)
                & (t2s_hat[voxel_indices] >= t2s_min)
                & (t2s_hat[voxel_indices] <= t2s_max),
                axis=1,
            )
        else:
            valid_baseline = (
                (s0_hat[voxel_indices] > 0)
                & (t2s_hat[voxel_indices] >= t2s_min)
                & (t2s_hat[voxel_indices] <= t2s_max)
            )

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
        yty = np.sum(Y**2, axis=1)

        # S0-only model: SSE_s0 = y'y - (phi_s0'y)^2 / (phi_s0'phi_s0)
        phi_s0_norm_sq = np.sum(phi_s0**2, axis=1, keepdims=True)  # (n_vox_batch, 1)
        phi_s0_dot_y = np.einsum("ve,vec->vc", phi_s0, Y)  # (n_vox_batch, n_comps)
        sse_s0 = yty - (phi_s0_dot_y**2) / phi_s0_norm_sq

        # T2*-only model: SSE_t2 = y'y - (phi_t2'y)^2 / (phi_t2'phi_t2)
        phi_t2_norm_sq = np.sum(phi_t2**2, axis=1, keepdims=True)  # (n_vox_batch, 1)
        phi_t2_dot_y = np.einsum("ve,vec->vc", phi_t2, Y)  # (n_vox_batch, n_comps)
        sse_t2 = yty - (phi_t2_dot_y**2) / phi_t2_norm_sq

        # Full model: need to solve 2x2 normal equations per voxel
        # X = [phi_s0, phi_t2], X'X is 2x2, X'Y is 2 x n_comps
        # SSE_full = y'y - y'X(X'X)^{-1}X'y

        # Compute X'X elements: (n_vox_batch,) each
        xtx_00 = phi_s0_norm_sq.squeeze()  # phi_s0' phi_s0
        xtx_11 = phi_t2_norm_sq.squeeze()  # phi_t2' phi_t2
        xtx_01 = np.sum(phi_s0 * phi_t2, axis=1)  # phi_s0' phi_t2

        # Determinant of X'X: (n_vox_batch,)
        det = xtx_00 * xtx_11 - xtx_01**2

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
            inv_00[:, None] * xty_0**2
            + inv_11[:, None] * xty_1**2
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

        f_t2star_vox[voxel_indices] = f_t2_batch
        f_s0_vox[voxel_indices] = f_s0_batch
        ss_t2_vox[voxel_indices] = ss_t2
        ss_s0_vox[voxel_indices] = ss_s0

    # Aggregate voxel-wise results to component-level using spatial weights
    # Use sum-of-sums aggregation for variance fractions (preserves decomposition)
    kappa_star = np.full(n_comps, np.nan)
    rho_star = np.full(n_comps, np.nan)
    f_t2star = np.full(n_comps, np.nan)
    f_s0 = np.full(n_comps, np.nan)

    for i_comp in range(n_comps):
        # Identify valid voxels for this component
        valid_mask = (
            ~np.isnan(ss_t2_vox[:, i_comp])
            & ~np.isnan(ss_s0_vox[:, i_comp])
            & ~np.isnan(f_t2star_vox[:, i_comp])
            & ~np.isnan(f_s0_vox[:, i_comp])
        )

        if not np.any(valid_mask):
            continue

        weights_comp = spatial_weights[valid_mask, i_comp]

        # Principled aggregation: sum-of-sums rather than average-of-ratios
        # Weight the voxel-wise SS values before summing
        weighted_ss_t2 = np.sum(ss_t2_vox[valid_mask, i_comp] * weights_comp)
        weighted_ss_s0 = np.sum(ss_s0_vox[valid_mask, i_comp] * weights_comp)
        total_weighted_ss = weighted_ss_t2 + weighted_ss_s0

        if total_weighted_ss > 0:
            kappa_star[i_comp] = weighted_ss_t2 / total_weighted_ss
            rho_star[i_comp] = weighted_ss_s0 / total_weighted_ss
        else:
            kappa_star[i_comp] = np.nan
            rho_star[i_comp] = np.nan

        # Weighted average for F-statistics
        f_t2star[i_comp] = np.average(f_t2star_vox[valid_mask, i_comp], weights=weights_comp)
        f_s0[i_comp] = np.average(f_s0_vox[valid_mask, i_comp], weights=weights_comp)

    return kappa_star, rho_star, f_t2star, f_s0


def compute_te_variance_permutation(
    echowise_pes: np.ndarray,
    tes: np.ndarray,
    s0_hat: np.ndarray,
    t2s_hat: np.ndarray,
    adaptive_mask: np.ndarray,
    spatial_weights: np.ndarray | None = None,
    n_perm: int = 1000,
    n_threads: int = 1,
    seed: int | None = None,
    t2s_min: float = 5.0,
    t2s_max: float = 500.0,
):
    """Permutation test for spatial specificity of component TE-dependence.

    Test whether each ICA component's echo-wise structure is specifically
    aligned with voxel-local T2* or S0 sensitivity, under a linearized
    monoexponential signal model. This provides valid permutation p-values
    for testing spatial specificity without parametric assumptions about
    the distribution of test statistics.

    Parameters
    ----------
    echowise_pes : (n_voxels, n_echos, n_comps) array
        Echo-wise *unstandardized* ICA component parameter estimates.
    tes : (n_echos,) array
        Echo times in milliseconds.
    s0_hat : (n_voxels,) or (n_voxels, n_vols) array
        Baseline S0 estimates. If 2D, averaged over volumes.
    t2s_hat : (n_voxels,) or (n_voxels, n_vols) array
        Baseline T2* estimates. If 2D, averaged over volumes.
    adaptive_mask : (n_voxels,) array of int
        Number of valid echoes per voxel.
    spatial_weights : (n_voxels, n_comps) array, optional
        Voxel weights for aggregation (e.g., component loadings).
        If None, equal weighting is used.
    n_perm : int, optional
        Number of permutations for null distribution. Default is 1000.
    t2s_min : float, optional
        Minimum valid T2* value in milliseconds. Voxels with T2* below this
        are excluded as unrealistic. Default is 5.0 ms.
    t2s_max : float, optional
        Maximum valid T2* value in milliseconds. Voxels with T2* above this
        are excluded as unrealistic. Default is 500.0 ms.
    n_threads : int, optional
        Number of parallel jobs. Default is 1 (sequential).
        Set to -1 to use all available cores.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    kappa_star : (n_comps,) array
        Observed T2* variance fraction for each component (descriptive only).
    rho_star : (n_comps,) array
        Observed S0 variance fraction for each component (descriptive only).
    p_t2 : (n_comps,) array
        Permutation p-values testing spatial specificity of T2*-alignment.
        Low values indicate the component's echo-wise structure specifically
        matches local T2* sensitivity (consistent with BOLD-like signal).
    p_s0 : (n_comps,) array
        Permutation p-values testing spatial specificity of S0-alignment.
        Low values indicate the component's echo-wise structure specifically
        matches local S0 sensitivity (consistent with S0-driven fluctuations).

    Notes
    -----
    **Important: Inference vs. Description**

    - ``kappa_star`` and ``rho_star`` are *descriptive* variance fractions
      and are not used for statistical inference.
    - Statistical inference is based solely on the permutation p-values
      (``p_t2``, ``p_s0``), which are empirical tail probabilities from
      the null distribution.

    **Null hypothesis**

    The component's echo-wise parameter estimates show echo-wise structure,
    but this structure is not specifically aligned with voxel-local T2* or
    S0 sensitivity. Under the null, permuting which voxel's basis functions
    are used for fitting should not systematically change model fit.

    **Permutation scheme**

    This test uses *independent* permutations for T2* and S0 to isolate
    spatial specificity of each component:

    - For ``p_t2``: Permute only φ_T2* while keeping φ_S0 fixed (local).
      This tests whether T2* explains more unique variance with its local
      basis than with a randomly assigned basis.

    - For ``p_s0``: Permute only φ_S0 while keeping φ_T2* fixed (local).
      This tests whether S0 explains more unique variance with its local
      basis than with a randomly assigned basis.

    Within groups of voxels sharing the same number of valid echoes (n_e),
    shuffle the assignment of the target basis function to voxels while
    keeping echo-wise PEs and the other basis fixed. This preserves:

    - Total variance structure of each component
    - Echo-wise correlation structure within voxels
    - Marginal distribution of basis function shapes
    - The contribution from the non-permuted basis (isolation of effects)

    While breaking:

    - Spatial alignment between the target basis and voxel-local sensitivity

    Note that permutations are performed within echo-availability strata
    only. The null is therefore: "alignment no better than random assignment
    within echo-availability groups."

    **Test statistics**

    - ``ss_t2``: Unique variance explained by T2* (Type III sum of squares).
      This is the variance that T2* explains *beyond* what S0 explains.
      Higher values indicate better spatial alignment with local T2*.

    - ``ss_s0``: Unique variance explained by S0 (Type III sum of squares).
      This is the variance that S0 explains *beyond* what T2* explains.
      Higher values indicate better spatial alignment with local S0.

    P-values are the proportion of permutations achieving unique variance
    as high or higher than observed (one-tailed test, higher is better).

    **Interpretation**

    - Low ``p_t2``: Echo-wise pattern is spatially aligned with local T2*
      sensitivity → consistent with BOLD-like signal
    - Low ``p_s0``: Echo-wise pattern is spatially aligned with local S0
      sensitivity → consistent with S0-driven fluctuations
    - Both low: Mixed signal with both T2* and S0 spatial specificity
    - Both high: Non-specific echo-wise structure

    **Limitations and caveats**

    1. *Model dependence*: This test evaluates alignment conditional on the
       linearized monoexponential model δS ≈ (∂S/∂S0)δS0 + (∂S/∂T2*)δT2*.
       If the linearization is poor (large fluctuations, nonlinear effects),
       the interpretation weakens.

    2. *Map quality dependence*: The test assumes the provided s0_hat and
       t2s_hat are meaningful voxelwise quantities. If T2*/S0 maps are noisy
       or biased, the null distribution becomes wider and power decreases.
       P-values are conditional on map quality.

    3. *Physiological interpretation*: This test identifies components whose
       echo-wise structure is consistent with voxel-local T2* sensitivity.
       This is a *necessary but not sufficient* condition for neuronal BOLD
       signal. A component could show T2*-alignment due to non-neuronal
       effects (e.g., motion-correlated susceptibility artifacts).

    **Why this approach**

    Traditional kappa/rho metrics and parametric F-tests for component-level
    TE-dependence suffer from invalid aggregation (averaging ratios) or
    incorrect degrees of freedom. This permutation test avoids parametric
    assumptions entirely and directly tests the spatial specificity that
    distinguishes BOLD from other echo-wise structure.
    """
    if not (
        echowise_pes.shape[0] == s0_hat.shape[0] == t2s_hat.shape[0] == adaptive_mask.shape[0]
    ):
        raise ValueError(
            "echowise_pes, s0_hat, t2s_hat, and adaptive_mask must have the same number of voxels"
        )
    if echowise_pes.shape[1] != len(tes):
        raise ValueError("echowise_pes and tes must have the same number of echoes")

    rng = np.random.default_rng(seed)
    n_voxels, _, n_comps = echowise_pes.shape
    tes = np.asarray(tes, dtype=np.float64)

    if spatial_weights is None:
        spatial_weights = np.ones((n_voxels, n_comps))

    # Handle voxel+volume-wise estimates by averaging over volumes
    if s0_hat.ndim == 2:
        s0_hat = np.mean(s0_hat, axis=1)
    if t2s_hat.ndim == 2:
        t2s_hat = np.mean(t2s_hat, axis=1)

    # Identify valid voxels (positive S0 and T2* within realistic bounds)
    valid_voxel = (
        (adaptive_mask >= 3)
        & (s0_hat > 0)
        & (t2s_hat >= t2s_min)
        & (t2s_hat <= t2s_max)
    )

    # Precompute everything that doesn't change with permutation, grouped by n_e
    # This is the key optimization: we only recompute dot products during permutation
    precomputed = {}

    for n_e in np.unique(adaptive_mask[valid_voxel]):
        mask_ne = valid_voxel & (adaptive_mask == n_e)
        vox_idx = np.where(mask_ne)[0]
        n_vox_ne = len(vox_idx)

        tes_v = tes[:n_e]
        s0_v = s0_hat[vox_idx]
        t2s_v = t2s_hat[vox_idx]

        # Basis functions: φ_S0 and φ_T2* (n_vox_ne, n_e)
        phi_s0 = np.exp(-tes_v[None, :] / t2s_v[:, None])
        phi_t2 = (
            s0_v[:, None]
            * np.exp(-tes_v[None, :] / t2s_v[:, None])
            * tes_v[None, :]
            / (t2s_v[:, None] ** 2)
        )

        # Data (fixed, never permuted): Y and y'y
        Y = echowise_pes[vox_idx, :n_e, :]  # (n_vox_ne, n_e, n_comps)
        yty = np.sum(Y**2, axis=1)  # (n_vox_ne, n_comps)
        weights = spatial_weights[vox_idx, :]  # (n_vox_ne, n_comps)

        # Basis function norms (permutation just reindexes these)
        phi_s0_norm_sq = np.sum(phi_s0**2, axis=1)  # (n_vox_ne,)
        phi_t2_norm_sq = np.sum(phi_t2**2, axis=1)  # (n_vox_ne,)

        precomputed[n_e] = {
            "n_vox": n_vox_ne,
            "phi_s0": phi_s0,
            "phi_t2": phi_t2,
            "Y": Y,
            "yty": yty,
            "weights": weights,
            "phi_s0_norm_sq": phi_s0_norm_sq,
            "phi_t2_norm_sq": phi_t2_norm_sq,
        }

    def compute_ss_t2(perm_t2_indices=None):
        """Compute unique T2* variance (ss_t2) with optional phi_t2 permutation.

        Parameters
        ----------
        perm_t2_indices : dict or None
            If provided, dict mapping n_e -> permutation indices for phi_t2.
            phi_s0 remains unpermuted (local).

        Returns
        -------
        ss_t2_total : (n_comps,) array
            Weighted sum of unique variance attributable to T2*.
        """
        ss_t2_total = np.zeros(n_comps)

        for n_e, data in precomputed.items():
            # phi_s0 always uses local (unpermuted) basis
            phi_s0 = data["phi_s0"]
            phi_s0_norm_sq = data["phi_s0_norm_sq"]

            # phi_t2 may be permuted
            phi_t2 = data["phi_t2"]
            phi_t2_norm_sq = data["phi_t2_norm_sq"]
            if perm_t2_indices is not None:
                perm = perm_t2_indices[n_e]
                phi_t2 = phi_t2[perm]
                phi_t2_norm_sq = phi_t2_norm_sq[perm]

            Y = data["Y"]
            yty = data["yty"]
            weights = data["weights"]

            # Compute dot products
            phi_s0_dot_y = np.einsum("ve,vec->vc", phi_s0, Y, optimize=True)
            phi_t2_dot_y = np.einsum("ve,vec->vc", phi_t2, Y, optimize=True)

            # SSE for S0-only model (local phi_s0)
            sse_s0 = yty - (phi_s0_dot_y**2) / phi_s0_norm_sq[:, None]

            # For full model, need to recompute Gram matrix inverse with mixed bases
            phi_s0_dot_t2 = np.sum(phi_s0 * phi_t2, axis=1)
            det = phi_s0_norm_sq * phi_t2_norm_sq - phi_s0_dot_t2**2
            # Use relative threshold consistent with compute_te_variance
            valid_det = det > 1e-10 * phi_s0_norm_sq * phi_t2_norm_sq
            det_safe = np.where(valid_det, det, 1.0)

            inv_00 = np.where(valid_det, phi_t2_norm_sq / det_safe, 0)
            inv_11 = np.where(valid_det, phi_s0_norm_sq / det_safe, 0)
            inv_01 = np.where(valid_det, -phi_s0_dot_t2 / det_safe, 0)

            # SSE for full model
            quadform = (
                inv_00[:, None] * phi_s0_dot_y**2
                + inv_11[:, None] * phi_t2_dot_y**2
                + 2 * inv_01[:, None] * phi_s0_dot_y * phi_t2_dot_y
            )
            sse_full = yty - quadform

            # ss_t2 = unique T2* variance = sse_s0 - sse_full
            ss_t2 = sse_s0 - sse_full

            # Accumulate
            valid = valid_det[:, None] & (sse_full > 0)
            ss_t2_total += np.sum(np.where(valid, ss_t2 * weights, 0), axis=0)

        return ss_t2_total

    def compute_ss_s0(perm_s0_indices=None):
        """Compute unique S0 variance (ss_s0) with optional phi_s0 permutation.

        Parameters
        ----------
        perm_s0_indices : dict or None
            If provided, dict mapping n_e -> permutation indices for phi_s0.
            phi_t2 remains unpermuted (local).

        Returns
        -------
        ss_s0_total : (n_comps,) array
            Weighted sum of unique variance attributable to S0.
        """
        ss_s0_total = np.zeros(n_comps)

        for n_e, data in precomputed.items():
            # phi_t2 always uses local (unpermuted) basis
            phi_t2 = data["phi_t2"]
            phi_t2_norm_sq = data["phi_t2_norm_sq"]

            # phi_s0 may be permuted
            phi_s0 = data["phi_s0"]
            phi_s0_norm_sq = data["phi_s0_norm_sq"]
            if perm_s0_indices is not None:
                perm = perm_s0_indices[n_e]
                phi_s0 = phi_s0[perm]
                phi_s0_norm_sq = phi_s0_norm_sq[perm]

            Y = data["Y"]
            yty = data["yty"]
            weights = data["weights"]

            # Compute dot products
            phi_s0_dot_y = np.einsum("ve,vec->vc", phi_s0, Y, optimize=True)
            phi_t2_dot_y = np.einsum("ve,vec->vc", phi_t2, Y, optimize=True)

            # SSE for T2*-only model (local phi_t2)
            sse_t2 = yty - (phi_t2_dot_y**2) / phi_t2_norm_sq[:, None]

            # For full model, need to recompute Gram matrix inverse with mixed bases
            phi_s0_dot_t2 = np.sum(phi_s0 * phi_t2, axis=1)
            det = phi_s0_norm_sq * phi_t2_norm_sq - phi_s0_dot_t2**2
            # Use relative threshold consistent with compute_te_variance
            valid_det = det > 1e-10 * phi_s0_norm_sq * phi_t2_norm_sq
            det_safe = np.where(valid_det, det, 1.0)

            inv_00 = np.where(valid_det, phi_t2_norm_sq / det_safe, 0)
            inv_11 = np.where(valid_det, phi_s0_norm_sq / det_safe, 0)
            inv_01 = np.where(valid_det, -phi_s0_dot_t2 / det_safe, 0)

            # SSE for full model
            quadform = (
                inv_00[:, None] * phi_s0_dot_y**2
                + inv_11[:, None] * phi_t2_dot_y**2
                + 2 * inv_01[:, None] * phi_s0_dot_y * phi_t2_dot_y
            )
            sse_full = yty - quadform

            # ss_s0 = unique S0 variance = sse_t2 - sse_full
            ss_s0 = sse_t2 - sse_full

            # Accumulate
            valid = valid_det[:, None] & (sse_full > 0)
            ss_s0_total += np.sum(np.where(valid, ss_s0 * weights, 0), axis=0)

        return ss_s0_total

    # Compute observed statistics (no permutation)
    obs_ss_t2 = compute_ss_t2(perm_t2_indices=None)
    obs_ss_s0 = compute_ss_s0(perm_s0_indices=None)

    # Compute kappa_star and rho_star from Type III SS
    total_ss = obs_ss_t2 + obs_ss_s0
    kappa_star = np.where(total_ss > 0, obs_ss_t2 / total_ss, np.nan)
    rho_star = np.where(total_ss > 0, obs_ss_s0 / total_ss, np.nan)

    # Debug: show observed SS
    LGR.debug("Observed unique variance (Type III SS):")
    for c in range(n_comps):
        LGR.debug(
            f"  Component {c}: ss_t2={obs_ss_t2[c]:.4f}, ss_s0={obs_ss_s0[c]:.4f}, "
            f"kappa_star={kappa_star[c]:.4f}, rho_star={rho_star[c]:.4f}"
        )

    # Pre-generate permutation indices (separate for T2* and S0 tests)
    perm_t2_indices_all = []
    perm_s0_indices_all = []
    for _ in range(n_perm):
        perm_t2_indices_all.append(
            {n_e: rng.permutation(data["n_vox"]) for n_e, data in precomputed.items()}
        )
        perm_s0_indices_all.append(
            {n_e: rng.permutation(data["n_vox"]) for n_e, data in precomputed.items()}
        )

    # Build null distributions with optional parallelization
    LGR.info(f"Running permutation test: {n_perm} permutations for T2* and S0 separately")

    if n_threads == 1:
        # Sequential execution with progress bar
        null_ss_t2_list = []
        null_ss_s0_list = []
        for i_perm in trange(n_perm, desc="Permutation test (T2*)"):
            null_ss_t2_list.append(compute_ss_t2(perm_t2_indices=perm_t2_indices_all[i_perm]))
        for i_perm in trange(n_perm, desc="Permutation test (S0)"):
            null_ss_s0_list.append(compute_ss_s0(perm_s0_indices=perm_s0_indices_all[i_perm]))
    else:
        # Parallel execution
        LGR.info(f"Using {n_threads} parallel jobs")
        null_ss_t2_list = Parallel(n_jobs=n_threads)(
            delayed(compute_ss_t2)(perm_t2_indices=perm_t2_indices_all[i])
            for i in tqdm(range(n_perm), desc="Permutation test (T2*)")
        )
        null_ss_s0_list = Parallel(n_jobs=n_threads)(
            delayed(compute_ss_s0)(perm_s0_indices=perm_s0_indices_all[i])
            for i in tqdm(range(n_perm), desc="Permutation test (S0)")
        )

    null_ss_t2 = np.array(null_ss_t2_list)
    null_ss_s0 = np.array(null_ss_s0_list)

    # Debug output: print observed vs null distributions
    LGR.debug("Permutation test diagnostics:")
    for c in range(n_comps):
        LGR.debug(f"  Component {c}:")
        LGR.debug(f"    obs_ss_t2 = {obs_ss_t2[c]:.6f}")
        LGR.debug(
            f"    null_ss_t2: mean={null_ss_t2[:, c].mean():.6f}, "
            f"std={null_ss_t2[:, c].std():.6f}, "
            f"min={null_ss_t2[:, c].min():.6f}, "
            f"max={null_ss_t2[:, c].max():.6f}"
        )
        LGR.debug(f"    obs_ss_s0 = {obs_ss_s0[c]:.6f}")
        LGR.debug(
            f"    null_ss_s0: mean={null_ss_s0[:, c].mean():.6f}, "
            f"std={null_ss_s0[:, c].std():.6f}, "
            f"min={null_ss_s0[:, c].min():.6f}, "
            f"max={null_ss_s0[:, c].max():.6f}"
        )

    # Compute p-values based on unique variance (higher is better spatial specificity)
    # p_t2: proportion of null permutations with ss_t2 >= observed
    # Low p_t2 means T2* explains significantly MORE unique variance with local basis
    # → component's T2* contribution is spatially specific (BOLD-like)
    p_t2 = (np.sum(null_ss_t2 >= obs_ss_t2, axis=0) + 1) / (n_perm + 1)

    # p_s0: proportion of null permutations with ss_s0 >= observed
    # Low p_s0 means S0 explains significantly MORE unique variance with local basis
    # → component's S0 contribution is spatially specific (non-BOLD)
    p_s0 = (np.sum(null_ss_s0 >= obs_ss_s0, axis=0) + 1) / (n_perm + 1)

    return kappa_star, rho_star, p_t2, p_s0
