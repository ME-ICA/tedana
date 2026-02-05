"""Metrics evaluating component TE-dependence or -independence."""

import logging
import typing

import nibabel as nb
import numpy as np
from nilearn import image, masking
from scipy import stats

from tedana import utils
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
    data_optcom : (Mb x T) array_like
        Optimally combined data, already masked, where `Mb` is samples in base mask,
        and `T` is time.
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
    data_optcom : (Mb x T) array_like
        Optimally combined data, already masked, where `Mb` is samples in base mask,
        and `T` is time.
    optcom_betas : (Mb x C) array_like
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
    data_cat : (Mb x E x T) array_like
        Multi-echo data, already masked, where `Mb` is samples in base mask, `E` is echos,
        and `T` is time.
    mixing : (T x C) array_like
        Mixing matrix
    adaptive_mask : (Mb) array_like
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
    f_t2_maps, f_s0_maps, pred_t2_maps, pred_s0_maps : (Mb x C) array_like
        Pseudo-F-statistic maps for TE-dependence and -independence models,
        respectively.
    """
    assert data_cat.shape[0] == adaptive_mask.shape[0]
    assert data_cat.shape[1] == tes.shape[0]
    assert data_cat.shape[2] == mixing.shape[0]

    n_voxels, n_echos, _ = data_cat.shape
    n_components = mixing.shape[1]
    me_betas = np.zeros([n_voxels, n_echos, n_components])
    for i_echo in range(n_echos):
        me_betas[:, i_echo, :] = get_coeffs(data_cat[:, i_echo, :], mixing, add_const=True)

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
    mask_img: nb.Nifti1Image,
    proportion_threshold: float = None,
    value_threshold: float = None,
    csize: typing.Union[int, None] = None,
) -> np.ndarray:
    """Perform cluster-extent thresholding.

    Parameters
    ----------
    maps : (M_s x C) array_like
        Statistical maps to be thresholded. M_s is the number of voxels in the strict mask.
    mask_img : img_like
        Strict mask image to convert to niimgs with.
    proportion_threshold : :obj:`float`
        Proportion threshold to apply to maps. Values between 0 and 100.
    value_threshold : float, optional
        Value threshold to apply to maps. Default is None.
    csize : :obj:`int` or :obj:`None`, optional
        Minimum cluster size. If None, standard thresholding (non-cluster-extent) will be done.
        Default is None.

    Returns
    -------
    maps_thresh : (M_s x C) array_like
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

    # Cluster-extent threshold and binarize F-maps
    img = masking.unmask(maps.T, mask_img)
    for i_comp in range(n_components):
        comp_img = image.index_img(img, i_comp)
        thresh_arr = utils.threshold_map(
            comp_img,
            min_cluster_size=csize,
            threshold=value_threshold[i_comp],
            binarize=True,
            sided="bi",
        )
        thresh_img = nb.Nifti1Image(thresh_arr, mask_img.affine, mask_img.header)
        maps_thresh[:, i_comp] = masking.apply_mask(thresh_img, mask_img)
    return maps_thresh


def threshold_to_match(
    *,
    maps: np.ndarray,
    n_sig_voxels: np.ndarray,
    mask_img: nb.Nifti1Image,
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
    mask_img : img_like
        Strict mask image to convert to niimgs with.
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
    if csize is None:
        csize = np.max([int(n_voxels * 0.0005) + 5, 20])
    else:
        csize = int(csize)

    clmaps = np.zeros([n_voxels, n_components], dtype=bool)
    # Avoid repeated apply_mask calls: `utils.threshold_map` returns a 3D array
    # in the same space as `mask_img`, so we can index directly into it.
    mask_bool = np.asanyarray(mask_img.dataobj).astype(bool)
    for i_comp in range(n_components):
        target = int(n_sig_voxels[i_comp])
        if target <= 0:
            clmaps[:, i_comp] = False
            continue
        if target >= n_voxels:
            clmaps[:, i_comp] = True
            continue

        # Initial cluster-defining threshold is defined based on the number
        # of significant voxels from the F-statistic maps. This threshold
        # will be relaxed until the number of significant voxels from both
        # maps is roughly equal.
        vals = np.abs(maps[:, i_comp])
        ccimg = masking.unmask(vals, mask_img)

        # Cluster-extent thresholding makes the voxel count depend on spatial structure.
        # Use a small, bounded bisection to find a threshold that yields approximately
        # `target` significant voxels after cluster-extent thresholding.
        lo = 0.0
        hi = float(np.max(vals))
        best_clmap = None
        best_diff = np.inf

        # 12 iterations gives ~1/4096 relative resolution on [lo, hi]
        for _ in range(12):
            thr = (lo + hi) / 2.0
            thresh_arr = utils.threshold_map(
                ccimg,
                min_cluster_size=csize,
                threshold=thr,
                binarize=True,
                sided="bi",
            )
            clmap = thresh_arr[mask_bool] != 0
            n_found = int(clmap.sum())
            diff = abs(target - n_found)
            if diff < best_diff:
                best_diff = diff
                best_clmap = clmap
                if best_diff == 0:
                    break

            # Lower threshold => more suprathreshold voxels
            if n_found > target:
                lo = thr
            else:
                hi = thr

        if best_clmap is None:
            best_clmap = np.zeros(n_voxels, dtype=bool)
        clmaps[:, i_comp] = best_clmap
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
