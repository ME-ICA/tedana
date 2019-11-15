"""
Fit models.
"""
import logging
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats

from tedana import io, utils
from tedana.stats import getfbounds, computefeats2, get_coeffs, t_to_z


LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def calculate_weights(data_optcom, mixing_z):
    """
    Calculate standardized parameter estimates between data and mixing matrix.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined data
    mixing_z : (T x C) array_like
        Z-scored mixing matrix

    Returns
    -------
    weights : (S x C) array_like
        Standardized parameter estimates for optimally combined data against
        the mixing matrix.
    """
    # compute un-normalized weight dataset (features)
    weights = computefeats2(data_optcom, mixing_z, normalize=False)
    return weights


def calculate_betas(data_optcom, mixing):
    """
    Calculate unstandardized parameter estimates between data and mixing
    matrix.

    Parameters
    ----------
    data_optcom
    mixing

    Returns
    -------
    betas
        Unstandardized parameter estimates
    """
    assert data_optcom.shape[1] == mixing.shape[0]
    # demean optimal combination
    data_optcom_dm = data_optcom - data_optcom.mean(axis=-1, keepdims=True)
    # compute PSC dataset - shouldn't have to refit data
    betas = get_coeffs(data_optcom_dm, mixing)
    return betas


def calculate_psc(data_optcom, optcom_betas):
    """
    Calculate percent signal change maps for components against optimally
    combined data.

    Parameters
    ----------
    data_optcom
    optcom_betas

    Returns
    -------
    PSC
        Component-wise percent signal change maps.
    """
    assert data_optcom.shape[1] == optcom_betas.shape[0]
    PSC = 100 * optcom_betas / data_optcom.mean(axis=-1, keepdims=True)
    return PSC


def calculate_z_maps(weights, z_max=8):
    """
    Calculate z-statistic maps by z-scoring standardized parameter estimate maps and cropping
    extreme values.

    Parameters
    ----------
    weights
    z_max

    Returns
    -------
    Z_maps
    """
    Z_maps = stats.zscore(weights, axis=0)
    extreme_idx = np.abs(Z_maps) > z_max
    Z_maps[extreme_idx] = z_max * np.sign(Z_maps[extreme_idx])
    return Z_maps


def calculate_f_maps(mixing, data_cat, tes, Z_maps, f_max=500):
    """
    Calculate pseudo-F-statistic maps (per component) for TE-dependence
    and -indepdence models.
    """
    me_betas = get_coeffs(data_cat, mixing)  # TODO: Remove mask arg from get_coeffs
    n_voxels, n_echos, n_components = me_betas.shape
    mu = data_cat.mean(axis=-1, dtype=float)
    tes = np.reshape(tes, (n_echos, 1))

    # set up Xmats
    X1 = mu.T  # Model 1
    X2 = np.tile(tes, (1, n_voxels)) * mu.T  # Model 2

    F_T2_maps = np.zeros([n_voxels, n_components])
    F_S0_maps = np.zeros([n_voxels, n_components])
    pred_T2_maps = np.zeros([n_voxels, n_echos, n_components])
    pred_S0_maps = np.zeros([n_voxels, n_echos, n_components])

    LGR.info('Fitting TE- and S0-dependent models to components')
    for i_comp in range(n_components):
        # size of comp_betas is (n_echoes, n_samples)
        comp_betas = np.atleast_3d(me_betas)[:, :, i_comp].T
        alpha = (np.abs(comp_betas)**2).sum(axis=0)

        # S0 Model
        # (S,) model coefficient map
        coeffs_S0 = (comp_betas * X1).sum(axis=0) / (X1**2).sum(axis=0)
        pred_S0 = X1 * np.tile(coeffs_S0, (n_echos, 1))
        pred_S0_maps[:, :, i_comp] = pred_S0.T
        SSE_S0 = (comp_betas - pred_S0)**2
        SSE_S0 = SSE_S0.sum(axis=0)  # (S,) prediction error map
        F_S0 = (alpha - SSE_S0) * (n_echos - 1) / (SSE_S0)

        # T2 Model
        coeffs_T2 = (comp_betas * X2).sum(axis=0) / (X2**2).sum(axis=0)
        pred_T2 = X2 * np.tile(coeffs_T2, (n_echos, 1))
        pred_T2_maps[:, :, i_comp] = pred_T2.T
        SSE_T2 = (comp_betas - pred_T2)**2
        SSE_T2 = SSE_T2.sum(axis=0)
        F_T2 = (alpha - SSE_T2) * (n_echos - 1) / (SSE_T2)

        F_S0[F_S0 > f_max] = f_max
        F_T2[F_T2 > f_max] = f_max
        F_S0_maps[:, i_comp] = F_S0
        F_T2_maps[:, i_comp] = F_T2

    return F_S0_maps, F_T2_maps


def threshold_to_match(maps, n_sig_voxels, mask, ref_img, csize):
    """
    Cluster-extent threshold a map to have roughly some requested number of
    significant voxels (with clusters accounted for).
    """
    n_voxels, n_components = maps.shape
    clmaps = np.zeros([n_voxels, n_components], bool)
    for i_comp in range(n_components):
        # Initial cluster-defining threshold is defined based on the number
        # of significant voxels from the F-statistic maps. This threshold
        # will be relaxed until the number of significant voxels from both
        # maps is roughly equal.
        ccimg = io.new_nii_like(
            ref_img,
            utils.unmask(stats.rankdata(maps[:, i_comp]), mask))
        step = int(n_sig_voxels[i_comp] / 10)
        rank_thresh = n_voxels - (n_sig_voxels[i_comp] - 1)
        while True:
            clmap = utils.threshold_map(
                ccimg, min_cluster_size=csize,
                threshold=rank_thresh, mask=mask,
                binarize=True)
            diff = n_sig_voxels[i_comp] - clmap.sum()
            if diff < 0 or clmap.sum() == 0:
                rank_thresh += step
                clmap = utils.threshold_map(
                    ccimg, min_cluster_size=csize,
                    threshold=rank_thresh, mask=mask,
                    binarize=True)
                break
            else:
                rank_thresh -= step
        clmaps[:, i_comp] = clmap
    return clmaps


def threshold_map(maps, mask, ref_img, threshold, csize=None):
    n_voxels, n_components = maps.shape
    maps_thresh = np.zeros([n_voxels, n_components], bool)
    LGR.info('Performing spatial clustering of components')
    if csize is None:
        csize = np.max([int(n_voxels * 0.0005) + 5, 20])
    else:
        csize = int(csize)
    LGR.debug('Using minimum cluster size: {}'.format(csize))

    for i_comp in range(n_components):
        # Cluster-extent threshold and binarize F-maps
        ccimg = io.new_nii_like(
            ref_img,
            np.squeeze(utils.unmask(maps[:, i_comp], mask)))
        maps_thresh[:, i_comp] = utils.threshold_map(
            ccimg, min_cluster_size=csize, threshold=threshold, mask=mask,
            binarize=True)
    return maps_thresh


def compute_countsignal(cl_arr):
    """
    Count the number of significant voxels, per map, in a set of cluster-extent
    thresholded maps.

    Parameters
    ----------
    cl_arr
        Statistical map after cluster-extent thresholding and binarization.

    Returns
    -------
    countsignal
        Number of significant (non-zero) voxels for each map in cl_arr.
    """
    countsignal = cl_arr.sum(axis=0)
    return countsignal


def compute_countnoise(stat_maps, stat_cl_maps, stat_thresh=1.95):
    """
    Count the number of significant voxels (after application of
    cluster-defining threshold) from non-significant clusters (after
    cluster-extent thresholding).

    Parameters
    ----------
    stat_maps
        Unthresholded statistical maps.
    stat_cl_maps
        Cluster-extent thresholded and binarized version of stat_maps.
    stat_thresh
        Statistical threshold. Default is 1.95 (Z-statistic threshold
        corresponding to p<X one-sided).
    """
    noise_idx = (np.abs(stat_maps) > stat_thresh) & (stat_cl_maps == 0)
    countnoise = noise_idx.sum(axis=0)
    return countnoise


def compute_signal_minus_noise_z(Z_maps, Z_clmaps, F_T2_maps, z_thresh=1.95):
    """
    Divide voxel-level thresholded F-statistic maps into distributions of
    signal (voxels in significant clusters) and noise (voxels from
    non-significant clusters) statistics, then compare these distributions
    with a two-sample t-test. Convert the resulting t-statistics (per map)
    to normally distributed z-statistics.

    Parameters
    ----------
    Z_maps
    Z_clmaps
    F_T2_maps
    z_thresh

    Returns
    -------
    signal_minus_noise_z
    signal_minus_noise_p
    """
    n_components = Z_maps.shape[1]
    signal_minus_noise_z = np.zeros(n_components)
    signal_minus_noise_p = np.zeros(n_components)
    noise_idx = (np.abs(Z_maps) > z_thresh) & (Z_clmaps == 0)
    countnoise = noise_idx.sum(axis=0)
    countsignal = Z_clmaps.sum(axis=0)
    for i_comp in range(n_components):
        noise_FT2_Z = 0.5 * np.log(F_T2_maps[noise_idx[:, i_comp], i_comp])
        signal_FT2_Z = 0.5 * np.log(F_T2_maps[Z_clmaps[:, i_comp] == 1, i_comp])
        n_noise_dupls = noise_FT2_Z.size - np.unique(noise_FT2_Z).size
        if n_noise_dupls:
            LGR.debug('For component {}, {} duplicate noise F-values '
                      'detected.'.format(i_comp, n_noise_dupls))
        n_signal_dupls = signal_FT2_Z.size - np.unique(signal_FT2_Z).size
        if n_signal_dupls:
            LGR.debug('For component {}, {} duplicate signal F-values '
                      'detected.'.format(i_comp, n_signal_dupls))
        dof = countnoise[i_comp] + countsignal[i_comp] - 2

        t_value, signal_minus_noise_p[i_comp] = stats.ttest_ind(
            signal_FT2_Z, noise_FT2_Z, equal_var=False)
        signal_minus_noise_z[i_comp] = t_to_z(t_value, dof)

    signal_minus_noise_z = np.nan_to_num(signal_minus_noise_z, 0)
    signal_minus_noise_p = np.nan_to_num(signal_minus_noise_p, 0)
    return signal_minus_noise_z, signal_minus_noise_p


def compute_signal_minus_noise_t(Z_maps, Z_clmaps, F_T2_maps, z_thresh=1.95):
    RepLGR.info('A t-test was performed between the distributions of T2*-model '
                'F-statistics associated with clusters (i.e., signal) and '
                'non-cluster voxels (i.e., noise) to generate a t-statistic '
                '(metric signal-noise_t) and p-value (metric signal-noise_p) '
                'measuring relative association of the component to signal '
                'over noise.')
    n_components = Z_maps.shape[1]
    signal_minus_noise_t = np.zeros(n_components)
    signal_minus_noise_p = np.zeros(n_components)
    noise_idx = (np.abs(Z_maps) > z_thresh) & (Z_clmaps == 0)
    for i_comp in range(n_components):
        # NOTE: Why only compare distributions of *unique* F-statistics?
        noise_FT2_Z = np.log10(np.unique(F_T2_maps[noise_idx[:, i_comp], i_comp]))
        signal_FT2_Z = np.log10(np.unique(F_T2_maps[Z_clmaps[:, i_comp] == 1, i_comp]))
        (signal_minus_noise_t[i_comp],
         signal_minus_noise_p[i_comp]) = stats.ttest_ind(
             signal_FT2_Z, noise_FT2_Z, equal_var=False)

    signal_minus_noise_t = np.nan_to_num(signal_minus_noise_t, 0)
    signal_minus_noise_p = np.nan_to_num(signal_minus_noise_p, 0)
    return signal_minus_noise_t, signal_minus_noise_p


def compute_dice(clmaps1, clmaps2, axis=0):
    """
    Compute the Dice similarity index between two thresholded and binarized maps.

    Parameters
    ----------
    Br_clmaps, F_clmaps
    axis

    Returns
    -------
    dice_values
    """
    dice_values = utils.dice(clmaps1, clmaps2, axis=axis)
    dice_values = np.nan_to_num(dice_values, 0)
    return dice_values


def generate_decision_table_score(kappa, dice_FT2, signal_minus_noise_t,
                                  countnoise, countsigFT2):
    """
    Generate a five-metric decision table. Metrics are ranked in either descending
    or ascending order if they measure TE-dependence or -independence, respectively,
    and are then averaged for each component.

    Parameters
    ----------
    kappa
    dice_FT2
    signal_minus_noise_t
    countnoise
    countsigFT2

    Returns
    -------
    d_table_score
    """
    d_table_rank = np.vstack([
        len(kappa) - stats.rankdata(kappa),
        len(kappa) - stats.rankdata(dice_FT2),
        len(kappa) - stats.rankdata(signal_minus_noise_t),
        stats.rankdata(countnoise),
        len(kappa) - stats.rankdata(countsigFT2)]).T
    d_table_score = d_table_rank.mean(axis=1)
    return d_table_score


def calculate_dependence_metrics(F_T2_maps, F_S0_maps, Z_maps):
    """
    Calculate Kappa and Rho metrics from F-statistic maps.
    Just a weighted average over voxels.

    Parameters
    ----------
    F_T2_maps, F_S0_maps
    Z_maps

    Returns
    -------
    kappas, rhos
    """
    RepLGR.info('Kappa (kappa) and Rho (rho) were calculated as measures of '
                'TE-dependence and TE-independence, respectively.')

    weight_maps = Z_maps ** 2.
    n_components = Z_maps.shape[1]
    kappas, rhos = np.zeros(n_components), np.zeros(n_components)
    for i_comp in range(n_components):
        kappas[i_comp] = np.average(F_T2_maps[:, i_comp], weights=weight_maps[:, i_comp])
        rhos[i_comp] = np.average(F_S0_maps[:, i_comp], weights=weight_maps[:, i_comp])
    return kappas, rhos


def calculate_varex(optcom_betas):
    """
    Calculate unnormalized(?) variance explained from unstandardized
    parameter estimate maps.

    Parameters
    ----------
    optcom_betas

    Returns
    -------
    varex
    """
    compvar = (optcom_betas ** 2).sum(axis=0)
    varex = 100 * (compvar / compvar.sum())
    return varex


def calculate_varex_norm(weights):
    """
    Calculate normalized variance explained from standardized parameter
    estimate maps.

    Parameters
    ----------
    weights

    Returns
    -------
    varex_norm
    """
    compvar = (weights ** 2).sum(axis=0)
    varex_norm = compvar / compvar.sum()
    return varex_norm
