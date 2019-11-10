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
    Calculate weights.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined data
    mixing_z : (T x C) array_like
        Z-scored mixing matrix

    Returns
    -------
    weights : (S x C) array_like
        Parameter estimates for optimally combined data against the mixing
        matrix.
    """
    # compute un-normalized weight dataset (features)
    weights = computefeats2(data_optcom, mixing_z, normalize=False)
    return weights


def calculate_betas(data_optcom, mixing):
    assert data_optcom.shape[1] == mixing.shape[0]
    # demean optimal combination
    data_optcom_dm = data_optcom - data_optcom.mean(axis=-1, keepdims=True)
    # compute PSC dataset - shouldn't have to refit data
    betas = get_coeffs(data_optcom_dm, mixing)
    return betas


def calculate_psc(data_optcom, optcom_betas):
    assert data_optcom.shape[1] == optcom_betas.shape[0]
    PSC = 100 * optcom_betas / data_optcom.mean(axis=-1, keepdims=True)
    return PSC


def compute_countsignal(cl_arr):
    countsignal = cl_arr.sum(axis=0)
    return countsignal


def compute_countnoise(Z_maps, Z_cl_maps, z_thresh=1.95):
    noise_idx = (np.abs(Z_maps) > z_thresh) & (Z_clmaps == 0)
    countnoise = np.sum(noise_idx).sum(axis=0)
    return countnoise


def compute_signal_minus_noise_z(Z_maps, Z_cl_maps, F_T2_maps, z_thresh=1.95):
    n_components = Z_maps.shape[1]
    signal_minus_noise_t = np.zeros(n_components)
    signal_minus_noise_p = np.zeros(n_components)
    for i_comp in range(n_components):
        # index voxels significantly loading on component but not from clusters
        comp_noise_idx = ((np.abs(Z_maps[:, i_comp]) > z_thresh) &
                          (Z_clmaps[:, i_comp] == 0))
        countsignal = Z_clmaps[:, i_comp].sum()
        # NOTE: Why only compare distributions of *unique* F-statistics?
        noise_FT2_Z = 0.5 * np.log(F_T2_maps[comp_noise_idx, i_comp])
        signal_FT2_Z = 0.5 * np.log(F_T2_maps[Z_clmaps[:, i_comp] == 1, i_comp]))
        n_noise_dupls = noise_FT2_Z.size - np.unique(noise_FT2_Z).size
        if n_noise_dupls:
            LGR.debug('For component {}, {} duplicate noise F-values '
                      'detected.'.format(i_comp, n_noise_dupls))
        n_signal_dupls = signal_FT2_Z.size - np.unique(signal_FT2_Z).size
        if n_signal_dupls:
            LGR.debug('For component {}, {} duplicate signal F-values '
                      'detected.'.format(i_comp, n_signal_dupes))
        dof = np.sum(comp_noise_idx) + countsignal - 2

        t_value, signal_minus_noise_p[i_comp] = stats.ttest_ind(
            signal_FT2_Z, noise_FT2_Z, equal_var=False)
        signal_minus_noise_z[i_comp] = t_to_z(t_value, dof)

    signal_minus_noise_z = np.nan_to_num(signal_minus_noise_z, 0)
    signal_minus_noise_p = np.nan_to_num(signal_minus_noise_p, 0)
    return signal_minus_noise_z, signal_minus_noise_p


def compute_signal_minus_noise_t(Z_maps, Z_cl_maps, F_T2_maps, z_thresh=1.95):
    RepLGR.info('A t-test was performed between the distributions of T2*-model '
                'F-statistics associated with clusters (i.e., signal) and '
                'non-cluster voxels (i.e., noise) to generate a t-statistic '
                '(metric signal-noise_t) and p-value (metric signal-noise_p) '
                'measuring relative association of the component to signal '
                'over noise.')
    n_components = Z_maps.shape[1]
    signal_minus_noise_t = np.zeros(n_components)
    signal_minus_noise_p = np.zeros(n_components)
    for i_comp in range(n_components):
        # index voxels significantly loading on component but not from clusters
        comp_noise_sel = ((np.abs(Z_maps[:, i_comp]) > z_thresh) &
                          (Z_clmaps[:, i_comp] == 0))
        comptable.loc[i_comp, 'countnoise'] = np.array(
            comp_noise_sel, dtype=np.int).sum()
        # NOTE: Why only compare distributions of *unique* F-statistics?
        noise_FT2_Z = np.log10(np.unique(F_T2_maps[comp_noise_sel, i_comp]))
        signal_FT2_Z = np.log10(np.unique(
            F_T2_maps[Z_clmaps[:, i_comp] == 1, i_comp]))
        (signal_minus_noise_t[i_comp],
         signal_minus_noise_p[i_comp]) = stats.ttest_ind(
             signal_FT2_Z, noise_FT2_Z, equal_var=False)

    signal_minus_noise_t = np.nan_to_num(signal_minus_noise_t, 0)
    signal_minus_noise_p = np.nan_to_num(signal_minus_noise_p, 0)
    return signal_minus_noise_t, signal_minus_noise_p


def compute_dice(Br_clmaps, F_clmaps):
    n_components = Br_clmaps.shape[1]
    dice_values = np.zeros(n_components)
    for i_comp in range(n_components):
        dice_values[i_comp] = utils.dice(Br_clmaps[:, i_comp], F_clmaps[:, i_comp])

    dice_values = np.nan_to_num(dice_values, 0)
    return dice_values


def generate_decision_table_score(kappa, dice_FT2, signal_minus_noise_t,
                                  countnoise, countsigFT2):
    d_table_rank = np.vstack([
        len(kappa) - stats.rankdata(kappa),
        len(kappa) - stats.rankdata(dice_FT2),
        len(kappa) - stats.rankdata(signal_minus_noise_t),
        stats.rankdata(countnoise),
        len(kappa) - stats.rankdata(countsigFT2)]).T
    d_table_score = d_table_rank.mean(axis=1)
    return d_table_score


def calculate_z_maps(weights, z_max=8):
    n_components = weights.shape[1]
    Z_maps = np.zeros(weights.shape)
    for i_comp in range(n_components):
        # compute weights as Z-values
        weights_z = (weights[:, i_comp] - weights[:, i_comp].mean()) / weights[:, i_comp].std()
        weights_z[np.abs(weights_z) > z_max] = (z_max * (np.abs(weights_z) / weights_z))[
            np.abs(weights_z) > z_max]
        Z_maps[:, i_comp] = weights_z
    return Z_maps


def calculate_dependence_metrics(comptable, F_T2_maps, F_S0_maps, Z_maps):
    RepLGR.info('Kappa (kappa) and Rho (rho) were calculated as measures of '
                'TE-dependence and TE-independence, respectively.')
    _names = ['kappa', 'rho']
    if any([name in comptable.columns for name in _names]):
        raise Exception('Metrics already exist in component table.')

    n_components = Z_maps.shape[1]
    kappas, rhos = np.zeros(n_components), np.zeros(n_components)
    for i_comp in range(n_components):
        norm_weights = np.abs(Z_maps[:, i_comp] ** 2.)
        kappas[i_comp] = np.average(F_T2_maps[:, i_comp], weights=norm_weights)
        rhos[i_comp] = np.average(F_S0_maps[:, i_comp], weights=norm_weights)
    comptable['kappa'] = kappas
    comptable['rho'] = rhos
    return comptable


def calculate_varex(optcom_betas):
    compvar = (optcom_betas ** 2).sum(axis=0)
    varex = 100 * (compvar / compvar.sum())
    return varex


def calculate_varex_norm(weights):
    compvar = (weights ** 2).sum(axis=0)
    varex_norm = compvar / compvar.sum()
    return varex_norm


def calculate_f_maps(mixing, data_cat, tes, Z_maps, f_max=500):
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


def spatial_cluster(F_T2_maps, F_S0_maps, Z_maps, optcom_betas, mask, n_echos, csize=None):
    n_voxels, n_components = Z_maps.shape
    fmin, _, _ = getfbounds(n_echos)

    optcom_betas_abs = np.abs(optcom_betas)

    # Generate clustering criteria for component selection
    Z_clmaps = np.zeros([n_voxels, n_components], bool)
    F_T2_clmaps = np.zeros([n_voxels, n_components], bool)
    F_S0_clmaps = np.zeros([n_voxels, n_components], bool)
    Br_T2_clmaps = np.zeros([n_voxels, n_components], bool)
    Br_S0_clmaps = np.zeros([n_voxels, n_components], bool)

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
            np.squeeze(utils.unmask(F_T2_maps[:, i_comp], mask)))
        F_T2_clmaps[:, i_comp] = utils.threshold_map(
            ccimg, min_cluster_size=csize, threshold=fmin, mask=mask,
            binarize=True)
        countsigFT2 = F_T2_clmaps[:, i_comp].sum()

        ccimg = io.new_nii_like(
            ref_img,
            np.squeeze(utils.unmask(F_S0_maps[:, i_comp], mask)))
        F_S0_clmaps[:, i_comp] = utils.threshold_map(
            ccimg, min_cluster_size=csize, threshold=fmin, mask=mask,
            binarize=True)
        countsigFS0 = F_S0_clmaps[:, i_comp].sum()

        # Cluster-extent threshold and binarize Z-maps with CDT of p < 0.05
        ccimg = io.new_nii_like(
            ref_img,
            np.squeeze(utils.unmask(Z_maps[:, i_comp], mask)))
        Z_clmaps[:, i_comp] = utils.threshold_map(
            ccimg, min_cluster_size=csize, threshold=1.95, mask=mask,
            binarize=True)

        # Initial cluster-defining threshold is defined based on the number
        # of significant voxels from the F-statistic maps. This threshold
        # will be relaxed until the number of significant voxels from both
        # maps is roughly equal.
        ccimg = io.new_nii_like(
            ref_img,
            utils.unmask(stats.rankdata(optcom_betas_abs[:, i_comp]), mask))
        step = int(countsigFT2 / 10)
        T2_thresh = n_voxels - (countsigFT2 - 1)
        while True:
            Br_T2_clmap = utils.threshold_map(
                ccimg, min_cluster_size=csize,
                threshold=T2_thresh, mask=mask,
                binarize=True)
            diff = countsigFT2 - Br_T2_clmap.sum()
            if diff < 0 or Br_T2_clmap.sum() == 0:
                T2_thresh += step
                Br_T2_clmap = utils.threshold_map(
                    ccimg, min_cluster_size=csize,
                    threshold=T2_thresh, mask=mask,
                    binarize=True)
                break
            else:
                T2_thresh -= step
        Br_T2_clmaps[:, i_comp] = Br_T2_clmap

        ccimg = io.new_nii_like(
            ref_img,
            utils.unmask(stats.rankdata(optcom_betas_abs[:, i_comp]), mask))
        step = int(countsigFS0 / 10)
        S0_thresh = n_voxels - (countsigFS0 - 1)
        while True:
            Br_S0_clmap = utils.threshold_map(
                ccimg, min_cluster_size=csize,
                threshold=S0_thresh, mask=mask,
                binarize=True)
            diff = countsigFS0 - Br_S0_clmap.sum()
            if diff < 0 or Br_S0_clmap.sum() == 0:
                S0_thresh += step
                Br_S0_clmap = utils.threshold_map(
                    ccimg, min_cluster_size=csize,
                    threshold=S0_thresh, mask=mask,
                    binarize=True)
                break
            else:
                S0_thresh -= step
        Br_S0_clmaps[:, i_comp] = Br_S0_clmap
    return Z_clmaps, F_T2_clmaps, F_S0_clmaps, Br_T2_clmaps, Br_S0_clmaps
