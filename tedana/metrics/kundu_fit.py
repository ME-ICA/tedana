"""
Fit models.
"""
import logging
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats

from tedana import io, utils
from tedana.stats import getfbounds, computefeats2, get_coeffs


LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

F_MAX = 500
Z_MAX = 8


def dependence_metrics(catd, tsoc, mmix, adaptive_mask, tes, ref_img,
                       reindex=False, mmixN=None, algorithm=None, label=None,
                       out_dir='.', verbose=False):
    """
    Fit TE-dependence and -independence models to components.

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input data, where `S` is samples, `E` is echos, and `T` is time
    tsoc : (S x T) array_like
        Optimally combined data
    mmix : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `catd`
    adaptive_mask : (S) array_like
        Adaptive mask, where each voxel's value is the number of echoes with
        "good signal".
    tes : list
        List of echo times associated with `catd`, in milliseconds
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    reindex : bool, optional
        Whether to sort components in descending order by Kappa. Default: False
    mmixN : (T x C) array_like, optional
        Z-scored mixing matrix. Default: None
    algorithm : {'kundu_v2', 'kundu_v3', None}, optional
        Decision tree to be applied to metrics. Determines which maps will be
        generated and stored in seldict. Default: None
    label : :obj:`str` or None, optional
        Prefix to apply to generated files. Default is None.
    out_dir : :obj:`str`, optional
        Output directory for generated files. Default is current working
        directory.
    verbose : :obj:`bool`, optional
        Whether or not to generate additional files. Default is False.

    Returns
    -------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index is the component number.
    seldict : :obj:`dict` or None
        Dictionary containing component-specific metric maps to be used for
        component selection. If `algorithm` is None, then seldict will be None as
        well.
    betas : :obj:`numpy.ndarray`
    mmix_corrected : :obj:`numpy.ndarray`
        Mixing matrix after sign correction and resorting (if reindex is True).
    """
    # Use adaptive_mask as mask
    mask = adaptive_mask >= 3

    if not (catd.shape[0] == adaptive_mask.shape[0] == tsoc.shape[0]):
        raise ValueError('First dimensions (number of samples) of catd ({0}), '
                         'tsoc ({1}), and adaptive_mask ({2}) do not '
                         'match'.format(catd.shape[0], tsoc.shape[0],
                                        adaptive_mask.shape[0]))
    elif catd.shape[1] != len(tes):
        raise ValueError('Second dimension of catd ({0}) does not match '
                         'number of echoes provided (tes; '
                         '{1})'.format(catd.shape[1], len(tes)))
    elif not (catd.shape[2] == tsoc.shape[1] == mmix.shape[0]):
        raise ValueError('Number of volumes in catd ({0}), '
                         'tsoc ({1}), and mmix ({2}) do not '
                         'match.'.format(catd.shape[2], tsoc.shape[1],
                                         mmix.shape[0]))

    RepLGR.info("A series of TE-dependence metrics were calculated for "
                "each component, including Kappa, Rho, and variance "
                "explained.")

    # mask everything we can
    tsoc = tsoc[mask, :]
    catd = catd[mask, ...]

    # demean optimal combination
    tsoc_dm = tsoc - tsoc.mean(axis=-1, keepdims=True)

    # compute un-normalized weight dataset (features)
    if mmixN is None:
        mmixN = mmix
    WTS = computefeats2(tsoc, mmixN, mask=None, normalize=False)

    # compute PSC dataset - shouldn't have to refit data
    tsoc_B = get_coeffs(tsoc_dm, mmix, mask=None, add_const=False)
    del tsoc_dm
    tsoc_Babs = np.abs(tsoc_B)
    PSC = tsoc_B / tsoc.mean(axis=-1, keepdims=True) * 100

    # compute skews to determine signs based on unnormalized weights,
    # correct mmix & WTS signs based on spatial distribution tails
    signs = stats.skew(WTS, axis=0)
    signs /= np.abs(signs)
    mmix_corrected = mmix * signs
    WTS *= signs
    PSC *= signs
    totvar = (tsoc_B**2).sum()
    totvar_norm = (WTS**2).sum()

    # compute Betas and means over TEs for TE-dependence analysis
    betas = get_coeffs(utils.unmask(catd, mask),
                       mmix_corrected,
                       np.repeat(mask[:, np.newaxis], len(tes), axis=1),
                       add_const=True)
    betas = betas[mask, ...]
    n_voxels, n_echos, n_components = betas.shape
    mu = catd.mean(axis=-1, dtype=float)
    tes = np.reshape(tes, (n_echos, 1))
    fmin, _, _ = getfbounds(n_echos)

    # set up design matrices
    X1 = mu.T  # Model 1: TE-independence model
    X2 = np.tile(tes, (1, n_voxels)) * mu.T  # Model 2: TE-dependence model

    # tables for component selection
    kappas = np.zeros([n_components])
    rhos = np.zeros([n_components])
    varex = np.zeros([n_components])
    varex_norm = np.zeros([n_components])
    Z_maps = np.zeros([n_voxels, n_components])
    F_R2_maps = np.zeros([n_voxels, n_components])
    F_S0_maps = np.zeros([n_voxels, n_components])
    if verbose:
        pred_R2_maps = np.zeros([n_voxels, n_echos, n_components])
        pred_S0_maps = np.zeros([n_voxels, n_echos, n_components])

    LGR.info('Fitting TE- and S0-dependent models to components')
    for i_comp in range(n_components):
        # size of comp_betas is (n_echoes, n_samples)
        comp_betas = np.atleast_3d(betas)[:, :, i_comp].T
        alpha = (np.abs(comp_betas)**2).sum(axis=0)
        varex[i_comp] = (tsoc_B[:, i_comp]**2).sum() / totvar * 100.
        varex_norm[i_comp] = (WTS[:, i_comp]**2).sum() / totvar_norm

        for j_echo in np.unique(adaptive_mask[adaptive_mask >= 3]):
            mask_idx = adaptive_mask == j_echo
            alpha = (np.abs(comp_betas[:j_echo])**2).sum(axis=0)

            # S0 Model
            # (S,) model coefficient map
            coeffs_S0 = (comp_betas[:j_echo] * X1[:j_echo, :]).sum(axis=0) /\
                (X1[:j_echo, :]**2).sum(axis=0)
            pred_S0 = X1[:j_echo, :] * np.tile(coeffs_S0, (j_echo, 1))
            SSE_S0 = (comp_betas[:j_echo] - pred_S0)**2
            SSE_S0 = SSE_S0.sum(axis=0)  # (S,) prediction error map
            F_S0 = (alpha - SSE_S0) * (j_echo - 1) / (SSE_S0)
            F_S0_maps[mask_idx[mask], i_comp] = F_S0[mask_idx[mask]]

            # R2 Model
            coeffs_R2 = (comp_betas[:j_echo] * X2[:j_echo, :]).sum(axis=0) /\
                (X2[:j_echo, :]**2).sum(axis=0)
            pred_R2 = X2[:j_echo] * np.tile(coeffs_R2, (j_echo, 1))
            SSE_R2 = (comp_betas[:j_echo] - pred_R2)**2
            SSE_R2 = SSE_R2.sum(axis=0)
            F_R2 = (alpha - SSE_R2) * (j_echo - 1) / (SSE_R2)
            F_R2_maps[mask_idx[mask], i_comp] = F_R2[mask_idx[mask]]

            if verbose:
                pred_S0_maps[mask_idx[mask], :j_echo, i_comp] = pred_S0.T[mask_idx[mask], :]
                pred_R2_maps[mask_idx[mask], :j_echo, i_comp] = pred_R2.T[mask_idx[mask], :]

        # compute weights as Z-values
        wtsZ = (WTS[:, i_comp] - WTS[:, i_comp].mean()) / WTS[:, i_comp].std()
        wtsZ[np.abs(wtsZ) > Z_MAX] = (Z_MAX * (np.abs(wtsZ) / wtsZ))[
            np.abs(wtsZ) > Z_MAX]
        Z_maps[:, i_comp] = wtsZ

        # compute Kappa and Rho
        F_S0[F_S0 > F_MAX] = F_MAX
        F_R2[F_R2 > F_MAX] = F_MAX
        norm_weights = np.abs(wtsZ ** 2.)
        kappas[i_comp] = np.average(F_R2, weights=norm_weights)
        rhos[i_comp] = np.average(F_S0, weights=norm_weights)
    del SSE_S0, SSE_R2, wtsZ, F_S0, F_R2, norm_weights, comp_betas
    if algorithm != 'kundu_v3':
        del WTS, PSC, tsoc_B

    # tabulate component values
    comptable = np.vstack([kappas, rhos, varex, varex_norm]).T
    if reindex:
        # re-index all components in descending Kappa order
        sort_idx = comptable[:, 0].argsort()[::-1]
        comptable = comptable[sort_idx, :]
        mmix_corrected = mmix_corrected[:, sort_idx]
        betas = betas[..., sort_idx]
        F_R2_maps = F_R2_maps[:, sort_idx]
        F_S0_maps = F_S0_maps[:, sort_idx]
        Z_maps = Z_maps[:, sort_idx]
        tsoc_Babs = tsoc_Babs[:, sort_idx]

        if verbose:
            pred_R2_maps = pred_R2_maps[:, :, sort_idx]
            pred_S0_maps = pred_S0_maps[:, :, sort_idx]

        if algorithm == 'kundu_v3':
            WTS = WTS[:, sort_idx]
            PSC = PSC[:, sort_idx]
            tsoc_B = tsoc_B[:, sort_idx]

    if verbose:
        # Echo-specific weight maps for each of the ICA components.
        io.filewrite(utils.unmask(betas, mask),
                     op.join(out_dir, '{0}betas_catd.nii'.format(label)),
                     ref_img)

        # Echo-specific maps of predicted values for R2 and S0 models for each
        # component.
        io.filewrite(utils.unmask(pred_R2_maps, mask),
                     op.join(out_dir, '{0}R2_pred.nii'.format(label)), ref_img)
        io.filewrite(utils.unmask(pred_S0_maps, mask),
                     op.join(out_dir, '{0}S0_pred.nii'.format(label)), ref_img)
        # Weight maps used to average metrics across voxels
        io.filewrite(utils.unmask(Z_maps ** 2., mask),
                     op.join(out_dir, '{0}metric_weights.nii'.format(label)),
                     ref_img)
        del pred_R2_maps, pred_S0_maps

    comptable = pd.DataFrame(comptable,
                             columns=['kappa', 'rho',
                                      'variance explained',
                                      'normalized variance explained'])
    comptable.index.name = 'component'

    # Generate clustering criteria for component selection
    if algorithm in ['kundu_v2', 'kundu_v3']:
        Z_clmaps = np.zeros([n_voxels, n_components], bool)
        F_R2_clmaps = np.zeros([n_voxels, n_components], bool)
        F_S0_clmaps = np.zeros([n_voxels, n_components], bool)
        Br_R2_clmaps = np.zeros([n_voxels, n_components], bool)
        Br_S0_clmaps = np.zeros([n_voxels, n_components], bool)

        LGR.info('Performing spatial clustering of components')
        csize = np.max([int(n_voxels * 0.0005) + 5, 20])
        LGR.debug('Using minimum cluster size: {}'.format(csize))
        for i_comp in range(n_components):
            # Cluster-extent threshold and binarize F-maps
            ccimg = io.new_nii_like(
                ref_img,
                np.squeeze(utils.unmask(F_R2_maps[:, i_comp], mask)))
            F_R2_clmaps[:, i_comp] = utils.threshold_map(
                ccimg, min_cluster_size=csize, threshold=fmin, mask=mask,
                binarize=True)
            countsigFR2 = F_R2_clmaps[:, i_comp].sum()

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

            # Cluster-extent threshold and binarize ranked signal-change map
            ccimg = io.new_nii_like(
                ref_img,
                utils.unmask(stats.rankdata(tsoc_Babs[:, i_comp]), mask))
            Br_R2_clmaps[:, i_comp] = utils.threshold_map(
                ccimg, min_cluster_size=csize,
                threshold=(max(tsoc_Babs.shape) - countsigFR2), mask=mask,
                binarize=True)
            Br_S0_clmaps[:, i_comp] = utils.threshold_map(
                ccimg, min_cluster_size=csize,
                threshold=(max(tsoc_Babs.shape) - countsigFS0), mask=mask,
                binarize=True)
        del ccimg, tsoc_Babs

        if algorithm == 'kundu_v2':
            # WTS, tsoc_B, PSC, and F_S0_maps are not used by Kundu v2.5
            selvars = ['Z_maps', 'F_R2_maps',
                       'Z_clmaps', 'F_R2_clmaps', 'F_S0_clmaps',
                       'Br_R2_clmaps', 'Br_S0_clmaps']
        elif algorithm == 'kundu_v3':
            selvars = ['WTS', 'tsoc_B', 'PSC',
                       'Z_maps', 'F_R2_maps', 'F_S0_maps',
                       'Z_clmaps', 'F_R2_clmaps', 'F_S0_clmaps',
                       'Br_R2_clmaps', 'Br_S0_clmaps']
        elif algorithm is None:
            selvars = []
        else:
            raise ValueError('Algorithm "{0}" not recognized.'.format(algorithm))

        seldict = {}
        for vv in selvars:
            seldict[vv] = eval(vv)
    else:
        seldict = None

    return comptable, seldict, betas, mmix_corrected


def kundu_metrics(comptable, metric_maps):
    """
    Compute metrics used by Kundu v2.5 and v3.2 decision trees.

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table, where `C` is components and `M` is metrics
    metric_maps : :obj:`dict`
        A dictionary with component-specific feature maps used for
        classification. The value for each key is a (S x C) array, where `S` is
        voxels and `C` is components. Generated by `dependence_metrics`

    Returns
    -------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metrics to be used for component selection, with new metrics
        added.
    """
    Z_maps = metric_maps['Z_maps']
    Z_clmaps = metric_maps['Z_clmaps']
    F_R2_maps = metric_maps['F_R2_maps']
    F_S0_clmaps = metric_maps['F_S0_clmaps']
    F_R2_clmaps = metric_maps['F_R2_clmaps']
    Br_S0_clmaps = metric_maps['Br_S0_clmaps']
    Br_R2_clmaps = metric_maps['Br_R2_clmaps']

    """
    Tally number of significant voxels for cluster-extent thresholded R2 and S0
    model F-statistic maps.
    """
    comptable['countsigFR2'] = F_R2_clmaps.sum(axis=0)
    comptable['countsigFS0'] = F_S0_clmaps.sum(axis=0)

    """
    Generate Dice values for R2 and S0 models
    - dice_FR2: Dice value of cluster-extent thresholded maps of R2-model betas
      and F-statistics.
    - dice_FS0: Dice value of cluster-extent thresholded maps of S0-model betas
      and F-statistics.
    """
    comptable['dice_FR2'] = np.zeros(comptable.shape[0])
    comptable['dice_FS0'] = np.zeros(comptable.shape[0])
    for i_comp in comptable.index:
        comptable.loc[i_comp, 'dice_FR2'] = utils.dice(Br_R2_clmaps[:, i_comp],
                                                       F_R2_clmaps[:, i_comp])
        comptable.loc[i_comp, 'dice_FS0'] = utils.dice(Br_S0_clmaps[:, i_comp],
                                                       F_S0_clmaps[:, i_comp])

    comptable.loc[np.isnan(comptable['dice_FR2']), 'dice_FR2'] = 0
    comptable.loc[np.isnan(comptable['dice_FS0']), 'dice_FS0'] = 0

    """
    Generate three metrics of component noise:
    - countnoise: Number of "noise" voxels (voxels highly weighted for
      component, but not from clusters)
    - signal-noise_t: T-statistic for two-sample t-test of F-statistics from
      "signal" voxels (voxels in clusters) against "noise" voxels (voxels not
      in clusters) for R2 model.
    - signal-noise_p: P-value from t-test.
    """
    comptable['countnoise'] = 0
    comptable['signal-noise_t'] = 0
    comptable['signal-noise_p'] = 0
    for i_comp in comptable.index:
        # index voxels significantly loading on component but not from clusters
        comp_noise_sel = ((np.abs(Z_maps[:, i_comp]) > 1.95) &
                          (Z_clmaps[:, i_comp] == 0))
        comptable.loc[i_comp, 'countnoise'] = np.array(
            comp_noise_sel, dtype=np.int).sum()
        # NOTE: Why only compare distributions of *unique* F-statistics?
        noise_FR2_Z = np.log10(np.unique(F_R2_maps[comp_noise_sel, i_comp]))
        signal_FR2_Z = np.log10(np.unique(
            F_R2_maps[Z_clmaps[:, i_comp] == 1, i_comp]))
        (comptable.loc[i_comp, 'signal-noise_t'],
         comptable.loc[i_comp, 'signal-noise_p']) = stats.ttest_ind(
             signal_FR2_Z, noise_FR2_Z, equal_var=False)

    comptable.loc[np.isnan(comptable['signal-noise_t']), 'signal-noise_t'] = 0
    comptable.loc[np.isnan(comptable['signal-noise_p']), 'signal-noise_p'] = 0

    """
    Assemble decision table with five metrics:
    - Kappa values ranked from largest to smallest
    - R2-model F-score map/beta map Dice scores ranked from largest to smallest
    - Signal F > Noise F t-statistics ranked from largest to smallest
    - Number of "noise" voxels (voxels highly weighted for component, but not
      from clusters) ranked from smallest to largest
    - Number of voxels with significant R2-model F-scores within clusters
      ranked from largest to smallest

    Smaller values (i.e., higher ranks) across metrics indicate more BOLD
    dependence and less noise.
    """
    d_table_rank = np.vstack([
        comptable.shape[0] - stats.rankdata(comptable['kappa']),
        comptable.shape[0] - stats.rankdata(comptable['dice_FR2']),
        comptable.shape[0] - stats.rankdata(comptable['signal-noise_t']),
        stats.rankdata(comptable['countnoise']),
        comptable.shape[0] - stats.rankdata(comptable['countsigFR2'])]).T
    comptable['d_table_score'] = d_table_rank.mean(axis=1)

    return comptable
