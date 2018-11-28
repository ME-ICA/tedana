"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging
import numpy as np
from scipy import stats

from tedana import utils
from tedana.selection._utils import getelbow

LGR = logging.getLogger(__name__)


def selcomps(seldict, comptable, mmix, manacc, n_echos):
    """
    Classify components in seldict as "accepted," "rejected," "midk," or "ignored."

    The selection process uses previously calculated parameters listed in `seldict`
    for each ICA component such as Kappa (a T2* weighting metric), Rho (an S0 weighting metric),
    and variance explained. See `Notes` for additional calculated metrics used to
    classify each component into one of the four listed groups.

    Parameters
    ----------
    seldict : :obj:`dict`
        A dictionary with component-specific features used for classification.
        As output from `fitmodels_direct`
    comptable : (C x 5) :obj:`pandas.DataFrame`
        Component metric table
    mmix : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the number of volumes in the original data
    manacc : :obj:`list`
        Comma-separated list of indices of manually accepted components
    n_echos : :obj:`int`
        Number of echos in original data

    Returns
    -------
    comptable : :obj:`pandas.DataFrame`
        Updated component table with additional metrics and with
        classification (accepted, rejected, midk, or ignored)

    Notes
    -----
    The selection algorithm used in this function was originated in ME-ICA
    by Prantik Kundu, and his original implementation is available at:
    https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model.py

    This component selection process uses multiple, previously calculated metrics that include:
    kappa, rho, variance explained, component spatial weighting maps, noise and spatial
    frequency metrics, and measures of spatial overlap across metrics.

    Prantik began to update these selection criteria to use SVMs to
    distinguish components, a hypercommented version of this attempt is available at:
    https://gist.github.com/emdupre/ca92d52d345d08ee85e104093b81482e
    """

    cols_at_end = ['classification', 'rationale']
    comptable['classification'] = 'accepted'
    comptable['rationale'] = ''

    Z_maps = seldict['Z_maps']
    Z_clmaps = seldict['Z_clmaps']
    F_R2_maps = seldict['F_R2_maps']
    F_S0_clmaps = seldict['F_S0_clmaps']
    F_R2_clmaps = seldict['F_R2_clmaps']
    Br_S0_clmaps = seldict['Br_S0_clmaps']
    Br_R2_clmaps = seldict['Br_R2_clmaps']

    n_vols, n_comps = mmix.shape

    # Set knobs
    LOW_PERC = 25
    HIGH_PERC = 90
    if n_vols < 100:
        EXTEND_FACTOR = 3
    else:
        EXTEND_FACTOR = 2
    RESTRICT_FACTOR = 2

    # List of components
    midk = []
    ign = []
    all_comps = np.arange(comptable.shape[0])
    acc = np.arange(comptable.shape[0])

    # If user has specified
    if manacc:
        acc = sorted([int(vv) for vv in manacc.split(',')])
        rej = sorted(np.setdiff1d(all_comps, acc))
        comptable.loc[acc, 'classification'] = 'accepted'
        comptable.loc[rej, 'classification'] = 'rejected'
        comptable.loc[rej, 'rationale'] += 'manual exclusion;'
        # Move decision columns to end
        comptable = comptable[[c for c in comptable if c not in cols_at_end] +
                              [c for c in cols_at_end if c in comptable]]
        return comptable

    """
    Do some tallies for no. of significant voxels
    """
    countnoise = np.zeros(n_comps)
    comptable['countsigFR2'] = F_R2_clmaps.sum(axis=0)
    comptable['countsigFS0'] = F_S0_clmaps.sum(axis=0)

    """
    Make table of dice values
    """
    comptable['dice_FR2'] = np.zeros(all_comps.shape[0])
    comptable['dice_FS0'] = np.zeros(all_comps.shape[0])
    for i_comp in acc:
        comptable.loc[i_comp, 'dice_FR2'] = utils.dice(Br_R2_clmaps[:, i_comp],
                                                       F_R2_clmaps[:, i_comp])
        comptable.loc[i_comp, 'dice_FS0'] = utils.dice(Br_S0_clmaps[:, i_comp],
                                                       F_S0_clmaps[:, i_comp])

    comptable.loc[np.isnan(comptable['dice_FR2']), 'dice_FR2'] = 0
    comptable.loc[np.isnan(comptable['dice_FS0']), 'dice_FS0'] = 0

    """
    Make table of noise gain
    """
    comptable['countnoise'] = 0
    comptable['signal-noise_t'] = 0
    comptable['signal-noise_p'] = 0
    for i_comp in all_comps:
        comp_noise_sel = ((np.abs(Z_maps[:, i_comp]) > 1.95) &
                          (Z_clmaps[:, i_comp] == 0))
        comptable.loc[i_comp, 'countnoise'] = np.array(
            comp_noise_sel, dtype=np.int).sum()
        noise_FR2_Z = np.log10(np.unique(F_R2_maps[comp_noise_sel, i_comp]))
        signal_FR2_Z = np.log10(np.unique(
            F_R2_maps[Z_clmaps[:, i_comp] == 1, i_comp]))
        (comptable.loc[i_comp, 'signal-noise_t'],
         comptable.loc[i_comp, 'signal-noise_p']) = stats.ttest_ind(
             signal_FR2_Z, noise_FR2_Z, equal_var=False)

    comptable.loc[np.isnan(comptable['signal-noise_t']), 'signal-noise_t'] = 0
    comptable.loc[np.isnan(comptable['signal-noise_p']), 'signal-noise_p'] = 0

    """
    Assemble decision table
    """
    d_table_rank = np.vstack([
        n_comps-stats.rankdata(comptable['kappa'], method='ordinal'),
        n_comps-stats.rankdata(comptable['dice_FR2'], method='ordinal'),
        n_comps-stats.rankdata(comptable['signal-noise_t'], method='ordinal'),
        stats.rankdata(countnoise, method='ordinal'),
        n_comps-stats.rankdata(comptable['countsigFR2'], method='ordinal')]).T
    n_decision_metrics = d_table_rank.shape[1]
    comptable['d_table_score'] = d_table_rank.sum(axis=1)

    """
    Step 1: Reject anything that's obviously an artifact
    a. Estimate a null variance
    """
    temp_rej0 = all_comps[(comptable['rho'] > comptable['kappa']) |
                          ((comptable['countsigFS0'] > comptable['countsigFR2']) &
                           (comptable['countsigFR2'] > 0))]
    comptable.loc[temp_rej0, 'classification'] = 'rejected'
    comptable.loc[temp_rej0, 'rationale'] += ('Rho>Kappa or more significant voxels '
                                              'in S0 model than R2 model;')

    temp_rej1 = all_comps[(comptable['dice_FS0'] > comptable['dice_FR2']) &
                          (comptable['variance explained'] >
                           np.median(comptable['variance explained']))]
    comptable.loc[temp_rej1, 'classification'] = 'rejected'
    comptable.loc[temp_rej1, 'rationale'] += ('S0 dice is higher than R2 dice '
                                              'and high variance explained;')
    rej = np.union1d(temp_rej0, temp_rej1)

    temp_rej2 = acc[(comptable.loc[acc, 'signal-noise_t'] < 0) &
                    (comptable.loc[acc, 'variance explained'] >
                     np.median(comptable['variance explained']))]
    comptable.loc[temp_rej2, 'classification'] = 'rejected'
    comptable.loc[temp_rej2, 'rationale'] += ('signal-noise_t < 0 '
                                              'and high variance explained;')
    rej = np.union1d(temp_rej2, rej)

    acc = np.setdiff1d(acc, rej)

    """
    Step 2: Make a guess for what the good components are, in order to
    estimate good component properties
    a. Not outlier variance
    b. Kappa>kappa_elbow
    c. Rho<Rho_elbow
    d. High R2* dice compared to S0 dice
    e. Gain of F_R2 in clusters vs noise
    f. Estimate a low and high variance
    """
    # Step 2a
    varex_upper_p = np.median(
        comptable.loc[comptable['kappa'] > getelbow(comptable['kappa'], return_val=True),
                      'variance explained'])
    ncls = acc.copy()
    # NOTE: We're not sure why this is done, nor why it's specifically done
    # three times. Need to look into this deeper, esp. to make sure the 3
    # isn't a hard-coded reference to the number of echoes.
    for nn in range(3):
        ncls = comptable.loc[ncls].loc[
            comptable.loc[
                ncls, 'variance explained'].diff() < varex_upper_p].index.values

    # Compute elbows
    kappas_lim = comptable.loc[comptable['kappa'] < utils.getfbounds(n_echos)[-1], 'kappa']
    kappa_elbow = np.min((getelbow(kappas_lim, return_val=True),
                          getelbow(comptable['kappa'], return_val=True)))
    rho_elbow = np.mean((getelbow(comptable.loc[ncls, 'rho'], return_val=True),
                         getelbow(comptable['rho'], return_val=True),
                         utils.getfbounds(n_echos)[0]))

    # Initial guess of good components based on Kappa and Rho elbows
    good_guess = ncls[(comptable.loc[ncls, 'kappa'] >= kappa_elbow) &
                      (comptable.loc[ncls, 'rho'] < rho_elbow)]

    if len(good_guess) == 0:
        LGR.warning('No BOLD-like components detected')
        ign = sorted(np.setdiff1d(all_comps, rej))
        comptable.loc[ign, 'classification'] = 'ignored'
        comptable.loc[ign, 'rationale'] += 'no good components found;'

        # Move decision columns to end
        comptable = comptable[[c for c in comptable if c not in cols_at_end] +
                              [c for c in cols_at_end if c in comptable]]
        return comptable

    kappa_rate = ((np.max(comptable.loc[good_guess, 'kappa']) -
                   np.min(comptable.loc[good_guess, 'kappa'])) /
                  (np.max(comptable.loc[good_guess, 'variance explained']) -
                   np.min(comptable.loc[good_guess, 'variance explained'])))
    kappa_ratios = kappa_rate * comptable['variance explained'] / comptable['kappa']
    varex_lower = stats.scoreatpercentile(
        comptable.loc[good_guess, 'variance explained'], LOW_PERC)
    varex_upper = stats.scoreatpercentile(
        comptable.loc[good_guess, 'variance explained'], HIGH_PERC)

    """
    Step 3: Get rid of midk components; i.e., those with higher than
    max decision score and high variance
    """
    max_good_d_score = EXTEND_FACTOR * len(good_guess) * n_decision_metrics
    midk = acc[(comptable.loc[acc, 'd_table_score'] > max_good_d_score) &
               (comptable.loc[acc, 'variance explained'] > EXTEND_FACTOR * varex_upper)]
    comptable.loc[midk, 'classification'] = 'rejected'
    comptable.loc[midk, 'rationale'] += 'midk;'
    acc = np.setdiff1d(acc, midk)

    """
    Step 4: Find components to ignore
    """
    good_guess = np.setdiff1d(good_guess, midk)
    loaded = np.union1d(good_guess, acc[comptable.loc[acc, 'variance explained'] > varex_lower])
    ign = np.setdiff1d(acc, loaded)
    ign = np.setdiff1d(
        ign, ign[comptable.loc[ign, 'd_table_score'] < max_good_d_score])
    ign = np.setdiff1d(ign, ign[comptable.loc[ign, 'kappa'] > kappa_elbow])
    comptable.loc[ign, 'classification'] = 'ignored'
    comptable.loc[ign, 'rationale'] += 'low variance;'
    acc = np.setdiff1d(acc, ign)

    """
    Step 5: Scrub the set
    """
    if len(acc) > len(good_guess):
        # Recompute the midk steps on the limited set to clean up the tail
        d_table_rank = np.vstack([
            len(acc) - stats.rankdata(comptable.loc[acc, 'kappa'], method='ordinal'),
            len(acc) - stats.rankdata(comptable.loc[acc, 'dice_FR2'], method='ordinal'),
            len(acc) - stats.rankdata(comptable.loc[acc, 'signal-noise_t'], method='ordinal'),
            stats.rankdata(countnoise[acc], method='ordinal'),
            len(acc) - stats.rankdata(comptable.loc[acc, 'countsigFR2'], method='ordinal')]).T
        comptable['d_table_score_scrub'] = np.nan
        comptable.loc[acc, 'd_table_score_scrub'] = d_table_rank.sum(1)
        num_acc_guess = int(np.mean([
            np.sum((comptable.loc[acc, 'kappa'] > kappa_elbow) &
                   (comptable.loc[acc, 'rho'] < rho_elbow)),
            np.sum(comptable.loc[acc, 'kappa'] > kappa_elbow)]))
        conservative_guess = num_acc_guess * n_decision_metrics / RESTRICT_FACTOR

        # Rejection candidate based on artifact type A: candartA
        candartA = np.intersect1d(
            acc[comptable.loc[acc, 'd_table_score_scrub'] > conservative_guess],
            acc[kappa_ratios[acc] > EXTEND_FACTOR * 2])
        candartA = np.intersect1d(
            candartA,
            candartA[comptable.loc[candartA, 'variance explained'] > varex_upper * EXTEND_FACTOR])
        comptable.loc[candartA, 'classification'] = 'rejected'
        comptable.loc[candartA, 'rationale'] += 'candartA;'  # TODO: Better rationale
        midk = np.union1d(midk, candartA)

        # Rejection candidate based on artifact type B: candartB
        candartB = comptable.loc[acc].loc[
            comptable.loc[acc, 'd_table_score_scrub'] >
            num_acc_guess * n_decision_metrics * HIGH_PERC / 100.].index.values
        candartB = np.intersect1d(
            candartB,
            candartB[comptable.loc[candartB, 'variance explained'] > varex_lower * EXTEND_FACTOR])
        midk = np.union1d(midk, candartB)
        comptable.loc[candartB, 'classification'] = 'rejected'
        comptable.loc[candartB, 'rationale'] += 'candartB;'  # TODO: Better rationale

        # Find comps to ignore
        new_varex_lower = stats.scoreatpercentile(
            comptable.loc[acc[:num_acc_guess], 'variance explained'],
            LOW_PERC)
        candart = comptable.loc[acc].loc[
            comptable.loc[acc, 'd_table_score'] >
            num_acc_guess * n_decision_metrics].index.values
        ign_add0 = np.intersect1d(
            candart[comptable.loc[candart, 'variance explained'] > new_varex_lower],
            candart)
        ign_add0 = np.setdiff1d(ign_add0, midk)
        comptable.loc[ign_add0, 'classification'] = 'ignored'
        comptable.loc[ign_add0, 'rationale'] += 'ign_add0;'  # TODO: Better rationale
        ign = np.union1d(ign, ign_add0)

        ign_add1 = np.intersect1d(
            acc[comptable.loc[acc, 'kappa'] <= kappa_elbow],
            acc[comptable.loc[acc, 'variance explained'] > new_varex_lower])
        ign_add1 = np.setdiff1d(ign_add1, midk)
        comptable.loc[ign_add1, 'classification'] = 'ignored'
        comptable.loc[ign_add1, 'rationale'] += 'ign_add1;'  # TODO: Better rationale
        ign = np.union1d(ign, ign_add1)
        acc = np.setdiff1d(acc, np.union1d(midk, ign))

    # Move decision columns to end
    comptable = comptable[[c for c in comptable if c not in cols_at_end] +
                          [c for c in cols_at_end if c in comptable]]
    return comptable
