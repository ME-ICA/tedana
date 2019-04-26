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
    Classify components in seldict as "accepted," "rejected," or "ignored."

    The selection process uses previously calculated parameters listed in `seldict`
    for each ICA component such as Kappa (a T2* weighting metric), Rho (an S0 weighting metric),
    and variance explained. See `Notes` for additional calculated metrics used to
    classify each component into one of the four listed groups.

    Parameters
    ----------
    seldict : :obj:`dict`
        A dictionary with component-specific features used for classification.
        As output from `fitmodels_direct`
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
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
        classification (accepted, rejected, or ignored)

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

    # Lists of components
    all_comps = np.arange(comptable.shape[0])
    # unclf is a full list that is whittled down over criteria
    # since the default classification is "accepted", at the end of the tree
    # the remaining elements in unclf are classified as accepted
    unclf = all_comps.copy()

    # If user has specified
    if manacc:
        LGR.info('Performing manual ICA component selection')
        if ('classification' in comptable.columns and
                'original_classification' not in comptable.columns):
            comptable['original_classification'] = comptable['classification']
            comptable['original_rationale'] = comptable['rationale']
        comptable['classification'] = 'accepted'
        comptable['rationale'] = ''
        acc = [int(comp) for comp in manacc]
        rej = sorted(np.setdiff1d(all_comps, acc))
        comptable.loc[acc, 'classification'] = 'accepted'
        comptable.loc[rej, 'classification'] = 'rejected'
        comptable.loc[rej, 'rationale'] += 'I001;'
        # Move decision columns to end
        comptable = comptable[[c for c in comptable if c not in cols_at_end] +
                              [c for c in cols_at_end if c in comptable]]
        comptable['rationale'] = comptable['rationale'].str.rstrip(';')
        return comptable

    comptable['classification'] = 'accepted'
    comptable['rationale'] = ''

    Z_maps = seldict['Z_maps']
    Z_clmaps = seldict['Z_clmaps']
    F_R2_maps = seldict['F_R2_maps']
    F_S0_clmaps = seldict['F_S0_clmaps']
    F_R2_clmaps = seldict['F_R2_clmaps']
    Br_S0_clmaps = seldict['Br_S0_clmaps']
    Br_R2_clmaps = seldict['Br_R2_clmaps']

    # Set knobs
    LOW_PERC = 25
    HIGH_PERC = 90
    if n_vols < 100:
        EXTEND_FACTOR = 3
    else:
        EXTEND_FACTOR = 2
    RESTRICT_FACTOR = 2

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
    comptable['dice_FR2'] = np.zeros(all_comps.shape[0])
    comptable['dice_FS0'] = np.zeros(all_comps.shape[0])
    for i_comp in all_comps:
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
    for i_comp in all_comps:
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
        n_comps - stats.rankdata(comptable['kappa']),
        n_comps - stats.rankdata(comptable['dice_FR2']),
        n_comps - stats.rankdata(comptable['signal-noise_t']),
        stats.rankdata(comptable['countnoise']),
        n_comps - stats.rankdata(comptable['countsigFR2'])]).T
    comptable['d_table_score'] = d_table_rank.mean(axis=1)

    """
    Step 1: Reject anything that's obviously an artifact
    a. Estimate a null variance
    """
    # Rho is higher than Kappa
    temp_rej0a = all_comps[(comptable['rho'] > comptable['kappa'])]
    comptable.loc[temp_rej0a, 'classification'] = 'rejected'
    comptable.loc[temp_rej0a, 'rationale'] += 'I002;'

    # Number of significant voxels for S0 model is higher than number for R2
    # model *and* number for R2 model is greater than zero.
    temp_rej0b = all_comps[((comptable['countsigFS0'] > comptable['countsigFR2']) &
                            (comptable['countsigFR2'] > 0))]
    comptable.loc[temp_rej0b, 'classification'] = 'rejected'
    comptable.loc[temp_rej0b, 'rationale'] += 'I003;'
    rej = np.union1d(temp_rej0a, temp_rej0b)

    # Dice score for S0 maps is higher than Dice score for R2 maps and variance
    # explained is higher than the median across components.
    temp_rej1 = all_comps[(comptable['dice_FS0'] > comptable['dice_FR2']) &
                          (comptable['variance explained'] >
                           np.median(comptable['variance explained']))]
    comptable.loc[temp_rej1, 'classification'] = 'rejected'
    comptable.loc[temp_rej1, 'rationale'] += 'I004;'
    rej = np.union1d(temp_rej1, rej)

    # T-value is less than zero (noise has higher F-statistics than signal in
    # map) and variance explained is higher than the median across components.
    temp_rej2 = unclf[(comptable.loc[unclf, 'signal-noise_t'] < 0) &
                      (comptable.loc[unclf, 'variance explained'] >
                      np.median(comptable['variance explained']))]
    comptable.loc[temp_rej2, 'classification'] = 'rejected'
    comptable.loc[temp_rej2, 'rationale'] += 'I005;'
    rej = np.union1d(temp_rej2, rej)
    unclf = np.setdiff1d(unclf, rej)

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
    # Upper limit for variance explained is median across components with high
    # Kappa values. High Kappa is defined as Kappa above Kappa elbow.
    varex_upper_p = np.median(
        comptable.loc[comptable['kappa'] > getelbow(comptable['kappa'], return_val=True),
                      'variance explained'])
    ncls = unclf.copy()
    # NOTE: We're not sure why this is done, nor why it's specifically done
    # three times. Need to look into this deeper, esp. to make sure the 3
    # isn't a hard-coded reference to the number of echoes.
    # Reduce components to investigate as "good" to ones in which change in
    # variance explained is less than the limit defined above.... What?
    for i_loop in range(3):
        ncls = comptable.loc[ncls].loc[
            comptable.loc[
                ncls, 'variance explained'].diff() < varex_upper_p].index.values

    # Compute elbows from other elbows
    f05, _, f01 = utils.getfbounds(n_echos)
    kappas_nonsig = comptable.loc[comptable['kappa'] < f01, 'kappa']
    # NOTE: Would an elbow from all Kappa values *ever* be lower than one from
    # a subset of lower values?
    kappa_elbow = np.min((getelbow(kappas_nonsig, return_val=True),
                          getelbow(comptable['kappa'], return_val=True)))
    rho_elbow = np.mean((getelbow(comptable.loc[ncls, 'rho'], return_val=True),
                         getelbow(comptable['rho'], return_val=True),
                         f05))

    # Provisionally accept components based on Kappa and Rho elbows
    acc_prov = ncls[(comptable.loc[ncls, 'kappa'] >= kappa_elbow) &
                    (comptable.loc[ncls, 'rho'] < rho_elbow)]

    if len(acc_prov) == 0:
        LGR.warning('No BOLD-like components detected')
        ign = sorted(np.setdiff1d(all_comps, rej))
        comptable.loc[ign, 'classification'] = 'ignored'
        comptable.loc[ign, 'rationale'] += 'I006;'

        # Move decision columns to end
        comptable = comptable[[c for c in comptable if c not in cols_at_end] +
                              [c for c in cols_at_end if c in comptable]]
        comptable['rationale'] = comptable['rationale'].str.rstrip(';')
        return comptable

    # Calculate "rate" for kappa: kappa range divided by variance explained
    # range, for potentially accepted components
    # NOTE: What is the logic behind this?
    kappa_rate = ((np.max(comptable.loc[acc_prov, 'kappa']) -
                   np.min(comptable.loc[acc_prov, 'kappa'])) /
                  (np.max(comptable.loc[acc_prov, 'variance explained']) -
                   np.min(comptable.loc[acc_prov, 'variance explained'])))
    comptable['kappa ratio'] = kappa_rate * comptable['variance explained'] / comptable['kappa']
    varex_lower = stats.scoreatpercentile(
        comptable.loc[acc_prov, 'variance explained'], LOW_PERC)
    varex_upper = stats.scoreatpercentile(
        comptable.loc[acc_prov, 'variance explained'], HIGH_PERC)

    """
    Step 3: Get rid of midk components; i.e., those with higher than
    max decision score and high variance
    """
    max_good_d_score = EXTEND_FACTOR * len(acc_prov)
    midk = unclf[(comptable.loc[unclf, 'd_table_score'] > max_good_d_score) &
                 (comptable.loc[unclf, 'variance explained'] > EXTEND_FACTOR * varex_upper)]
    comptable.loc[midk, 'classification'] = 'rejected'
    comptable.loc[midk, 'rationale'] += 'I007;'
    unclf = np.setdiff1d(unclf, midk)
    acc_prov = np.setdiff1d(acc_prov, midk)

    """
    Step 4: Find components to ignore
    """
    # collect high variance unclassified components
    # and mix of high/low provisionally accepted
    high_varex = np.union1d(
        acc_prov,
        unclf[comptable.loc[unclf, 'variance explained'] > varex_lower])
    # ignore low variance components
    ign = np.setdiff1d(unclf, high_varex)
    # but only if they have bad decision scores
    ign = np.setdiff1d(
        ign, ign[comptable.loc[ign, 'd_table_score'] < max_good_d_score])
    # and low kappa
    ign = np.setdiff1d(ign, ign[comptable.loc[ign, 'kappa'] > kappa_elbow])
    comptable.loc[ign, 'classification'] = 'ignored'
    comptable.loc[ign, 'rationale'] += 'I008;'
    unclf = np.setdiff1d(unclf, ign)

    """
    Step 5: Scrub the set if there are components that haven't been rejected or
    ignored, but are still not listed in the provisionally accepted group.
    """
    if len(unclf) > len(acc_prov):
        comptable['d_table_score_scrub'] = np.nan
        # Recompute the midk steps on the limited set to clean up the tail
        d_table_rank = np.vstack([
            len(unclf) - stats.rankdata(comptable.loc[unclf, 'kappa']),
            len(unclf) - stats.rankdata(comptable.loc[unclf, 'dice_FR2']),
            len(unclf) - stats.rankdata(comptable.loc[unclf, 'signal-noise_t']),
            stats.rankdata(comptable.loc[unclf, 'countnoise']),
            len(unclf) - stats.rankdata(comptable.loc[unclf, 'countsigFR2'])]).T
        comptable.loc[unclf, 'd_table_score_scrub'] = d_table_rank.mean(1)
        num_acc_guess = int(np.mean([
            np.sum((comptable.loc[unclf, 'kappa'] > kappa_elbow) &
                   (comptable.loc[unclf, 'rho'] < rho_elbow)),
            np.sum(comptable.loc[unclf, 'kappa'] > kappa_elbow)]))

        # Rejection candidate based on artifact type A: candartA
        conservative_guess = num_acc_guess / RESTRICT_FACTOR
        candartA = np.intersect1d(
            unclf[comptable.loc[unclf, 'd_table_score_scrub'] > conservative_guess],
            unclf[comptable.loc[unclf, 'kappa ratio'] > EXTEND_FACTOR * 2])
        candartA = (candartA[comptable.loc[candartA, 'variance explained'] >
                    varex_upper * EXTEND_FACTOR])
        comptable.loc[candartA, 'classification'] = 'rejected'
        comptable.loc[candartA, 'rationale'] += 'I009;'
        midk = np.union1d(midk, candartA)
        unclf = np.setdiff1d(unclf, midk)

        # Rejection candidate based on artifact type B: candartB
        conservative_guess2 = num_acc_guess * HIGH_PERC / 100.
        candartB = unclf[comptable.loc[unclf, 'd_table_score_scrub'] > conservative_guess2]
        candartB = (candartB[comptable.loc[candartB, 'variance explained'] >
                    varex_lower * EXTEND_FACTOR])
        comptable.loc[candartB, 'classification'] = 'rejected'
        comptable.loc[candartB, 'rationale'] += 'I010;'
        midk = np.union1d(midk, candartB)
        unclf = np.setdiff1d(unclf, midk)

        # Find components to ignore
        # Ignore high variance explained, poor decision tree scored components
        new_varex_lower = stats.scoreatpercentile(
            comptable.loc[unclf[:num_acc_guess], 'variance explained'],
            LOW_PERC)
        candart = unclf[comptable.loc[unclf, 'd_table_score_scrub'] > num_acc_guess]
        ign_add0 = candart[comptable.loc[candart, 'variance explained'] > new_varex_lower]
        ign_add0 = np.setdiff1d(ign_add0, midk)
        comptable.loc[ign_add0, 'classification'] = 'ignored'
        comptable.loc[ign_add0, 'rationale'] += 'I011;'
        ign = np.union1d(ign, ign_add0)
        unclf = np.setdiff1d(unclf, ign)

        # Ignore low Kappa, high variance explained components
        ign_add1 = np.intersect1d(
            unclf[comptable.loc[unclf, 'kappa'] <= kappa_elbow],
            unclf[comptable.loc[unclf, 'variance explained'] > new_varex_lower])
        ign_add1 = np.setdiff1d(ign_add1, midk)
        comptable.loc[ign_add1, 'classification'] = 'ignored'
        comptable.loc[ign_add1, 'rationale'] += 'I012;'

    # at this point, unclf is equivalent to accepted

    # Move decision columns to end
    comptable = comptable[[c for c in comptable if c not in cols_at_end] +
                          [c for c in cols_at_end if c in comptable]]
    comptable['rationale'] = comptable['rationale'].str.rstrip(';')
    return comptable
