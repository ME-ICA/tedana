"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging
import numpy as np
import pkg_resources
from scipy import stats
from tedana import utils
from tedana.selection._utils import (getelbow_mod)

LGR = logging.getLogger(__name__)
RESOURCES = pkg_resources.resource_filename('tedana', 'tests/data')


def selcomps(seldict, mmix, manacc, n_echos):
    """
    Classify components in seldict as accepted, rejected, midk, or ignored
    based on Kundu approach v2.5.

    Parameters
    ----------
    seldict : :obj:`dict`
        Component-specific metrics and maps produced by `fitmodels_direct`
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the number of volumes in the original data
    manacc : :obj:`list`
        Comma-separated list of indices of manually accepted components
    n_echos : :obj:`int`
        Number of echos in original data

    Returns
    -------
    acc : :obj:`list`
        Indices of accepted (BOLD) components in `mmix`
    rej : :obj:`list`
        Indices of rejected (non-BOLD) components in `mmix`
    midk : :obj:`list`
        Indices of mid-K (questionable) components in `mmix`
        These components are typically removed from the data during denoising
    ign : :obj:`list`
        Indices of ignored components in `mmix`
        Ignored components are considered to have too low variance to matter.
        They are not processed through the accept vs reject decision tree and
        are NOT removed during the denoising process

    Notes
    -----
    The selection algorithm used in this function is from work by prantikk
    It is from selcomps function in select_model.py in version 2.5 of MEICA at:
    https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model.py
    Some later publications using and evaluating the MEICA method used a
    different selection algorithm by prantikk. The 3.2 version of that
    algorithm in the selcomps function in select_model_fft20e.py at:
    https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model_fft20e.py
    In both algorithms, the ICA component selection process uses multiple
    metrics that include: kappa, rho, variance explained, component spatial
    weighting maps, noise and spatial frequency metrics, and measures of
    spatial overlap across metrics. The precise calculations may vary between
    algorithms. The most notable difference is that the v2.5 algorithm is a
    fixed decision tree where all sections were made based on whether
    combinations of metrics crossed various thresholds. In the v3.2 algorithm,
    clustering and support vector machines are also used to classify components
    based on how similar metrics in one component are similar to metrics in
    other components.
    """
    Kappas = seldict['Kappas']
    Rhos = seldict['Rhos']
    varex = seldict['varex']
    Z_maps = seldict['Z_maps']
    Z_clmaps = seldict['Z_clmaps']
    F_S0_clmaps = seldict['F_S0_clmaps']
    F_R2_clmaps = seldict['F_R2_clmaps']
    Br_clmaps_S0 = seldict['Br_clmaps_S0']
    Br_clmaps_R2 = seldict['Br_clmaps_R2']

    n_vols = mmix.shape[1]

    # List of components
    all_comps = np.arange(len(Kappas))
    acc = np.arange(len(Kappas))
    midk = []
    ign = []

    # If user has specified
    if manacc:
        acc = sorted([int(vv) for vv in manacc.split(',')])
        midk, ign = [], []
        rej = sorted(np.setdiff1d(acc, acc))
        return acc, rej, midk, ign

    """
    Set knobs
    """
    LOW_PERC = 25
    HIGH_PERC = 90
    if n_vols < 100:
        EXTEND_FACTOR = 3
    else:
        EXTEND_FACTOR = 2
    RESTRICT_FACTOR = 2

    """
    Do some tallies for no. of significant voxels
    """
    countsigFS0 = F_S0_clmaps.sum(0)
    countsigFR2 = F_R2_clmaps.sum(0)
    countnoise = np.zeros(len(all_comps))

    """
    Make table of dice values
    """
    dice_table = np.zeros([all_comps.shape[0], 2])
    for i_comp in acc:
        dice_FR2 = utils.dice(Br_clmaps_R2[:, i_comp], F_R2_clmaps[:, i_comp])
        dice_FS0 = utils.dice(Br_clmaps_S0[:, i_comp], F_S0_clmaps[:, i_comp])
        dice_table[i_comp, :] = [dice_FR2, dice_FS0]  # step 3a here and above
        dice_table[np.isnan(dice_table)] = 0

    """
    Make table of noise gain
    """
    tt_table = np.zeros([len(all_comps), 4])
    counts_FR2_Z = np.zeros([len(all_comps), 2])
    for i_comp in all_comps:
        comp_noise_sel = utils.andb([np.abs(Z_maps[:, i_comp]) > 1.95,
                                     Z_clmaps[:, i_comp] == 0]) == 2
        countnoise[i_comp] = np.array(comp_noise_sel, dtype=np.int).sum()
        noise_FR2_Z = np.log10(np.unique(F_R2_maps[comp_noise_sel, i_comp]))
        signal_FR2_Z = np.log10(np.unique(F_R2_maps[Z_clmaps[:, i_comp] == 1,
                                          i_comp]))
        counts_FR2_Z[i_comp, :] = [len(signal_FR2_Z), len(noise_FR2_Z)]
        tt_table[i_comp, :2] = stats.ttest_ind(signal_FR2_Z, noise_FR2_Z,
                                               equal_var=False)
        tt_table[np.isnan(tt_table)] = 0

    """
    Assemble decision table
    """
    d_table_rank = np.vstack([
        len(all_comps)-stats.rankdata(Kappas, method='ordinal'),
        len(all_comps)-stats.rankdata(dice_table[:, 0], method='ordinal'),
        len(all_comps)-stats.rankdata(tt_table[:, 0], method='ordinal'),
        stats.rankdata(countnoise, method='ordinal'),
        len(all_comps)-stats.rankdata(countsigFR2, method='ordinal')]).T
    d_table_score = d_table_rank.sum(1)

    """
    Step 1: Reject anything that's obviously an artifact
    a. Estimate a null variance
    """
    rej = acc[utils.andb([Rhos > Kappas, countsigFS0 > countsigFR2]) > 0]
    rej = np.union1d(rej, acc[utils.andb([dice_table[:, 1] > dice_table[:, 0],
                                          varex > np.median(varex)]) == 2])
    rej = np.union1d(rej,
                     acc[utils.andb([tt_table[acc, 0] < 0,
                                     varex[acc] > np.median(varex)]) == 2])
    acc = np.setdiff1d(acc, rej)
    varex_ub_p = np.median(varex[Kappas > Kappas[getelbow_mod(Kappas)]])

    """
    Step 2: Make a  guess for what the good components are, in order to
    estimate good component properties
    a. Not outlier variance
    b. Kappa>kappa_elbow
    c. Rho<Rho_elbow
    d. High R2* dice compared to S0 dice
    e. Gain of F_R2 in clusters vs noise
    f. Estimate a low and high variance
    """
    # Step 2a
    ncls = acc.copy()
    for nn in range(3):
        ncls = ncls[1:][(varex[ncls][1:] - varex[ncls][:-1]) < varex_ub_p]
    Kappas_lim = Kappas[Kappas < utils.getfbounds(n_echos)[-1]]
    Rhos_lim = np.array(sorted(Rhos[ncls])[::-1])
    Rhos_sorted = np.array(sorted(Rhos)[::-1])
    Kappas_elbow = min(Kappas_lim[getelbow_mod(Kappas_lim)],
                       Kappas[getelbow_mod(Kappas)])
    Rhos_elbow = np.mean([Rhos_lim[getelbow_mod(Rhos_lim)],
                          Rhos_sorted[getelbow_mod(Rhos_sorted)],
                         utils.getfbounds(n_echos)[0]])
    good_guess = ncls[utils.andb([Kappas[ncls] >= Kappas_elbow,
                                  Rhos[ncls] < Rhos_elbow]) == 2]

    if len(good_guess) == 0:
        return [], sorted(rej), [], sorted(np.setdiff1d(all_comps, rej))

    Kappa_rate = ((max(Kappas[good_guess]) - min(Kappas[good_guess])) /
                  (max(varex[good_guess]) - min(varex[good_guess])))
    Kappa_ratios = Kappa_rate * varex / Kappas
    varex_lb = stats.scoreatpercentile(varex[good_guess], LOW_PERC)
    varex_ub = stats.scoreatpercentile(varex[good_guess], HIGH_PERC)

    """
    Step 3: Get rid of midk components; i.e., those with higher than
    max decision score and high variance
    """
    max_good_d_score = EXTEND_FACTOR*len(good_guess) * d_table_rank.shape[1]
    midkadd = acc[utils.andb([
        d_table_score[acc] > max_good_d_score,
        varex[acc] > EXTEND_FACTOR*varex_ub]) == 2]
    midk = np.union1d(midkadd, midk)
    acc = np.setdiff1d(acc, midk)

    """
    Step 4: Find components to ignore
    """
    good_guess = np.setdiff1d(good_guess, midk)
    loaded = np.union1d(good_guess, acc[varex[acc] > varex_lb])
    igncand = np.setdiff1d(acc, loaded)
    igncand = np.setdiff1d(igncand,
                           igncand[d_table_score[igncand] < max_good_d_score])
    igncand = np.setdiff1d(igncand, igncand[Kappas[igncand] > Kappas_elbow])
    ign = np.array(np.union1d(ign, igncand), dtype=np.int)
    acc = np.setdiff1d(acc, ign)

    """
    Step 5: Scrub the set
    """
    if len(acc) > len(good_guess):
        # Recompute the midk steps on the limited set to clean up the tail
        d_table_rank = np.vstack([
            len(acc) - stats.rankdata(Kappas[acc], method='ordinal'),
            len(acc) - stats.rankdata(dice_table[acc, 0], method='ordinal'),
            len(acc) - stats.rankdata(tt_table[acc, 0], method='ordinal'),
            stats.rankdata(countnoise[acc], method='ordinal'),
            len(acc) - stats.rankdata(countsigFR2[acc], method='ordinal')]).T
        d_table_score = d_table_rank.sum(1)
        num_acc_guess = np.mean([
            np.sum(utils.andb([Kappas[acc] > Kappas_elbow,
                               Rhos[acc] < Rhos_elbow]) == 2),
            np.sum(Kappas[acc] > Kappas_elbow)])
        conservative_guess = num_acc_guess * d_table_rank.shape[1] / RESTRICT_FACTOR
        candartA = np.intersect1d(acc[d_table_score > conservative_guess],
                                  acc[Kappa_ratios[acc] > EXTEND_FACTOR * 2])
        midkadd = np.union1d(midkadd, np.intersect1d(
            candartA, candartA[varex[candartA] > varex_ub * EXTEND_FACTOR]))
        candartB = acc[d_table_score > num_acc_guess * d_table_rank.shape[1] * HIGH_PERC / 100.]
        midkadd = np.union1d(midkadd, np.intersect1d(
            candartB, candartB[varex[candartB] > varex_lb * EXTEND_FACTOR]))
        midk = np.union1d(midk, midkadd)

        # Find comps to ignore
        new_varex_lb = stats.scoreatpercentile(varex[acc[:num_acc_guess]],
                                               LOW_PERC)
        candart = np.setdiff1d(
            acc[d_table_score > num_acc_guess * d_table_rank.shape[1]], midk)
        ignadd = np.intersect1d(candart[varex[candart] > new_varex_lb],
                                candart)
        ignadd = np.union1d(ignadd, np.intersect1d(
            acc[Kappas[acc] <= Kappas_elbow], acc[varex[acc] > new_varex_lb]))
        ign = np.setdiff1d(np.union1d(ign, ignadd), midk)
        acc = np.setdiff1d(acc, np.union1d(midk, ign))

    acc = list(sorted(acc))
    rej = list(sorted(rej))
    midk = list(sorted(midk))
    ign = list(sorted(ign))

    return acc, rej, midk, ign
