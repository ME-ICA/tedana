"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging
import numpy as np
import pkg_resources
from scipy import stats
from tedana import utils
from tedana.selection._utils import (getelbow)

LGR = logging.getLogger(__name__)
RESOURCES = pkg_resources.resource_filename('tedana', 'tests/data')


def selcomps(seldict):

    # Dump dictionary into variable names
    for key in seldict.keys():
        exec("%s=seldict['%s']" % (key, key))

    # List of components
    midk = []
    ign = []
    nc = np.arange(len(Kappas))
    ncl = np.arange(len(Kappas))

    """
    Set knobs
    """
    LOW_PERC = 25
    HIGH_PERC = 90
    if nt < 100:
        EXTEND_FACTOR = 3
    else:
        EXTEND_FACTOR = 2
    RESTRICT_FACTOR = 2

    """
    Do some tallies for no. of significant voxels
    """
    countsigFS0 = F_S0_clmaps.sum(0)
    countsigFR2 = F_R2_clmaps.sum(0)
    countnoise = np.zeros(len(nc))

    """
    Make table of dice values
    """
    dice_table = np.zeros([nc.shape[0], 2])
    for ii in ncl:
        dice_FR2 = utils.dice(Br_clmaps_R2[:, ii], F_R2_clmaps[:, ii])
        dice_FS0 = utils.dice(Br_clmaps_S0[:, ii], F_S0_clmaps[:, ii])
        dice_table[ii, :] = [dice_FR2, dice_FS0]  # step 3a here and above
        dice_table[np.isnan(dice_table)] = 0

    """
    Make table of noise gain
    """
    tt_table = np.zeros([len(nc), 4])
    counts_FR2_Z = np.zeros([len(nc), 2])
    for ii in nc:
        comp_noise_sel = utils.andb([np.abs(Z_maps[:, ii]) > 1.95, Z_clmaps[:, ii] == 0]) == 2
        countnoise[ii] = np.array(comp_noise_sel, dtype=np.int).sum()
        noise_FR2_Z = np.log10(np.unique(F_R2_maps[comp_noise_sel, ii]))
        signal_FR2_Z = np.log10(np.unique(F_R2_maps[Z_clmaps[:, ii] == 1, ii]))
        counts_FR2_Z[ii, :] = [len(signal_FR2_Z), len(noise_FR2_Z)]
        tt_table[ii, :2] = stats.ttest_ind(signal_FR2_Z, noise_FR2_Z, equal_var=False)
        tt_table[np.isnan(tt_table)] = 0

    """
    Assemble decision table
    """
    d_table_rank = np.vstack([len(nc)-stats.rankdata(Kappas, method='ordinal'),
                              len(nc)-stats.rankdata(dice_table[:, 0], method='ordinal'),
                              len(nc)-stats.rankdata(tt_table[:, 0], method='ordinal'),
                              stats.rankdata(countnoise, method='ordinal'),
                              len(nc)-stats.rankdata(countsigFR2, method='ordinal')]).T
    d_table_score = d_table_rank.sum(1)

    """
    Step 1: Reject anything that's obviously an artifact
    a. Estimate a null variance
    """
    rej = ncl[utils.andb([Rhos > Kappas, countsigFS0 > countsigFR2]) > 0]
    rej = np.union1d(rej, ncl[utils.andb([dice_table[:, 1] > dice_table[:, 0],
                                          varex > np.median(varex)]) == 2])
    rej = np.union1d(rej, ncl[utils.andb([tt_table[ncl, 0] < 0,
                                          varex[ncl] > np.median(varex)]) == 2])
    ncl = np.setdiff1d(ncl, rej)
    varex_ub_p = np.median(varex[Kappas > Kappas[getelbow(Kappas)]])

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
    ncls = ncl.copy()
    for nn in range(3):
        ncls = ncls[1:][(varex[ncls][1:] - varex[ncls][:-1]) < varex_ub_p]  # Step 2a
    Kappas_lim = Kappas[Kappas < utils.getfbounds(ne)[-1]]
    Rhos_lim = np.array(sorted(Rhos[ncls])[::-1])
    Rhos_sorted = np.array(sorted(Rhos)[::-1])
    Kappas_elbow = min(Kappas_lim[getelbow(Kappas_lim)], Kappas[getelbow(Kappas)])
    Rhos_elbow = np.mean([Rhos_lim[getelbow(Rhos_lim)], Rhos_sorted[getelbow(Rhos_sorted)],
                         utils.getfbounds(ne)[0]])
    good_guess = ncls[utils.andb([Kappas[ncls] >= Kappas_elbow, Rhos[ncls] < Rhos_elbow]) == 2]

    if len(good_guess) == 0:
        return [], sorted(rej), [], sorted(np.setdiff1d(nc, rej))

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
    midkadd = ncl[utils.andb([d_table_score[ncl] > max_good_d_score,
                  varex[ncl] > EXTEND_FACTOR*varex_ub]) == 2]
    midk = np.union1d(midkadd, midk)
    ncl = np.setdiff1d(ncl, midk)

    """
    Step 4: Find components to ignore
    """
    good_guess = np.setdiff1d(good_guess, midk)
    loaded = np.union1d(good_guess, ncl[varex[ncl] > varex_lb])
    igncand = np.setdiff1d(ncl, loaded)
    igncand = np.setdiff1d(igncand, igncand[d_table_score[igncand] < max_good_d_score])
    igncand = np.setdiff1d(igncand, igncand[Kappas[igncand] > Kappas_elbow])
    ign = np.array(np.union1d(ign, igncand), dtype=np.int)
    ncl = np.setdiff1d(ncl, ign)

    """
    Step 5: Scrub the set
    """

    if len(ncl) > len(good_guess):
        # Recompute the midk steps on the limited set to clean up the tail
        d_table_rank = np.vstack([len(ncl) - stats.rankdata(Kappas[ncl], method='ordinal'),
                                  len(ncl) - stats.rankdata(dice_table[ncl, 0], method='ordinal'),
                                  len(ncl) - stats.rankdata(tt_table[ncl, 0], method='ordinal'),
                                  stats.rankdata(countnoise[ncl], method='ordinal'),
                                  len(ncl) - stats.rankdata(countsigFR2[ncl], method='ordinal')]).T
        d_table_score = d_table_rank.sum(1)
        num_acc_guess = np.mean([np.sum(utils.andb([Kappas[ncl] > Kappas_elbow,
                                                    Rhos[ncl] < Rhos_elbow]) == 2),
                                 np.sum(Kappas[ncl] > Kappas_elbow)])
        conservative_guess = num_acc_guess * d_table_rank.shape[1] / RESTRICT_FACTOR
        candartA = np.intersect1d(ncl[d_table_score > conservative_guess],
                                  ncl[Kappa_ratios[ncl] > EXTEND_FACTOR * 2])
        midkadd = np.union1d(midkadd,
                             np.intersect1d(candartA,
                                            candartA[varex[candartA] > varex_ub * EXTEND_FACTOR]))
        candartB = ncl[d_table_score > num_acc_guess * d_table_rank.shape[1]*HIGH_PERC/100.]
        midkadd = np.union1d(midkadd,
                             np.intersect1d(candartB,
                                            candartB[varex[candartB] > varex_lb * EXTEND_FACTOR]))
        midk = np.union1d(midk, midkadd)
        # Find comps to ignore
        new_varex_lb = stats.scoreatpercentile(varex[ncl[:num_acc_guess]], LOW_PERC)
        candart = np.setdiff1d(ncl[d_table_score > num_acc_guess * d_table_rank.shape[1]], midk)
        ignadd = np.intersect1d(candart, candart[varex[candart] > new_varex_lb])
        ignadd = np.union1d(ignadd, np.intersect1d(ncl[Kappas[ncl] <= Kappas_elbow],
                                                   ncl[varex[ncl] > new_varex_lb]))
        ign = np.setdiff1d(np.union1d(ign, ignadd), midk)
        ncl = np.setdiff1d(ncl, np.union1d(midk, ign))

    return list(sorted(ncl)), list(sorted(rej)), list(sorted(midk)), list(sorted(ign))
