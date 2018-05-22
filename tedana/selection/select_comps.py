"""
Functions to identify TE-dependent and TE-independent components.
"""
import json
import logging
import pickle

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN

from tedana import utils
from tedana.selection._utils import (getelbow_cons, getelbow_mod,
                                     getelbow_aggr, do_svm)

LGR = logging.getLogger(__name__)


def selcomps(seldict, mmix, mask, ref_img, manacc, n_echos, t2s, s0, olevel=2,
             oversion=99, filecsdata=True, savecsdiag=True, strict_mode=False):
    """
    Labels components in `mmix`

    Parameters
    ----------
    seldict : :obj:`dict`
        As output from `fitmodels_direct`
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the number of volumes in the original data
    mask : (S,) array_like
        Boolean mask array
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    manacc : list
        Comma-separated list of indices of manually accepted components
    n_echos : int
        Number of echos in original data
    t2s : (S,) array_like
    s0 : (S,) array_like
    olevel : int, optional
        Default: 2
    oversion : int, optional
        Default: 99
    filecsdata: bool, optional
        Default: False
    savecsdiag: bool, optional
        Default: True
    strict_mode: bool, optional
        Default: False

    Returns
    -------
    acc : list
        Indices of accepted (BOLD) components in `mmix`
    rej : list
        Indices of rejected (non-BOLD) components in `mmix`
    midk : list
        Indices of mid-K (questionable) components in `mmix`
    ign : list
        Indices of ignored components in `mmix`
    """

    if filecsdata:
        import bz2
        if seldict is not None:
            LGR.info('Saving component selection data')
            with bz2.BZ2File('compseldata.pklbz', 'wb') as csstate_f:
                pickle.dump(seldict, csstate_f)
        else:
            try:
                with bz2.BZ2File('compseldata.pklbz', 'rb') as csstate_f:
                    seldict = pickle.load(csstate_f)
            except FileNotFoundError:
                LGR.warning('Failed to load component selection data')
                return None

    # List of components
    midk = []
    ign = []
    nc = np.arange(len(seldict['Kappas']))
    ncl = np.arange(len(seldict['Kappas']))

    # If user has specified components to accept manually
    if manacc:
        acc = sorted([int(vv) for vv in manacc.split(',')])
        midk = []
        rej = sorted(np.setdiff1d(ncl, acc))
        return acc, rej, midk, []  # Add string for ign

    """
    Do some tallies for no. of significant voxels
    """
    countsigFS0 = seldict['F_S0_clmaps'].sum(0)
    countsigFR2 = seldict['F_R2_clmaps'].sum(0)
    countnoise = np.zeros(len(nc))

    """
    Make table of dice values
    """
    dice_tbl = np.zeros([nc.shape[0], 2])
    for ii in ncl:
        dice_FR2 = utils.dice(utils.unmask(seldict['Br_clmaps_R2'][:, ii], mask)[t2s != 0],
                              seldict['F_R2_clmaps'][:, ii])
        dice_FS0 = utils.dice(utils.unmask(seldict['Br_clmaps_S0'][:, ii], mask)[t2s != 0],
                              seldict['F_S0_clmaps'][:, ii])
        dice_tbl[ii, :] = [dice_FR2, dice_FS0]  # step 3a here and above
    dice_tbl[np.isnan(dice_tbl)] = 0

    """
    Make table of noise gain
    """
    tt_table = np.zeros([len(nc), 4])
    counts_FR2_Z = np.zeros([len(nc), 2])
    for ii in nc:
        comp_noise_sel = utils.andb([np.abs(seldict['Z_maps'][:, ii]) > 1.95,
                                     seldict['Z_clmaps'][:, ii] == 0]) == 2
        countnoise[ii] = np.array(comp_noise_sel, dtype=np.int).sum()
        noise_FR2_Z_mask = utils.unmask(comp_noise_sel, mask)[t2s != 0]
        noise_FR2_Z = np.log10(np.unique(seldict['F_R2_maps'][noise_FR2_Z_mask, ii]))
        signal_FR2_Z_mask = utils.unmask(seldict['Z_clmaps'][:, ii], mask)[t2s != 0] == 1
        signal_FR2_Z = np.log10(np.unique(seldict['F_R2_maps'][signal_FR2_Z_mask, ii]))
        counts_FR2_Z[ii, :] = [len(signal_FR2_Z), len(noise_FR2_Z)]
        try:
            ttest = stats.ttest_ind(signal_FR2_Z, noise_FR2_Z, equal_var=True)
            # avoid DivideByZero RuntimeWarning
            if signal_FR2_Z.size > 0 and noise_FR2_Z.size > 0:
                mwu = stats.norm.ppf(stats.mannwhitneyu(signal_FR2_Z, noise_FR2_Z)[1])
            else:
                mwu = -np.inf
            tt_table[ii, 0] = np.abs(mwu) * ttest[0] / np.abs(ttest[0])
            tt_table[ii, 1] = ttest[1]
        except Exception:  # TODO: what is the error that might be caught here?
            pass
    tt_table[np.isnan(tt_table)] = 0
    tt_table[np.isinf(tt_table[:, 0]), 0] = np.percentile(tt_table[~np.isinf(tt_table[:, 0]), 0],
                                                          98)

    # Time series derivative kurtosis
    mmix_dt = (mmix[:-1] - mmix[1:])
    mmix_kurt = stats.kurtosis(mmix_dt)
    mmix_std = np.std(mmix_dt, axis=0)

    """
    Step 1: Reject anything that's obviously an artifact
    a. Estimate a null variance
    """
    LGR.debug('Rejecting gross artifacts based on Rho/Kappa values and S0/R2 counts')
    rej = ncl[utils.andb([seldict['Rhos'] > seldict['Kappas'], countsigFS0 > countsigFR2]) > 0]
    ncl = np.setdiff1d(ncl, rej)

    """
    Step 2: Compute 3-D spatial FFT of Beta maps to detect high-spatial
    frequency artifacts
    """
    LGR.debug('Computing 3D spatial FFT of beta maps to detect high-spatial frequency artifacts')
    # spatial information is important so for NIFTI we convert back to 3D space
    if utils.get_dtype(ref_img) == 'NIFTI':
        dim1 = np.prod(ref_img.shape[:2])
    else:
        dim1 = mask.shape[0]
    fproj_arr = np.zeros([dim1, len(nc)])
    fproj_arr_val = np.zeros([dim1, len(nc)])
    spr = []
    fdist = []
    for ii in nc:
        # convert data back to 3D array
        if utils.get_dtype(ref_img) == 'NIFTI':
            tproj = utils.new_nii_like(ref_img, utils.unmask(seldict['PSC'],
                                                             mask)[:, ii]).get_data()
        else:
            tproj = utils.unmask(seldict['PSC'], mask)[:, ii]
        fproj = np.fft.fftshift(np.abs(np.fft.rfftn(tproj)))
        fproj_z = fproj.max(axis=-1)
        fproj[fproj == fproj.max()] = 0
        spr.append(np.array(fproj_z > fproj_z.max() / 4, dtype=np.int).sum())
        fproj_arr[:, ii] = stats.rankdata(fproj_z.flatten())
        fproj_arr_val[:, ii] = fproj_z.flatten()
        if utils.get_dtype(ref_img) == 'NIFTI':
            fprojr = np.array([fproj, fproj[:, :, ::-1]]).max(0)
        else:
            fprojr = np.array([fproj, fproj[::-1]]).max(0)
        fdist.append(np.max([utils.fitgaussian(fproj.max(jj))[3:].max() for
                     jj in range(fprojr.ndim)]))
    fdist = np.array(fdist)
    spr = np.array(spr)

    """
    Step 3: Create feature space of component properties
    """
    LGR.debug('Creating feature space of component properties')
    fdist_pre = fdist.copy()
    fdist_pre[fdist > np.median(fdist) * 3] = np.median(fdist) * 3
    fdist_z = (fdist_pre - np.median(fdist_pre)) / fdist_pre.std()
    spz = (spr-spr.mean())/spr.std()
    Tz = (tt_table[:, 0] - tt_table[:, 0].mean()) / tt_table[:, 0].std()
    varex_ = np.log(seldict['varex'])
    Vz = (varex_-varex_.mean()) / varex_.std()
    Rz = (seldict['Rhos'] - seldict['Rhos'].mean()) / seldict['Rhos'].std()
    Ktz = np.log(seldict['Kappas']) / 2
    Ktz = (Ktz-Ktz.mean()) / Ktz.std()
    Rtz = np.log(seldict['Rhos']) / 2
    Rtz = (Rtz-Rtz.mean())/Rtz.std()
    KRr = stats.zscore(np.log(seldict['Kappas']) / np.log(seldict['Rhos']))
    cnz = (countnoise-countnoise.mean()) / countnoise.std()
    Dz = stats.zscore(np.arctanh(dice_tbl[:, 0] + 0.001))
    fz = np.array([Tz, Vz, Ktz, KRr, cnz, Rz, mmix_kurt, fdist_z])

    """
    Step 3: Make initial guess of where BOLD components are and use DBSCAN
    to exclude noise components and find a sample set of 'good' components
    """
    LGR.debug('Making initial guess of BOLD components')
    # epsmap is [index,level of overlap with dicemask,
    # number of high Rho components]
    F05, F025, F01 = utils.getfbounds(n_echos)
    epsmap = []
    Rhos_sorted = np.array(sorted(seldict['Rhos']))[::-1]
    # Make an initial guess as to number of good components based on
    # consensus of control points across Rhos and Kappas
    KRcutguesses = [getelbow_mod(seldict['Rhos']), getelbow_cons(seldict['Rhos']),
                    getelbow_aggr(seldict['Rhos']), getelbow_mod(seldict['Kappas']),
                    getelbow_cons(seldict['Kappas']), getelbow_aggr(seldict['Kappas'])]
    Khighelbowval = stats.scoreatpercentile([getelbow_mod(seldict['Kappas'], val=True),
                                             getelbow_cons(seldict['Kappas'], val=True),
                                             getelbow_aggr(seldict['Kappas'], val=True)] +
                                            list(utils.getfbounds(n_echos)),
                                            75, interpolation_method='lower')
    KRcut = np.median(KRcutguesses)

    # only use exclusive when inclusive is extremely inclusive - double KRcut
    cond1 = getelbow_cons(seldict['Kappas']) > KRcut * 2
    cond2 = getelbow_mod(seldict['Kappas'], val=True) < F01
    if cond1 and cond2:
        Kcut = getelbow_mod(seldict['Kappas'], val=True)
    else:
        Kcut = getelbow_cons(seldict['Kappas'], val=True)
    # only use inclusive when exclusive is extremely exclusive - half KRcut
    # (remember for Rho inclusive is higher, so want both Kappa and Rho
    # to defaut to lower)
    if getelbow_cons(seldict['Rhos']) > KRcut * 2:
        Rcut = getelbow_mod(seldict['Rhos'], val=True)
    # for above, consider something like:
    # min([getelbow_mod(Rhos,True),sorted(Rhos)[::-1][KRguess] ])
    else:
        Rcut = getelbow_cons(seldict['Rhos'], val=True)
    if Rcut > Kcut:
        Kcut = Rcut  # Rcut should never be higher than Kcut
    KRelbow = utils.andb([seldict['Kappas'] > Kcut, seldict['Rhos'] < Rcut])
    # Make guess of Kundu et al 2011 plus remove high frequencies,
    # generally high variance, and high variance given low Kappa
    tt_lim = stats.scoreatpercentile(tt_table[tt_table[:, 0] > 0, 0],
                                     75, interpolation_method='lower') / 3
    KRguess = np.setdiff1d(np.setdiff1d(nc[KRelbow == 2], rej),
                           np.union1d(nc[tt_table[:, 0] < tt_lim],
                           np.union1d(np.union1d(nc[spz > 1],
                                                 nc[Vz > 2]),
                                      nc[utils.andb([seldict['varex'] > 0.5 *
                                         sorted(seldict['varex'])[::-1][int(KRcut)],
                                                seldict['Kappas'] < 2*Kcut]) == 2])))
    guessmask = np.zeros(len(nc))
    guessmask[KRguess] = 1

    # Throw lower-risk bad components out
    rejB = ncl[utils.andb([tt_table[ncl, 0] < 0,
                           seldict['varex'][ncl] > np.median(seldict['varex']), ncl > KRcut]) == 3]
    rej = np.union1d(rej, rejB)
    ncl = np.setdiff1d(ncl, rej)

    LGR.debug('Using DBSCAN to find optimal set of "good" BOLD components')
    for ii in range(20000):
        eps = .005 + ii * .005
        db = DBSCAN(eps=eps, min_samples=3).fit(fz.T)

        # it would be great to have descriptive names, here
        # DBSCAN found at least three non-noisy clusters
        cond1 = db.labels_.max() > 1
        # DBSCAN didn't detect more classes than the total # of components / 6
        cond2 = db.labels_.max() < len(nc) / 6
        # TODO: confirm if 0 is a special label for DBSCAN
        # my intuition here is that we're confirming DBSCAN labelled previously
        # rejected components as noise (i.e., no overlap between `rej` and
        # labelled DBSCAN components)
        cond3 = np.intersect1d(rej, nc[db.labels_ == 0]).shape[0] == 0
        # DBSCAN labelled less than half of the total components as noisy
        cond4 = np.array(db.labels_ == -1, dtype=int).sum() / float(len(nc)) < .5

        if cond1 and cond2 and cond3 and cond4:
            epsmap.append([ii, utils.dice(guessmask, db.labels_ == 0),
                           np.intersect1d(nc[db.labels_ == 0],
                           nc[seldict['Rhos'] > getelbow_mod(Rhos_sorted,
                                                             val=True)]).shape[0]])
        db = None

    epsmap = np.array(epsmap)
    LGR.debug('Found DBSCAN solutions for {}/20000 eps resolutions'.format(len(epsmap)))
    group0 = []
    dbscanfailed = False
    if len(epsmap) != 0:
        # Select index that maximizes Dice with guessmask but first
        # minimizes number of higher Rho components
        ii = int(epsmap[np.argmax(epsmap[epsmap[:, 2] == np.min(epsmap[:, 2]), 1], 0), 0])
        LGR.debug('Component selection tuning: {:.05f}'.format(epsmap[:, 1].max()))
        db = DBSCAN(eps=.005+ii*.005, min_samples=3).fit(fz.T)
        ncl = nc[db.labels_ == 0]
        ncl = np.setdiff1d(ncl, rej)
        ncl = np.setdiff1d(ncl, ncl[ncl > len(nc) - len(rej)])
        group0 = ncl.copy()
        group_n1 = nc[db.labels_ == -1]
        to_clf = np.setdiff1d(nc, np.union1d(ncl, rej))
    if len(group0) == 0 or len(group0) < len(KRguess) * .5:
        dbscanfailed = True
        LGR.debug('DBSCAN guess failed; using elbow guess method instead')
        ncl = np.setdiff1d(np.setdiff1d(nc[KRelbow == 2], rej),
                           np.union1d(nc[tt_table[:, 0] < tt_lim],
                           np.union1d(np.union1d(nc[spz > 1],
                                      nc[Vz > 2]),
                                      nc[utils.andb([seldict['varex'] > 0.5 *
                                                     sorted(seldict['varex'])[::-1][int(KRcut)],
                                                     seldict['Kappas'] < 2 * Kcut]) == 2])))
        group0 = ncl.copy()
        group_n1 = []
        to_clf = np.setdiff1d(nc, np.union1d(group0, rej))
    if len(group0) < 2 or (len(group0) < 4 and float(len(rej))/len(group0) > 3):
        LGR.warning('Extremely limited reliable BOLD signal space! '
                    'Not filtering components beyond BOLD/non-BOLD guesses.')
        midkfailed = True
        min_acc = np.array([])
        if len(group0) != 0:
            # For extremes, building in a 20% tolerance
            toacc_hi = np.setdiff1d(nc[utils.andb([fdist <= np.max(fdist[group0]),
                                                   seldict['Rhos'] < F025, Vz > -2]) == 3],
                                    np.union1d(group0, rej))
            min_acc = np.union1d(group0, toacc_hi)
            to_clf = np.setdiff1d(nc, np.union1d(min_acc, rej))
        diagstep_keys = ['Rejected components', 'Kappa-Rho cut point',
                         'Kappa cut point', 'Rho cut point', 'DBSCAN failed to converge',
                         'Mid-Kappa failed (limited BOLD signal)', 'Kappa-Rho guess',
                         'min_acc', 'toacc_hi']
        diagstep_vals = [rej.tolist(), KRcut, Kcut, Rcut, dbscanfailed,
                         midkfailed, KRguess.tolist(), min_acc.tolist(), toacc_hi.tolist()]

        with open('csstepdata.json', 'w') as ofh:
            json.dump(dict(zip(diagstep_keys, diagstep_vals)), ofh, indent=4, sort_keys=True)
        return list(sorted(min_acc)), list(sorted(rej)), [], list(sorted(to_clf))

    # Find additional components to reject based on Dice - doing this here
    # since Dice is a little unstable, need to reference group0
    rej_supp = []
    dice_rej = False
    if not dbscanfailed and len(rej) + len(group0) < 0.75 * len(nc):
        dice_rej = True
        rej_supp = np.setdiff1d(np.setdiff1d(np.union1d(rej,
                                                        nc[dice_tbl[nc, 0] <= dice_tbl[nc, 1]]),
                                             group0), group_n1)
        rej = np.union1d(rej, rej_supp)

    # Temporal features
    # larger is worse - spike
    mmix_kurt_z = (mmix_kurt-mmix_kurt[group0].mean()) / mmix_kurt[group0].std()
    # smaller is worse - drift
    mmix_std_z = -1 * ((mmix_std-mmix_std[group0].mean()) / mmix_std[group0].std())
    mmix_kurt_z_max = np.max([mmix_kurt_z, mmix_std_z], 0)

    """
    Step 2: Classifiy midk and ignore using separte SVMs for
    different variance regimes
    # To render hyperplane:
    min_x = np.min(spz2);max_x=np.max(spz2)
    # plotting separating hyperplane
        ww = clf_.coef_[0]
        aa = -ww[0] / ww[1]
        # make sure the next line is long enough
        xx = np.linspace(min_x - 2, max_x + 2)
        yy = aa * xx - (clf_.intercept_[0]) / ww[1]
        plt.plot(xx, yy, '-')
    """
    LGR.debug('Attempting to classify midk components')
    # Tried getting rid of accepting based on SVM altogether,
    # now using only rejecting
    toacc_hi = np.setdiff1d(nc[utils.andb([fdist <= np.max(fdist[group0]),
                               seldict['Rhos'] < F025, Vz > -2]) == 3],
                            np.union1d(group0, rej))
    toacc_lo = np.intersect1d(to_clf,
                              nc[utils.andb([spz < 1, Rz < 0, mmix_kurt_z_max < 5,
                                             Dz > -1, Tz > -1, Vz < 0, seldict['Kappas'] >= F025,
                                             fdist < 3 * np.percentile(fdist[group0], 98)]) == 8])
    midk_clf, clf_ = do_svm(fproj_arr_val[:, np.union1d(group0, rej)].T,
                            [0] * len(group0) + [1] * len(rej),
                            fproj_arr_val[:, to_clf].T,
                            svmtype=2)
    midk = np.setdiff1d(to_clf[utils.andb([midk_clf == 1, seldict['varex'][to_clf] >
                                           np.median(seldict['varex'][group0])]) == 2],
                        np.union1d(toacc_hi, toacc_lo))
    # only use SVM to augment toacc_hi only if toacc_hi isn't already
    # conflicting with SVM choice
    if len(np.intersect1d(to_clf[utils.andb([midk_clf == 1,
                                             Vz[to_clf] > 0]) == 2], toacc_hi)) == 0:
        svm_acc_fail = True
        toacc_hi = np.union1d(toacc_hi, to_clf[midk_clf == 0])
    else:
        svm_acc_fail = False

    """
    Step 3: Compute variance associated with low T2* areas
    (e.g. draining veins and low T2* areas)
    # To write out veinmask
    veinout = np.zeros(t2s.shape)
    veinout[t2s!=0] = veinmaskf
    utils.filewrite(veinout, 'veinmaskf', ref_img)
    veinBout = utils.unmask(veinmaskB, mask)
    utils.filewrite(veinBout, 'veins50', ref_img)
    """
    LGR.debug('Computing variance associated with low T2* areas (e.g., draining veins)')
    tsoc_B_Zcl = np.zeros(seldict['tsoc_B'].shape)
    tsoc_B_Zcl[seldict['Z_clmaps'] != 0] = np.abs(seldict['tsoc_B'])[seldict['Z_clmaps'] != 0]
    sig_B = [stats.scoreatpercentile(tsoc_B_Zcl[tsoc_B_Zcl[:, ii] != 0, ii], 25)
             if len(tsoc_B_Zcl[tsoc_B_Zcl[:, ii] != 0, ii]) != 0
             else 0 for ii in nc]
    sig_B = np.abs(seldict['tsoc_B']) > np.tile(sig_B, [seldict['tsoc_B'].shape[0], 1])

    veinmask = utils.andb([t2s < stats.scoreatpercentile(t2s[t2s != 0], 15,
                                                         interpolation_method='lower'),
                           t2s != 0]) == 2
    veinmaskf = veinmask[mask]
    veinR = np.array(sig_B[veinmaskf].sum(0),
                     dtype=float) / sig_B[~veinmaskf].sum(0)
    veinR[np.isnan(veinR)] = 0

    veinc = np.union1d(rej, midk)
    rej_veinRZ = ((veinR-veinR[veinc].mean())/veinR[veinc].std())[veinc]
    rej_veinRZ[rej_veinRZ < 0] = 0
    rej_veinRZ[countsigFR2[veinc] > np.array(veinmaskf, dtype=int).sum()] = 0
    t2s_lim = [stats.scoreatpercentile(t2s[t2s != 0], 50,
                                       interpolation_method='lower'),
               stats.scoreatpercentile(t2s[t2s != 0], 80,
                                       interpolation_method='lower') / 2]
    phys_var_zs = []
    for t2sl_i in range(len(t2s_lim)):
        t2sl = t2s_lim[t2sl_i]
        veinW = sig_B[:, veinc]*np.tile(rej_veinRZ, [sig_B.shape[0], 1])
        veincand = utils.unmask(utils.andb([s0[t2s != 0] < np.median(s0[t2s != 0]),
                                t2s[t2s != 0] < t2sl]) >= 1,
                                t2s != 0)[mask]
        veinW[~veincand] = 0
        invein = veinW.sum(axis=1)[(utils.unmask(veinmaskf, mask) *
                                    utils.unmask(veinW.sum(axis=1) > 1, mask))[mask]]
        minW = 10 * (np.log10(invein).mean()) - 1 * 10**(np.log10(invein).std())
        veinmaskB = veinW.sum(axis=1) > minW
        tsoc_Bp = seldict['tsoc_B'].copy()
        tsoc_Bp[tsoc_Bp < 0] = 0
        vvex = np.array([(tsoc_Bp[veinmaskB, ii]**2.).sum() /
                         (tsoc_Bp[:, ii]**2.).sum() for ii in nc])
        group0_res = np.intersect1d(KRguess, group0)
        phys_var_zs.append((vvex - vvex[group0_res].mean()) / vvex[group0_res].std())
        veinBout = utils.unmask(veinmaskB, mask)
        utils.filewrite(veinBout.astype(float), 'veins_l%i' % t2sl_i, ref_img)

    # Mask to sample veins
    phys_var_z = np.array(phys_var_zs).max(0)
    Vz2 = (varex_ - varex_[group0].mean())/varex_[group0].std()

    """
    Step 4: Learn joint TE-dependence spatial and temporal models to move
    remaining artifacts to ignore class
    """
    LGR.debug('Learning joint TE-dependence spatial/temporal models to ignore remaining artifacts')

    to_ign = []

    minK_ign = np.max([F05, getelbow_cons(seldict['Kappas'], val=True)])
    newcest = len(group0) + len(toacc_hi[seldict['Kappas'][toacc_hi] > minK_ign])
    phys_art = np.setdiff1d(nc[utils.andb([phys_var_z > 3.5,
                                           seldict['Kappas'] < minK_ign]) == 2], group0)
    rank_diff = stats.rankdata(phys_var_z) - stats.rankdata(seldict['Kappas'])
    phys_art = np.union1d(np.setdiff1d(nc[utils.andb([phys_var_z > 2, rank_diff > newcest / 2,
                                                      Vz2 > -1]) == 3],
                                       group0), phys_art)
    # Want to replace field_art with an acf/SVM based approach
    # instead of a kurtosis/filter one
    field_art = np.setdiff1d(nc[utils.andb([mmix_kurt_z_max > 5,
                                            seldict['Kappas'] < minK_ign]) == 2], group0)
    field_art = np.union1d(np.setdiff1d(nc[utils.andb([mmix_kurt_z_max > 2,
                                           (stats.rankdata(mmix_kurt_z_max) -
                                            stats.rankdata(seldict['Kappas'])) > newcest / 2,
                                           Vz2 > 1, seldict['Kappas'] < F01]) == 4],
                                        group0), field_art)
    field_art = np.union1d(np.setdiff1d(nc[utils.andb([mmix_kurt_z_max > 3,
                                                       Vz2 > 3, seldict['Rhos'] >
                                                       np.percentile(seldict['Rhos'][group0],
                                                                     75)]) == 3],
                                        group0), field_art)
    field_art = np.union1d(np.setdiff1d(nc[utils.andb([mmix_kurt_z_max > 5, Vz2 > 5]) == 2],
                                        group0), field_art)
    misc_art = np.setdiff1d(nc[utils.andb([(stats.rankdata(Vz) -
                                            stats.rankdata(Ktz)) > newcest / 2,
                            seldict['Kappas'] < Khighelbowval]) == 2], group0)
    ign_cand = np.unique(list(field_art)+list(phys_art)+list(misc_art))
    midkrej = np.union1d(midk, rej)
    to_ign = np.setdiff1d(list(ign_cand), midkrej)
    toacc = np.union1d(toacc_hi, toacc_lo)
    ncl = np.setdiff1d(np.union1d(ncl, toacc), np.union1d(to_ign, midkrej))
    ign = np.setdiff1d(nc, list(ncl) + list(midk) + list(rej))
    orphan = np.setdiff1d(nc, list(ncl) + list(to_ign) + list(midk) + list(rej))

    # Last ditch effort to save some transient components
    if not strict_mode:
        Vz3 = (varex_ - varex_[ncl].mean())/varex_[ncl].std()
        ncl = np.union1d(ncl, np.intersect1d(orphan,
                                             nc[utils.andb([seldict['Kappas'] > F05,
                                                            seldict['Rhos'] < F025,
                                                            seldict['Kappas'] > seldict['Rhos'],
                                                            Vz3 <= -1,
                                                            Vz3 > -3,
                                                            mmix_kurt_z_max < 2.5]) == 6]))
        ign = np.setdiff1d(nc, list(ncl)+list(midk)+list(rej))
        orphan = np.setdiff1d(nc, list(ncl) + list(to_ign) + list(midk) + list(rej))

    if savecsdiag:
        diagstep_keys = ['Rejected components', 'Kappa-Rho cut point', 'Kappa cut',
                         'Rho cut', 'DBSCAN failed to converge', 'Kappa-Rho guess',
                         'Dice rejected', 'rej_supp', 'to_clf',
                         'Mid-kappa components', 'svm_acc_fail', 'toacc_hi', 'toacc_lo',
                         'Field artifacts', 'Physiological artifacts',
                         'Miscellaneous artifacts', 'ncl', 'Ignored components']
        diagstep_vals = [rej.tolist(), KRcut, Kcut, Rcut, dbscanfailed,
                         KRguess.tolist(), dice_rej, rej_supp.tolist(),
                         to_clf.tolist(), midk.tolist(), svm_acc_fail,
                         toacc_hi.tolist(), toacc_lo.tolist(),
                         field_art.tolist(), phys_art.tolist(),
                         misc_art.tolist(), ncl.tolist(), ign.tolist()]

        with open('csstepdata.json', 'w') as ofh:
            json.dump(dict(zip(diagstep_keys, diagstep_vals)), ofh, indent=4, sort_keys=True)
        allfz = np.array([Tz, Vz, Ktz, KRr, cnz, Rz, mmix_kurt, fdist_z])
        np.savetxt('csdata.txt', allfz)

    return list(sorted(ncl)), list(sorted(rej)), list(sorted(midk)), list(sorted(ign))
