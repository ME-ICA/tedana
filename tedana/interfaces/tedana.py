import os
import os.path as op
import shutil
import pickle
import textwrap
import numpy as np
from scipy import stats
from sklearn import svm
from scipy.special import lpmv
from sklearn.cluster import DBSCAN
from tedana.interfaces import (make_optcom, t2sadmap)
from tedana.utils import (load_image, load_data, get_dtype,
                          make_min_mask, makeadmask,
                          fmask, unmask, filewrite,
                          fitgaussian, dice, andb)

import logging
logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
lgr = logging.getLogger(__name__)

"""
PROCEDURE 2 : Computes ME-PCA and ME-ICA
-Computes T2* map
-Computes PCA of concatenated ME data, then computes TE-dependence of PCs
-Computes ICA of TE-dependence PCs
-Identifies TE-dependent ICs, outputs high-\kappa (BOLD) component
   and denoised time series
-or- Computes TE-dependence of each component of a general linear model
   specified by input (includes MELODIC FastICA mixing matrix)
PROCEDURE 2a: Model fitting and component selection routines
"""

F_MAX = 500
Z_MAX = 8


def do_svm(X_train, y_train, X_test, svmtype=0):
    """
    sklearn's Support Vector Classification (SVC).
    For svmtype=1, implemented in liblinear rather than libsvm.

    Parameters
    ----------
    X_train : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples in the
        training dataset and n_features is the number of features.
    y_train : array-like, shape (n_samples,)
        Target values (class labels in classification, real numbers in
        regression)
    X_test : {array-like, sparse matrix}, shape (n_samples, n_features)
        Test vectors, where n_samples is the number of samples in the test
        dataset and n_features is the number of features.
    svmtype : int
        Desired support vector machine type.

    Returns
    -------
    y_pred : array, shape (n_samples,)
        Predicted class labels for samples in X_test.
    clf : {:obj:`sklearn.svm.classes.SVC`, :obj:`sklearn.svm.classes.LinearSVC`}
        Trained sklearn model instance.
    """

    if svmtype == 0:
        clf = svm.SVC(kernel='linear')
    elif svmtype == 1:
        clf = svm.LinearSVC(loss='squared_hinge', penalty='l1', dual=False)
    elif svmtype == 2:
        clf = svm.SVC(kernel='linear', probability=True)
    else:
        raise ValueError('Input svmtype not in range (3)')

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred, clf


def spatclust(data, mask, csize, thr, ref_img, infile=None, dindex=0,
              tindex=0):
    """
    Thresholds and spatially clusters `data`

    Parameters
    ----------
    data : array_like
    mask :
    csize : int
    thr :
    header :
    aff :
    infile :
    dindex :
    tindex :

    Returns
    -------
    clustered :
    """

    if infile is None:
        data = data.copy()
        data[data < thr] = 0
        infile = filewrite(unmask(data, mask), '__clin', ref_img, gzip=True)

    # FIXME: ideally no calls to os.system!!! (or AFNI, for that matter)
    addopts = ''
    if data is not None and data.squeeze().ndim > 1 and dindex + tindex == 0:
        addopts = '-doall'
    else:
        addopts = '-1dindex {0} -1tindex {1}'.format(str(dindex), str(tindex))

    cmd_str = '3dmerge -overwrite {0} -dxyz=1 -1clust 1 {1:d} ' \
              '-1thresh {2:.02f} -prefix __clout.nii.gz {3}'
    os.system(cmd_str.format(addopts, int(csize), float(thr), infile))

    clustered = load_image('__clout.nii.gz')[mask] != 0
    return clustered


def get_coeffs(data, mask, X, add_const=False):
    """
    get_coeffs(data, X)

    Parameters
    ----------
    data : (S x T) array-like
        Array where `S` is samples and `T` is time
    mask : (S,) array-like
        Boolean mask array
    X : (T x C) array-like
        Array where `T` is time and `C` is components
    add_const : bool, optional
        Add intercept column to `X` before fitting. Default: False

    Returns
    -------
    out : (S x C) np.ndarray
        Array of betas for all samples `S`
    """

    mdata = data[mask].T

    # Coerce X to >=2d
    X = np.atleast_2d(X)

    if X.shape[0] == 1:
        X = X.T
    if add_const:  # add intercept
        Xones = np.ones((np.min(mdata.shape), 1))
        X = np.column_stack([X, Xones])

    tmpbetas = np.linalg.lstsq(X, mdata)[0].T
    if add_const:  # drop beta for intercept
        tmpbetas = tmpbetas[:, :-1]
    out = unmask(tmpbetas, mask)

    return out


def getelbow_cons(ks, val=False):
    """
    Elbow using mean/variance method - conservative

    Parameters
    ----------
    ks : array-like
    val : bool, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    array-like
        Either the elbow index (if val is True) or the values at the elbow
        index (if val is False)
    """

    ks = np.sort(ks)[::-1]
    nk = len(ks)
    temp1 = [(ks[nk - 5 - ii - 1] > ks[nk - 5 - ii:nk].mean() + 2 * ks[nk - 5 - ii:nk].std())
             for ii in range(nk - 5)]
    ds = np.array(temp1[::-1], dtype=np.int)
    dsum = []
    c_ = 0
    for d_ in ds:
        c_ = (c_ + d_) * d_
        dsum.append(c_)
    e2 = np.argmax(np.array(dsum))
    elind = np.max([getelbow_mod(ks), e2])

    if val:
        return ks[elind]
    else:
        return elind


def getelbow_mod(ks, val=False):
    """
    Elbow using linear projection method - moderate

    Parameters
    ----------
    ks : array-like
    val : bool, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    array-like
        Either the elbow index (if val is True) or the values at the elbow
        index (if val is False)
    """

    ks = np.sort(ks)[::-1]
    n_components = ks.shape[0]
    coords = np.array([np.arange(n_components), ks])
    p = coords - coords[:, 0].reshape(2, 1)
    b = p[:, -1]
    b_hat = np.reshape(b / np.sqrt((b ** 2).sum()), (2, 1))
    proj_p_b = p - np.dot(b_hat.T, p) * np.tile(b_hat, (1, n_components))
    d = np.sqrt((proj_p_b ** 2).sum(axis=0))
    k_min_ind = d.argmax()

    if val:
        return ks[k_min_ind]
    else:
        return k_min_ind


def getelbow_aggr(ks, val=False):
    """
    Elbow using curvature - aggressive

    Parameters
    ----------
    ks : array-like

    val : bool, optional
        Default is False

    Returns
    -------
    array-like
        Either the elbow index (if val is True) or the values at the elbow
        index (if val is False)
    """

    ks = np.sort(ks)[::-1]
    dKdt = ks[:-1] - ks[1:]
    dKdt2 = dKdt[:-1] - dKdt[1:]
    curv = np.abs((dKdt2 / (1 + dKdt[:-1]**2.) ** (3. / 2.)))
    curv[np.isnan(curv)] = -1 * 10**6
    maxcurv = np.argmax(curv) + 2

    if val:
        return(ks[maxcurv])
    else:
        return maxcurv


def getfbounds(n_echos):
    """
    Parameters
    ----------
    n_echos : int
        Number of echoes

    Returns
    -------
    fmin, fmid, fmax : float
        Minimum, mid, and max F bounds
    """

    if not isinstance(n_echos, int):
        raise IOError('Input n_echos must be int')
    elif n_echos <= 0:
        raise ValueError('Input n_echos must be greater than 0')
    idx = n_echos - 1

    F05s = [None, None, 18.5, 10.1, 7.7, 6.6, 6.0, 5.6, 5.3, 5.1, 5.0]
    F025s = [None, None, 38.5, 17.4, 12.2, 10, 8.8, 8.1, 7.6, 7.2, 6.9]
    F01s = [None, None, 98.5, 34.1, 21.2, 16.2, 13.8, 12.2, 11.3, 10.7, 10.]
    return F05s[idx], F025s[idx], F01s[idx]


def eimask(dd, ees=None):
    """
    Returns mask for data between [0.001, 5] * 98th percentile of dd
    """

    if ees is None:
        ees = range(dd.shape[1])
    imask = np.zeros([dd.shape[0], len(ees)], dtype=bool)
    for ee in ees:
        lgr.info(ee)
        perc98 = stats.scoreatpercentile(dd[:, ee, :].flatten(), 98,
                                         interpolation_method='lower')
        lthr, hthr = 0.001 * perc98, 5 * perc98
        lgr.info(lthr, hthr)
        m = dd[:, ee, :].mean(axis=1)
        imask[np.logical_and(m > lthr, m < hthr), ee] = True

    return imask


def split_ts(data, comptable, mmix, acc, rej, midk):
    cbetas = get_coeffs(data-data.mean(-1)[:, :, :, np.newaxis], mask, mmix)
    betas = fmask(cbetas, mask)
    if len(acc) != 0:
        hikts = unmask(betas[:, acc].dot(mmix.T[acc, :]), mask)
    else:
        hikts = None
    return hikts, data-hikts


def computefeats2(data, mmix, mask, normalize=True):
    # Write feature versions of components
    data = data[mask]
    # demean data
    data_vn = stats.zscore(data, axis=-1)
    # get betas for demeaned data against `mmix`
    data_R = get_coeffs(unmask(data_vn, mask), mask, mmix)[mask]
    # cap betas to range [-0.999, 0.999]
    data_R[data_R < -0.999] = -0.999
    data_R[data_R > 0.999] = 0.999
    # R-to-Z transform?
    data_Z = np.arctanh(data_R)
    if len(data_Z.shape) == 1:
        data_Z = np.atleast_2d(data_Z).T
    if normalize:
        # standardize
        data_Zm = stats.zscore(data_Z, axis=0)
        # add back (mean / stdev)
        data_Z = data_Zm + (data_Z.mean(axis=0, keepdims=True) / data_Z.std(axis=0, keepdims=True))
    return data_Z


def ctabsel(ctabfile):
    ctlines = open(ctabfile).readlines()
    class_tags = ['#ACC', '#REJ', '#MID', '#IGN']
    class_dict = {}
    for ii, ll in enumerate(ctlines):
        for kk in class_tags:
            if ll[:4] is kk and ll[4:].strip() is not '':
                class_dict[kk] = ll[4:].split('#')[0].split(',')
    return tuple([np.array(class_dict[kk], dtype=int) for kk in class_tags])


def fitmodels_direct(catd, mmix, mask, t2s, t2sG, tes, combmode, ref_img,
                     fout=None, reindex=False, mmixN=None, full_sel=True):
    """
    Parameters
    ----------
    catd : (S x E x T) array_like
    mmix : (T x C) array_like
    mask : (S,) array_like
    t2s : (S,) array_like
    t2sG : (S,) array_like
    tes : (E,) list
    combmode : str
    ref_img : str or img_like
    fout : bool
        Whether to output per-component TE-dependencen maps Default: None
    reindex : bool, optional
        Default: False
    mmixN : array_like, optional
        Default: None
    full_sel : bool, optional
        Default: True
    """

    # compute optimal combination of raw data
    tsoc = np.array(make_optcom(catd, t2sG, tes, mask, combmode),
                    dtype=float)[mask]
    # demean optimal combination
    tsoc_dm = tsoc - tsoc.mean(axis=-1, keepdims=True)

    # compute un-normalized weight dataset (features)
    if mmixN is None:
        mmixN = mmix
    WTS = computefeats2(unmask(tsoc, mask), mmixN, mask, normalize=False)

    # compute PSC dataset - shouldn't have to refit data
    global tsoc_B
    tsoc_B = get_coeffs(unmask(tsoc_dm, mask), mask, mmix)[mask]
    tsoc_Babs = np.abs(tsoc_B)
    PSC = tsoc_B / tsoc.mean(axis=-1, keepdims=True) * 100

    # compute skews to determine signs based on unnormalized weights,
    # correct mmix & WTS signs based on spatial distribution tails
    signs = stats.skew(WTS, axis=0)
    signs /= np.abs(signs)
    mmix = mmix.copy()
    mmix *= signs
    WTS *= signs
    PSC *= signs
    totvar = (tsoc_B**2).sum()
    totvar_norm = (WTS**2).sum()

    # compute Betas and means over TEs for TE-dependence analysis
    betas = get_coeffs(catd, np.repeat(mask[:, np.newaxis], len(tes), axis=1), mmix)
    n_samp, n_echos, n_components = betas.shape
    n_voxels = mask.sum()
    n_data_voxels = (t2s != 0).sum()
    mu = catd.mean(axis=-1)
    tes = np.reshape(tes, (n_echos, 1))
    fmin, fmid, fmax = getfbounds(n_echos)

    # mask arrays
    mumask = mu[t2s != 0]
    t2smask = t2s[t2s != 0]
    betamask = betas[t2s != 0]

    # set up Xmats
    X1 = mumask.T  # Model 1
    X2 = np.tile(tes, (1, n_data_voxels)) * mumask.T / t2smask.T  # Model 2

    # tables for component selection
    global Kappas, Rhos, varex, varex_norm
    global Z_maps, F_R2_maps, F_S0_maps
    global Z_clmaps, F_R2_clmaps, F_S0_clmaps
    global Br_clmaps_R2, Br_clmaps_S0
    Kappas = np.zeros([n_components])
    Rhos = np.zeros([n_components])
    varex = np.zeros([n_components])
    varex_norm = np.zeros([n_components])
    Z_maps = np.zeros([n_voxels, n_components])
    F_R2_maps = np.zeros([n_data_voxels, n_components])
    F_S0_maps = np.zeros([n_data_voxels, n_components])
    Z_clmaps = np.zeros([n_voxels, n_components])
    F_R2_clmaps = np.zeros([n_data_voxels, n_components])
    F_S0_clmaps = np.zeros([n_data_voxels, n_components])
    Br_clmaps_R2 = np.zeros([n_voxels, n_components])
    Br_clmaps_S0 = np.zeros([n_voxels, n_components])

    for i in range(n_components):
        # size of B is (n_components, nx*ny*nz)
        B = np.atleast_3d(betamask)[:, :, i].T
        alpha = (np.abs(B)**2).sum(axis=0)
        varex[i] = (tsoc_B[:, i]**2).sum() / totvar * 100.
        varex_norm[i] = (unmask(WTS, mask)[t2s != 0][:, i]**2).sum() / totvar_norm * 100.

        # S0 Model
        coeffs_S0 = (B * X1).sum(axis=0) / (X1**2).sum(axis=0)
        SSE_S0 = (B - X1 * np.tile(coeffs_S0, (n_echos, 1)))**2
        SSE_S0 = SSE_S0.sum(axis=0)
        F_S0 = (alpha - SSE_S0) * 2 / (SSE_S0)
        F_S0_maps[:, i] = F_S0

        # R2 Model
        coeffs_R2 = (B * X2).sum(axis=0) / (X2**2).sum(axis=0)
        SSE_R2 = (B - X2 * np.tile(coeffs_R2, (n_echos, 1)))**2
        SSE_R2 = SSE_R2.sum(axis=0)
        F_R2 = (alpha - SSE_R2) * 2 / (SSE_R2)
        F_R2_maps[:, i] = F_R2

        # compute weights as Z-values
        wtsZ = (WTS[:, i] - WTS[:, i].mean()) / WTS[:, i].std()
        wtsZ[np.abs(wtsZ) > Z_MAX] = (Z_MAX * (np.abs(wtsZ) / wtsZ))[np.abs(wtsZ) > Z_MAX]
        Z_maps[:, i] = wtsZ

        # compute Kappa and Rho
        F_S0[F_S0 > F_MAX] = F_MAX
        F_R2[F_R2 > F_MAX] = F_MAX
        norm_weights = np.abs(np.squeeze(unmask(wtsZ, mask)[t2s != 0]**2.))
        Kappas[i] = np.average(F_R2, weights=norm_weights)
        Rhos[i] = np.average(F_S0, weights=norm_weights)

    # tabulate component values
    comptab_pre = np.vstack([np.arange(n_components), Kappas, Rhos, varex, varex_norm]).T
    if reindex:
        # re-index all components in Kappa order
        comptab = comptab_pre[comptab_pre[:, 1].argsort()[::-1], :]
        Kappas = comptab[:, 1]
        Rhos = comptab[:, 2]
        varex = comptab[:, 3]
        varex_norm = comptab[:, 4]
        nnc = np.array(comptab[:, 0], dtype=np.int)
        mmix_new = mmix[:, nnc]
        F_S0_maps = F_S0_maps[:, nnc]
        F_R2_maps = F_R2_maps[:, nnc]
        Z_maps = Z_maps[:, nnc]
        WTS = WTS[:, nnc]
        PSC = PSC[:, nnc]
        tsoc_B = tsoc_B[:, nnc]
        tsoc_Babs = tsoc_Babs[:, nnc]
        comptab[:, 0] = np.arange(comptab.shape[0])
    else:
        comptab = comptab_pre
        mmix_new = mmix

    # full selection including clustering criteria
    seldict = None
    if full_sel:
        for i in range(n_components):
            # save out files
            out = np.zeros((n_samp, 4))
            if fout is not None:
                ccname, gzip = 'cc{:03d}'.format(i), False
            else:
                ccname, gzip = '.cc_temp', True

            out[:, 0] = np.squeeze(unmask(PSC[:, i], mask))
            out[:, 1] = np.squeeze(unmask(F_R2_maps[:, i], t2s != 0))
            out[:, 2] = np.squeeze(unmask(F_S0_maps[:, i], t2s != 0))
            out[:, 3] = np.squeeze(unmask(Z_maps[:, i], mask))

            filewrite(out, ccname, ref_img, gzip=gzip)

            if get_dtype(ref_img) == 'GIFTI':
                continue  # TODO: pass through GIFTI file data as below

            os.system('3drefit -sublabel 0 PSC -sublabel 1 F_R2 -sublabel 2 F_SO '
                      '-sublabel 3 Z_sn {} 2> /dev/null > /dev/null'.format(ccname))

            csize = np.max([int(n_voxels * 0.0005) + 5, 20])

            # Do simple clustering on F
            # TODO: can be replaced with nilearn.image.threshold_img
            os.system('3dcalc -overwrite -a {}[1..2] -expr \'a*step(a-{})\' -prefix '
                      '.fcl_in.nii.gz -overwrite'.format(ccname, fmin))
            # TODO: can be replaced with nilearn.regions.connected_regions
            os.system('3dmerge -overwrite -dxyz=1 -1clust 1 {} -doall '
                      '-prefix .fcl_out.nii.gz .fcl_in.nii.gz'.format(csize))
            sel = load_image('.fcl_out.nii.gz')[t2s != 0]
            sel = np.array(sel != 0, dtype=np.int)
            F_R2_clmaps[:, i] = sel[:, 0]
            F_S0_clmaps[:, i] = sel[:, 1]

            # Do simple clustering on Z at p<0.05
            sel = spatclust(None, mask, csize, 1.95, ref_img,
                            infile=ccname, dindex=3, tindex=3)
            Z_clmaps[:, i] = sel

            # Do simple clustering on ranked signal-change map
            countsigFR2 = F_R2_clmaps[:, i].sum()
            countsigFS0 = F_S0_clmaps[:, i].sum()
            Br_clmaps_R2[:, i] = spatclust(stats.rankdata(tsoc_Babs[:, i]), mask,
                                           csize, max(tsoc_Babs.shape)-countsigFR2, ref_img)
            Br_clmaps_S0[:, i] = spatclust(stats.rankdata(tsoc_Babs[:, i]), mask,
                                           csize, max(tsoc_Babs.shape)-countsigFS0,
                                           ref_img)

        seldict = {}
        selvars = ['Kappas', 'Rhos', 'WTS', 'varex', 'Z_maps', 'F_R2_maps',
                   'F_S0_maps', 'Z_clmaps', 'F_R2_clmaps', 'F_S0_clmaps',
                   'tsoc_B', 'Br_clmaps_R2', 'Br_clmaps_S0', 'PSC']
        for vv in selvars:
            seldict[vv] = eval(vv)

    return seldict, comptab, betas, mmix_new


def selcomps(seldict, mmix, head, manacc, n_echos, debug=False, olevel=2, oversion=99,
             filecsdata=False, savecsdiag=True,
             strict_mode=False):
    """
    Select components.
    """
    if filecsdata:
        import bz2
        if seldict is not None:
            lgr.info('Saving component selection data')
            csstate_f = bz2.BZ2File('compseldata.pklbz', 'wb')
            pickle.dump(seldict, csstate_f)
            csstate_f.close()
        else:
            try:
                csstate_f = bz2.BZ2File('compseldata.pklbz', 'rb')
                seldict = pickle.load(csstate_f)
                csstate_f.close()
            except FileNotFoundError:
                lgr.info('No component data found!')
                return None

    # Dump dictionary into variable names
    for key in seldict.keys():
        exec("%s=seldict['%s']" % (key, key))

    # List of components
    midk = []
    ign = []
    nc = np.arange(len(Kappas))
    ncl = np.arange(len(Kappas))

    # If user has specified components to accept manually
    if manacc:
        acc = sorted([int(vv) for vv in manacc.split(',')])
        midk = []
        rej = sorted(np.setdiff1d(ncl, acc))
        return acc, rej, midk, []  # Add string for ign

    """
    Do some tallies for no. of significant voxels
    """
    countsigFS0 = F_S0_clmaps.sum(0)
    countsigFR2 = F_R2_clmaps.sum(0)
    countnoise = np.zeros(len(nc))

    """
    Make table of dice values
    """
    dice_tbl = np.zeros([nc.shape[0], 2])
    for ii in ncl:
        dice_FR2 = dice(unmask(Br_clmaps_R2[:, ii], mask)[t2s != 0],
                        F_R2_clmaps[:, ii])
        dice_FS0 = dice(unmask(Br_clmaps_S0[:, ii], mask)[t2s != 0],
                        F_S0_clmaps[:, ii])
        dice_tbl[ii, :] = [dice_FR2, dice_FS0]  # step 3a here and above
    dice_tbl[np.isnan(dice_tbl)] = 0

    """
    Make table of noise gain
    """
    tt_table = np.zeros([len(nc), 4])
    counts_FR2_Z = np.zeros([len(nc), 2])
    for ii in nc:
        comp_noise_sel = andb([np.abs(Z_maps[:, ii]) > 1.95,
                               Z_clmaps[:, ii] == 0]) == 2
        countnoise[ii] = np.array(comp_noise_sel, dtype=np.int).sum()
        noise_FR2_Z = np.log10(np.unique(F_R2_maps[unmask(comp_noise_sel, mask)[t2s != 0], ii]))
        signal_FR2_Z = np.log10(np.unique(F_R2_maps[unmask(Z_clmaps[:, ii],
                                                           mask)[t2s != 0] == 1, ii]))
        counts_FR2_Z[ii, :] = [len(signal_FR2_Z), len(noise_FR2_Z)]
        try:
            ttest = stats.ttest_ind(signal_FR2_Z, noise_FR2_Z, equal_var=True)
            mwu = stats.norm.ppf(stats.mannwhitneyu(signal_FR2_Z, noise_FR2_Z)[1])
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
    mmix_std = np.std(mmix_dt, 0)

    """
    Step 1: Reject anything that's obviously an artifact
    a. Estimate a null variance
    """

    rej = ncl[andb([Rhos > Kappas, countsigFS0 > countsigFR2]) > 0]
    ncl = np.setdiff1d(ncl, rej)

    """
    Step 2: Compute 3-D spatial FFT of Beta maps to detect high-spatial
    frequency artifacts
    """
    fproj_arr = np.zeros([np.prod(mask.shape[0:2]), len(nc)])
    fproj_arr_val = np.zeros([np.prod(mask.shape[0:2]), len(nc)])
    spr = []
    fdist = []
    for ii in nc:
        fproj = np.fft.fftshift(np.abs(np.fft.rfftn(unmask(seldict['PSC'],
                                                           mask)[:, :, :, ii])))
        fproj_z = fproj.max(2)
        fproj[fproj == fproj.max()] = 0
        fproj_arr[:, ii] = stats.rankdata(fproj_z.flatten())
        fproj_arr_val[:, ii] = fproj_z.flatten()
        spr.append(np.array(fproj_z > fproj_z.max() / 4, dtype=np.int).sum())
        fprojr = np.array([fproj, fproj[:, :, ::-1]]).max(0)
        fdist.append(np.max([fitgaussian(fproj.max(jj))[3:].max() for
                     jj in range(len(fprojr.shape))]))
    fdist = np.array(fdist)
    spr = np.array(spr)

    """
    Step 3: Create feature space of component properties
    """
    fdist_pre = fdist.copy()
    fdist_pre[fdist > np.median(fdist) * 3] = np.median(fdist) * 3
    fdist_z = (fdist_pre - np.median(fdist_pre)) / fdist_pre.std()
    spz = (spr-spr.mean())/spr.std()
    Tz = (tt_table[:, 0] - tt_table[:, 0].mean()) / tt_table[:, 0].std()
    varex_ = np.log(varex)
    Vz = (varex_-varex_.mean()) / varex_.std()
    Rz = (Rhos-Rhos.mean()) / Rhos.std()
    Ktz = np.log(Kappas) / 2
    Ktz = (Ktz-Ktz.mean()) / Ktz.std()
    Rtz = np.log(Rhos) / 2
    Rtz = (Rtz-Rtz.mean())/Rtz.std()
    KRr = stats.zscore(np.log(Kappas) / np.log(Rhos))
    cnz = (countnoise-countnoise.mean()) / countnoise.std()
    Dz = stats.zscore(np.arctanh(dice_tbl[:, 0] + 0.001))
    fz = np.array([Tz, Vz, Ktz, KRr, cnz, Rz, mmix_kurt, fdist_z])

    """
    Step 3: Make initial guess of where BOLD components are and use DBSCAN
    to exclude noise components and find a sample set of 'good' components
    """
    # epsmap is [index,level of overlap with dicemask,
    # number of high Rho components]
    F05, F025, F01 = getfbounds(n_echos)
    epsmap = []
    Rhos_sorted = np.array(sorted(Rhos))[::-1]
    # Make an initial guess as to number of good components based on
    # consensus of control points across Rhos and Kappas
    KRcutguesses = [getelbow_mod(Rhos), getelbow_cons(Rhos),
                    getelbow_aggr(Rhos), getelbow_mod(Kappas),
                    getelbow_cons(Kappas), getelbow_aggr(Kappas)]
    Khighelbowval = stats.scoreatpercentile([getelbow_mod(Kappas, val=True),
                                             getelbow_cons(Kappas, val=True),
                                             getelbow_aggr(Kappas, val=True)] +
                                            list(getfbounds(n_echos)),
                                            75, interpolation_method='lower')
    KRcut = np.median(KRcutguesses)
    # only use exclusive when inclusive is extremely inclusive - double KRcut
    if getelbow_cons(Kappas) > KRcut * 2 and getelbow_mod(Kappas, val=True) < F01:
        Kcut = getelbow_mod(Kappas, val=True)
    else:
        Kcut = getelbow_cons(Kappas, val=True)
    # only use inclusive when exclusive is extremely exclusive - half KRcut
    # (remember for Rho inclusive is higher, so want both Kappa and Rho
    # to defaut to lower)
    if getelbow_cons(Rhos) > KRcut * 2:
        Rcut = getelbow_mod(Rhos, val=True)
    # for above, consider something like:
    # min([getelbow_mod(Rhos,True),sorted(Rhos)[::-1][KRguess] ])
    else:
        Rcut = getelbow_cons(Rhos, val=True)
    if Rcut > Kcut:
        Kcut = Rcut  # Rcut should never be higher than Kcut
    KRelbow = andb([Kappas > Kcut, Rhos < Rcut])
    # Make guess of Kundu et al 2011 plus remove high frequencies,
    # generally high variance, and high variance given low Kappa
    tt_lim = stats.scoreatpercentile(tt_table[tt_table[:, 0] > 0, 0],
                                     75, interpolation_method='lower')/3
    KRguess = np.setdiff1d(np.setdiff1d(nc[KRelbow == 2], rej),
                           np.union1d(nc[tt_table[:, 0] < tt_lim],
                           np.union1d(np.union1d(nc[spz > 1],
                                                 nc[Vz > 2]),
                                      nc[andb([varex > 0.5 * sorted(varex)[::-1][int(KRcut)],
                                               Kappas < 2*Kcut]) == 2])))
    guessmask = np.zeros(len(nc))
    guessmask[KRguess] = 1

    # Throw lower-risk bad components out
    rejB = ncl[andb([tt_table[ncl, 0] < 0,
                     varex[ncl] > np.median(varex), ncl > KRcut]) == 3]
    rej = np.union1d(rej, rejB)
    ncl = np.setdiff1d(ncl, rej)

    for ii in range(20000):
        db = DBSCAN(eps=.005 + ii * .005, min_samples=3).fit(fz.T)

        # it would be great to have descriptive names, here
        cond1 = db.labels_.max() > 1
        cond2 = db.labels_.max() < len(nc) / 6
        cond3 = np.intersect1d(rej, nc[db.labels_ == 0]).shape[0] == 0
        cond4 = np.array(db.labels_ == -1, dtype=int).sum() / float(len(nc)) < .5

        if cond1 and cond2 and cond3 and cond4:
            epsmap.append([ii, dice(guessmask, db.labels_ == 0),
                           np.intersect1d(nc[db.labels_ == 0],
                           nc[Rhos > getelbow_mod(Rhos_sorted,
                                                  val=True)]).shape[0]])
            if debug:
                lgr.info('found solution', ii, db.labels_)
        db = None

    epsmap = np.array(epsmap)
    group0 = []
    dbscanfailed = False
    if len(epsmap) != 0:
        # Select index that maximizes Dice with guessmask but first
        # minimizes number of higher Rho components
        ii = int(epsmap[np.argmax(epsmap[epsmap[:, 2] == np.min(epsmap[:, 2]), 1], 0), 0])
        lgr.info('Component selection tuning: ', epsmap[:, 1].max())
        db = DBSCAN(eps=.005+ii*.005, min_samples=3).fit(fz.T)
        ncl = nc[db.labels_ == 0]
        ncl = np.setdiff1d(ncl, rej)
        ncl = np.setdiff1d(ncl, ncl[ncl > len(nc) - len(rej)])
        group0 = ncl.copy()
        group_n1 = nc[db.labels_ == -1]
        to_clf = np.setdiff1d(nc, np.union1d(ncl, rej))
    if len(group0) == 0 or len(group0) < len(KRguess) * .5:
        dbscanfailed = True
        lgr.info('DBSCAN based guess failed. Using elbow guess method.')
        ncl = np.setdiff1d(np.setdiff1d(nc[KRelbow == 2], rej),
                           np.union1d(nc[tt_table[:, 0] < tt_lim],
                           np.union1d(np.union1d(nc[spz > 1],
                                      nc[Vz > 2]),
                                      nc[andb([varex > 0.5 * sorted(varex)[::-1][int(KRcut)],
                                               Kappas < 2 * Kcut]) == 2])))
        group0 = ncl.copy()
        group_n1 = []
        to_clf = np.setdiff1d(nc, np.union1d(group0, rej))
    if len(group0) < 2 or (len(group0) < 4 and float(len(rej))/len(group0) > 3):
        lgr.info('WARNING: Extremely limited reliable BOLD signal space. '
                 'Not filtering further into midk etc.')
        midkfailed = True
        min_acc = np.array([])
        if len(group0) != 0:
            # For extremes, building in a 20% tolerance
            toacc_hi = np.setdiff1d(nc[andb([fdist <= np.max(fdist[group0]),
                                             Rhos < F025, Vz > -2]) == 3],
                                    np.union1d(group0, rej))
            min_acc = np.union1d(group0, toacc_hi)
            to_clf = np.setdiff1d(nc, np.union1d(min_acc, rej))
        diagstepkeys = ['rej', 'KRcut', 'Kcut', 'Rcut', 'dbscanfailed',
                        'midkfailed', 'KRguess', 'group0', 'min_acc',
                        'toacc_hi']
        diagstepout = []
        for ddk in diagstepkeys:
            diagstepout.append("%s: %s" % (ddk, eval('str(%s)' % ddk)))
        with open('csstepdata.txt', 'w') as ofh:
            ofh.write('\n'.join(diagstepout))
        ofh.close()
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
    # Tried getting rid of accepting based on SVM altogether,
    # now using only rejecting
    toacc_hi = np.setdiff1d(nc[andb([fdist <= np.max(fdist[group0]),
                               Rhos < F025, Vz > -2]) == 3],
                            np.union1d(group0, rej))
    toacc_lo = np.intersect1d(to_clf,
                              nc[andb([spz < 1, Rz < 0, mmix_kurt_z_max < 5,
                                       Dz > -1, Tz > -1, Vz < 0, Kappas >= F025,
                                       fdist < 3 * np.percentile(fdist[group0], 98)]) == 8])
    midk_clf, clf_ = do_svm(fproj_arr_val[:, np.union1d(group0, rej)].T,
                            [0] * len(group0) + [1] * len(rej),
                            fproj_arr_val[:, to_clf].T,
                            svmtype=2)
    midk = np.setdiff1d(to_clf[andb([midk_clf == 1,
                                     varex[to_clf] > np.median(varex[group0])]) == 2],
                        np.union1d(toacc_hi, toacc_lo))
    # only use SVM to augment toacc_hi only if toacc_hi isn't already
    # conflicting with SVM choice
    if len(np.intersect1d(to_clf[andb([midk_clf == 1,
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
    filewrite(veinout, 'veinmaskf', ref_img)
    veinBout = unmask(veinmaskB, mask)
    filewrite(veinBout, 'veins50', ref_img)
    """

    tsoc_B_Zcl = np.zeros(tsoc_B.shape)
    tsoc_B_Zcl[Z_clmaps != 0] = np.abs(tsoc_B)[Z_clmaps != 0]
    sig_B = [stats.scoreatpercentile(tsoc_B_Zcl[tsoc_B_Zcl[:, ii] != 0, ii], 25)
             if len(tsoc_B_Zcl[tsoc_B_Zcl[:, ii] != 0, ii]) != 0
             else 0 for ii in nc]
    sig_B = np.abs(tsoc_B) > np.tile(sig_B, [tsoc_B.shape[0], 1])

    veinmask = andb([t2s < stats.scoreatpercentile(t2s[t2s != 0], 15,
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
        veincand = fmask(unmask(andb([s0[t2s != 0] < np.median(s0[t2s != 0]),
                                t2s[t2s != 0] < t2sl]) >= 1, t2s != 0), mask)
        veinW[~veincand] = 0
        invein = veinW.sum(1)[fmask(unmask(veinmaskf, mask) * unmask(veinW.sum(1) > 1, mask),
                                    mask)]
        minW = 10 * (np.log10(invein).mean()) - 1 * 10**(np.log10(invein).std())
        veinmaskB = veinW.sum(1) > minW
        tsoc_Bp = tsoc_B.copy()
        tsoc_Bp[tsoc_Bp < 0] = 0
        vvex = np.array([(tsoc_Bp[veinmaskB, ii]**2.).sum() /
                         (tsoc_Bp[:, ii]**2.).sum() for ii in nc])
        group0_res = np.intersect1d(KRguess, group0)
        phys_var_zs.append((vvex - vvex[group0_res].mean()) / vvex[group0_res].std())
        veinBout = unmask(veinmaskB, mask)
        filewrite(veinBout, 'veins_l%i' % t2sl_i, ref_img)

    # Mask to sample veins
    phys_var_z = np.array(phys_var_zs).max(0)
    Vz2 = (varex_ - varex_[group0].mean())/varex_[group0].std()

    """
    Step 4: Learn joint TE-dependence spatial and temporal models to move
    remaining artifacts to ignore class
    """

    to_ign = []

    minK_ign = np.max([F05, getelbow_cons(Kappas, val=True)])
    newcest = len(group0)+len(toacc_hi[Kappas[toacc_hi] > minK_ign])
    phys_art = np.setdiff1d(nc[andb([phys_var_z > 3.5,
                                     Kappas < minK_ign]) == 2], group0)
    phys_art = np.union1d(np.setdiff1d(nc[andb([phys_var_z > 2,
                                                (stats.rankdata(phys_var_z) -
                                                 stats.rankdata(Kappas)) > newcest / 2,
                                                Vz2 > -1]) == 3],
                                       group0), phys_art)
    # Want to replace field_art with an acf/SVM based approach
    # instead of a kurtosis/filter one
    field_art = np.setdiff1d(nc[andb([mmix_kurt_z_max > 5,
                                      Kappas < minK_ign]) == 2], group0)
    field_art = np.union1d(np.setdiff1d(nc[andb([mmix_kurt_z_max > 2,
                                           (stats.rankdata(mmix_kurt_z_max) -
                                            stats.rankdata(Kappas)) > newcest / 2,
                                           Vz2 > 1, Kappas < F01]) == 4],
                                        group0), field_art)
    field_art = np.union1d(np.setdiff1d(nc[andb([mmix_kurt_z_max > 3, Vz2 > 3,
                                                 Rhos > np.percentile(Rhos[group0], 75)]) == 3],
                                        group0), field_art)
    field_art = np.union1d(np.setdiff1d(nc[andb([mmix_kurt_z_max > 5, Vz2 > 5]) == 2],
                                        group0), field_art)
    misc_art = np.setdiff1d(nc[andb([(stats.rankdata(Vz) - stats.rankdata(Ktz)) > newcest / 2,
                            Kappas < Khighelbowval]) == 2], group0)
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
        ncl = np.union1d(ncl, np.intersect1d(orphan, nc[andb([Kappas > F05,
                                                              Rhos < F025,
                                                              Kappas > Rhos,
                                                              Vz3 <= -1,
                                                              Vz3 > -3,
                                                              mmix_kurt_z_max < 2.5]) == 6]))
        ign = np.setdiff1d(nc, list(ncl)+list(midk)+list(rej))
        orphan = np.setdiff1d(nc, list(ncl) + list(to_ign) + list(midk) + list(rej))

    if savecsdiag:
        diagstepkeys = ['rej', 'KRcut', 'Kcut', 'Rcut', 'dbscanfailed',
                        'KRguess', 'group0', 'dice_rej', 'rej_supp', 'to_clf',
                        'midk', 'svm_acc_fail', 'toacc_hi', 'toacc_lo',
                        'field_art', 'phys_art', 'misc_art', 'ncl', 'ign']
        diagstepout = []
        for ddk in diagstepkeys:
            diagstepout.append("%s: %s" % (ddk, eval('str(%s)' % ddk)))
        with open('csstepdata.txt', 'w') as ofh:
            ofh.write('\n'.join(diagstepout))
        allfz = np.array([Tz, Vz, Ktz, KRr, cnz, Rz, mmix_kurt, fdist_z])
        np.savetxt('csdata.txt', allfz)

    return list(sorted(ncl)), list(sorted(rej)), list(sorted(midk)), list(sorted(ign))


def tedpca(catd, combmode, mask, stabilize, ref_img, tes, kdaw, rdaw, ste=0,
           mlepca=True):
    n_samp, n_echos, n_vols = catd.shape
    ste = np.array([int(ee) for ee in str(ste).split(',')])

    if len(ste) == 1 and ste[0] == -1:
        lgr.info('-Computing PCA of optimally combined multi-echo data')
        d = OCcatd[make_min_mask(OCcatd[:, np.newaxis, :])][:, np.newaxis, :]
    elif len(ste) == 1 and ste[0] == 0:
        lgr.info('-Computing PCA of spatially concatenated multi-echo data')
        d = catd[mask].astype('float64')
    else:
        lgr.info('-Computing PCA of TE #%s' % ','.join([str(ee) for ee in ste]))
        d = np.stack([catd[mask, ee] for ee in ste - 1], axis=1).astype('float64')

    eim = np.squeeze(eimask(d))
    d = np.squeeze(d[eim])

    dz = ((d.T - d.T.mean(axis=0)) / d.T.std(axis=0)).T  # var normalize ts
    dz = (dz - dz.mean()) / dz.std()  # var normalize everything

    if not op.exists('pcastate.pkl'):
        # Do PC dimension selection and get eigenvalue cutoff
        if mlepca:
            from sklearn.decomposition import PCA
            ppca = PCA(n_components='mle', svd_solver='full')
            ppca.fit(dz)
            v = ppca.components_
            s = ppca.explained_variance_
            u = np.dot(np.dot(dz, v.T), np.diag(1. / s))
        else:
            u, s, v = np.linalg.svd(dz, full_matrices=0)

        # actual variance explained (normalized)
        sp = s / s.sum()
        eigelb = getelbow_mod(sp, val=True)

        spdif = np.abs(np.diff(sp))
        spdifh = spdif[(len(spdif)//2):]
        spdthr = np.mean([spdifh.max(), spdif.min()])
        spmin = sp[(len(spdif)//2) + np.arange(len(spdifh))[spdifh >= spdthr][0] + 1]
        spcum = np.cumsum(sp)

        # Compute K and Rho for PCA comps
        eimum = np.atleast_2d(eim)
        eimum = np.transpose(eimum, np.argsort(eimum.shape)[::-1])
        eimum = eimum.prod(axis=1)
        o = np.zeros((mask.shape[0], *eimum.shape[1:]))
        o[mask] = eimum
        eimum = np.squeeze(o).astype(bool)

        vTmix = v.T
        vTmixN = ((vTmix.T - vTmix.T.mean(0)) / vTmix.T.std(0)).T
        _, ctb, betasv, v_T = fitmodels_direct(catd, v.T, eimum, t2s, t2sG,
                                               tes, combmode, ref_img,
                                               mmixN=vTmixN, full_sel=False)
        ctb = ctb[ctb[:, 0].argsort(), :]
        ctb = np.vstack([ctb.T[:3], sp]).T

        # Save state
        lgr.info('Saving PCA')
        pcastate = {'u': u, 's': s, 'v': v, 'ctb': ctb,
                    'eigelb': eigelb, 'spmin': spmin, 'spcum': spcum}
        try:
            with open('pcastate.pkl', 'wb') as handle:
                pickle.dump(pcastate, handle)
        except TypeError:
            lgr.info('Could not save PCA solution!')

    else:  # if loading existing state
        lgr.info('Loading PCA')
        with open('pcastate.pkl', 'rb') as handle:
            pcastate = pickle.load(handle)
        u, s, v = pcastate['u'], pcastate['s'], pcastate['v']
        ctb, eigelb = pcastate['ctb'], pcastate['eigelb']
        spmin, spcum = pcastate['spmin'], pcastate['spcum']

    np.savetxt('comp_table_pca.txt', ctb[ctb[:, 1].argsort(), :][::-1])
    np.savetxt('mepca_mix.1D', v[ctb[:, 1].argsort()[::-1], :].T)

    kappas = ctb[ctb[:, 1].argsort(), 1]
    rhos = ctb[ctb[:, 2].argsort(), 2]
    fmin, fmid, fmax = getfbounds(n_echos)
    kappa_thr = np.average(sorted([fmin, getelbow_mod(kappas, val=True)/2, fmid]),
                           weights=[kdaw, 1, 1])
    rho_thr = np.average(sorted([fmin, getelbow_cons(rhos, val=True)/2, fmid]),
                         weights=[rdaw, 1, 1])
    if int(kdaw) == -1:
        kappas_lim = kappas[andb([kappas < fmid, kappas > fmin]) == 2]
        kappa_thr = kappas_lim[getelbow_mod(kappas_lim)]
        rhos_lim = rhos[andb([rhos < fmid, rhos > fmin]) == 2]
        rho_thr = rhos_lim[getelbow_mod(rhos_lim)]
        stabilize = True
    if int(kdaw) != -1 and int(rdaw) == -1:
        rhos_lim = rhos[andb([rhos < fmid, rhos > fmin]) == 2]
        rho_thr = rhos_lim[getelbow_mod(rhos_lim)]

    is_hik = np.array(ctb[:, 1] > kappa_thr, dtype=np.int)
    is_hir = np.array(ctb[:, 2] > rho_thr, dtype=np.int)
    is_hie = np.array(ctb[:, 3] > eigelb, dtype=np.int)
    is_his = np.array(ctb[:, 3] > spmin, dtype=np.int)
    is_not_fmax1 = np.array(ctb[:, 1] != F_MAX, dtype=np.int)
    is_not_fmax2 = np.array(ctb[:, 2] != F_MAX, dtype=np.int)
    pcscore = (is_hik + is_hir + is_hie) * is_his * is_not_fmax1 * is_not_fmax2
    if stabilize:
        temp7 = np.array(spcum < 0.95, dtype=np.int)
        temp8 = np.array(ctb[:, 2] > fmin, dtype=np.int)
        temp9 = np.array(ctb[:, 1] > fmin, dtype=np.int)
        pcscore = pcscore * temp7 * temp8 * temp9

    pcsel = pcscore > 0
    dd = u.dot(np.diag(s*np.array(pcsel, dtype=np.int))).dot(v)

    n_components = s[pcsel].shape[0]
    lgr.info('--Selected {0} components. Minimum Kappa={1:.02f} '
             'Rho={2:.02f}'.format(n_components, kappa_thr, rho_thr))

    dd = stats.zscore(dd.T, axis=0).T  # variance normalize timeseries
    dd = stats.zscore(dd, axis=None)  # variance normalize everything

    return n_components, dd


def tedica(n_components, dd, conv, fixed_seed, cost, final_cost):
    """
    Input is dimensionally reduced spatially concatenated multi-echo
    time series dataset from `tedpca`. Output is comptable, mmix, smaps
    from ICA, and betas from fitting catd to mmix.
    """

    import mdp
    climit = float(conv)
    mdp.numx_rand.seed(fixed_seed)
    icanode = mdp.nodes.FastICANode(white_comp=n_components, approach='symm', g=cost,
                                    fine_g=final_cost, coarse_limit=climit*100,
                                    limit=climit, verbose=True)
    icanode.train(dd)
    smaps = icanode.execute(dd)  # noqa
    mmix = icanode.get_recmatrix().T
    mmix = (mmix-mmix.mean(0))/mmix.std(0)
    return mmix


def gscontrol_raw(catd, optcom, n_echos, ref_img, dtrank=4):
    """
    This function uses the spatial global signal estimation approach to
    modify catd (global variable) to removal global signal out of individual
    echo time series datasets. The spatial global signal is estimated
    from the optimally combined data after detrending with a Legendre
    polynomial basis of `order = 0` and `degree = dtrank`.
    """

    lgr.info('++ Applying amplitude-based T1 equilibration correction')

    # Legendre polynomial basis for denoising
    bounds = np.linspace(-1, 1, optcom.shape[-1])
    Lmix = np.column_stack([lpmv(0, vv, bounds) for vv in range(dtrank)])

    # compute mean, std, mask local to this function
    # inefficient, but makes this function a bit more modular
    Gmu = optcom.mean(axis=-1)  # temporal mean
    Gmask = Gmu != 0

    # find spatial global signal
    dat = optcom[Gmask] - Gmu[Gmask][:, np.newaxis]
    sol = np.linalg.lstsq(Lmix, dat.T)[0]  # Legendre basis for detrending
    detr = dat - np.dot(sol.T, Lmix.T)[0]
    sphis = (detr).min(axis=1)
    sphis -= sphis.mean()
    filewrite(unmask(sphis, Gmask), 'T1gs', ref_img)

    # find time course ofc the spatial global signal
    # make basis with the Legendre basis
    glsig = np.linalg.lstsq(np.atleast_2d(sphis).T, dat)[0]
    glsig = stats.zscore(glsig, axis=None)
    np.savetxt('glsig.1D', glsig)
    glbase = np.hstack([Lmix, glsig.T])

    # Project global signal out of optimally combined data
    sol = np.linalg.lstsq(np.atleast_2d(glbase), dat.T)[0]
    tsoc_nogs = dat - np.dot(np.atleast_2d(sol[dtrank]).T,
                             np.atleast_2d(glbase.T[dtrank])) + Gmu[Gmask][:, np.newaxis]

    filewrite(optcom, 'tsoc_orig', ref_img)
    optcom = unmask(tsoc_nogs, Gmask)
    filewrite(optcom, 'tsoc_nogs.nii', ref_img)

    # Project glbase out of each echo
    dm_catd = catd.copy()  # don't overwrite catd
    for echo in range(n_echos):
        dat = dm_catd[:, echo, :][Gmask]
        sol = np.linalg.lstsq(np.atleast_2d(glbase), dat.T)[0]
        e_nogs = dat - np.dot(np.atleast_2d(sol[dtrank]).T,
                              np.atleast_2d(glbase.T[dtrank]))
        dm_catd[:, echo, :] = unmask(e_nogs, Gmask)

    return dm_catd, optcom


def gscontrol_mmix(mmix, acc, rej, midk, empty, ref_img):

    Gmu = OCcatd.mean(axis=-1)
    Gstd = OCcatd.std(axis=-1)
    Gmask = (Gmu != 0)

    """
    Compute temporal regression
    """
    dat = (OCcatd[Gmask] - Gmu[Gmask][:, np.newaxis]) / Gstd[mask][:, np.newaxis]
    solG = np.linalg.lstsq(mmix, dat.T)[0]
    resid = dat - np.dot(solG.T, mmix.T)

    """
    Build BOLD time series without amplitudes, and save T1-like effect
    """
    bold_ts = np.dot(solG.T[:, acc], mmix[:, acc].T)
    sphis = bold_ts.min(axis=-1)
    sphis -= sphis.mean()
    lgr.info(sphis.shape)
    filewrite(unmask(sphis, mask), 'sphis_hik', ref_img)

    """
    Find the global signal based on the T1-like effect
    """
    sol = np.linalg.lstsq(np.atleast_2d(sphis).T, dat)
    glsig = sol[0]

    """
    T1 correct time series by regression
    """
    bold_noT1gs = bold_ts - np.dot(np.linalg.lstsq(glsig.T, bold_ts.T)[0].T, glsig)
    filewrite(unmask(bold_noT1gs * Gstd[mask][:, np.newaxis], mask),
              'hik_ts_OC_T1c.nii', ref_img)

    """
    Make medn version of T1 corrected time series
    """
    filewrite(Gmu[..., np.newaxis] + unmask((bold_noT1gs+resid)*Gstd[mask][:, np.newaxis], mask),
              'dn_ts_OC_T1c', ref_img)

    """
    Orthogonalize mixing matrix w.r.t. T1-GS
    """
    mmixnogs = mmix.T - np.dot(np.linalg.lstsq(glsig.T, mmix)[0].T, glsig)
    mmixnogs_mu = mmixnogs.mean(-1)
    mmixnogs_std = mmixnogs.std(-1)
    mmixnogs_norm = (mmixnogs - mmixnogs_mu[:, np.newaxis]) / mmixnogs_std[:, np.newaxis]
    mmixnogs_norm = np.vstack([np.atleast_2d(np.ones(max(glsig.shape))), glsig, mmixnogs_norm])

    """
    Write T1-GS corrected components and mixing matrix
    """
    sol = np.linalg.lstsq(mmixnogs_norm.T, dat.T)
    filewrite(unmask(sol[0].T[:, 2:], mask), 'betas_hik_OC_T1c', ref_img)
    np.savetxt('meica_mix_T1c.1D', mmixnogs)


def write_split_ts(data, comptable, mmix, acc, rej, midk, ref_img, suffix=''):
    mdata = fmask(data, mask)
    betas = fmask(get_coeffs(unmask((mdata.T - mdata.T.mean(0)).T, mask),
                             mask, mmix), mask)
    dmdata = mdata.T-mdata.T.mean(0)
    varexpl = (1-((dmdata.T-betas.dot(mmix.T))**2.).sum()/(dmdata**2.).sum())*100
    lgr.info('Variance explained: ', varexpl, '%')
    midkts = betas[:, midk].dot(mmix.T[midk, :])
    lowkts = betas[:, rej].dot(mmix.T[rej, :])
    if len(acc) != 0:
        filewrite(unmask(betas[:, acc].dot(mmix.T[acc, :]), mask),
                  'hik_ts_{0}'.format(suffix), ref_img)
    if len(midk) != 0:
        filewrite(unmask(midkts, mask), 'midk_ts_{0}'.format(suffix), ref_img)
    if len(rej) != 0:
        filewrite(unmask(lowkts, mask), 'lowk_ts_{0}'.format(suffix), ref_img)
    filewrite(unmask(data[mask] - lowkts - midkts, mask),
              'dn_ts_{0}'.format(suffix), ref_img)
    return varexpl


def writefeats(data, mmix, mask, ref_img, suffix=''):
    # Write feature versions of components
    feats = computefeats2(data, mmix, mask)
    filewrite(unmask(feats, mask), 'feats_{0}'.format(suffix), ref_img)


def writect(comptable, nt, acc, rej, midk, empty, ctname='', varexpl='-1'):
    n_components = comptable.shape[0]
    sortab = comptable[comptable[:, 1].argsort()[::-1], :]
    if ctname is '':
        ctname = 'comp_table.txt'
    open('accepted.txt', 'w').write(','.join([str(int(cc)) for cc in acc]))
    open('rejected.txt', 'w').write(','.join([str(int(cc)) for cc in rej]))
    open('midk_rejected.txt',
         'w').write(','.join([str(int(cc)) for cc in midk]))

    _computed_vars = dict(file=op.abspath(op.curdir),
                          vex=varexpl,
                          n_components=n_components,
                          dfe=len(acc),
                          rjn=len(midk) + len(rej),
                          dfn=nt - len(midk) - len(rej),
                          acc=','.join([str(int(cc)) for cc in acc]),
                          rej=','.join([str(int(cc)) for cc in rej]),
                          mid=','.join([str(int(cc)) for cc in midk]),
                          ign=','.join([str(int(cc)) for cc in empty]))
    heading = textwrap.dedent("""\
        # ME-ICA Component statistics table for: {file} #
        # Dataset variance explained by ICA (VEx): {vex:.2f}
        # Total components generated by decomposition (TCo): {n_components}
        # No. accepted BOLD-like components, i.e. effective degrees
          of freedom for correlation (lower bound; DFe): {dfe}
        # Total number of rejected components (RJn): {rjn}
        # Nominal degress of freedom in denoised time series
          (..._medn.nii.gz; DFn): {dfn}
        # ACC {acc} \t# Accepted BOLD-like components
        # REJ {rej} \t# Rejected non-BOLD components
        # MID {mid} \t# Rejected R2*-weighted artifacts
        # IGN {ign} \t# Ignored components (kept in denoised time series)
        # VEx   TCo DFe RJn DFn
        # {vex:.2f} {n_components} {dfe} {rjn} {dfn}
        # comp    Kappa   Rho Var   Var(norm)
        """).format(**_computed_vars)

    with open(ctname, 'w') as f:
        f.write(heading)
        for i in range(n_components):
            f.write('%d\t%f\t%f\t%.2f\t%.2f\n' % (sortab[i, 0], sortab[i, 1],
                                                  sortab[i, 2], sortab[i, 3],
                                                  sortab[i, 4]))


def writeresults(OCcatd, comptable, mmix, nt, acc, rej, midk, empty, ref_img):
    lgr.info('++ Writing optimally combined time series')
    ts = OCcatd
    filewrite(ts, 'ts_OC', ref_img)
    print("++ Writing Kappa-filtered optimally combined timeseries")
    varexpl = write_split_ts(ts, comptable, mmix, acc, rej, midk, ref_img,
                             suffix='OC')
    print("++ Writing signal versions of components")
    ts_B = get_coeffs(ts, mask, mmix)
    filewrite(ts_B[:, :, :, :], 'betas_OC', ref_img)

    if len(acc) != 0:
        filewrite(ts_B[:, :, :, acc], 'betas_hik_OC', ref_img)
        print("++ Writing optimally combined high-Kappa features")
        writefeats(split_ts(ts, comptable, mmix, acc, rej, midk)[0],
                   mmix[:, acc], mask, ref_img, suffix='OC2')
    print("++ Writing component table")
    writect(comptable, nt, acc, rej, midk, empty, ctname='comp_table.txt',
            varexpl=varexpl)


def writeresults_echoes(acc, rej, midk, ref_img, comptable, mmix, n_echos):
    for i_echo in range(n_echos):
        print("++ Writing Kappa-filtered TE#%i timeseries" % (i_echo+1))
        write_split_ts(catd[:, :, :, i_echo, :], comptable, mmix,
                       acc, rej, midk, ref_img, suffix='e%i' % (i_echo+1))


def main(data, tes, mixm=None, ctab=None, manacc=None, strict=False,
         no_gscontrol=False, kdaw=10., rdaw=1., conv=2.5e-5, ste=-1,
         combmode='t2s', dne=False, initcost='tanh', finalcost='tanh',
         stabilize=False, fout=False, filecsdata=False, label=None,
         fixed_seed=42):
    """
    Parameters
    ----------
    data : :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.
    mixm : :obj:`str`, optional
        File containing mixing matrix. If not provided, ME-PCA and ME-ICA are
        done.
    ctab : :obj:`str`, optional
        File containing component table from which to extract pre-computed
        classifications.
    manacc : :obj:`str`, optional
        Comma separated list of manually accepted components in string form.
        Default is None.
    strict : :obj:`bool`, optional
        Ignore low-variance ambiguous components. Default is False.
    no_gzcontrol : :obj:`bool`, optional
        Control global signal using spatial approach. Default is False.
    kdaw : :obj:`float`, optional
        Dimensionality augmentation weight (Kappa). Default is 10.
        -1 for low-dimensional ICA.
    rdaw : :obj:`float`, optional
        Dimensionality augmentation weight (Rho). Default is 1.
        -1 for low-dimensional ICA.
    conv : :obj:`float`, optional
        Convergence limit. Default is 2.5e-5.
    ste : :obj:`int`, optional
        Source TEs for models. 0 for all, -1 for optimal combination.
        Default is -1.
    combmode : {'t2s', 'ste'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'ste' (Poser).
    dne : :obj:`bool`, optional
        Denoise each TE dataset separately. Default is False.
    initcost : {'tanh', 'pow3', 'gaus', 'skew'}, optional
        Initial cost function for ICA. Default is 'tanh'.
    finalcost : {'tanh', 'pow3', 'gaus', 'skew'}, optional
        Final cost function. Default is 'tanh'.
    stabilize : :obj:`bool`, optional
        Stabilize convergence by reducing dimensionality, for low quality data.
        Default is False.
    fout : :obj:`bool`, optional
        Save output TE-dependence Kappa/Rho SPMs. Default is False.
    filecsdata : :obj:`bool`, optional
        Save component selection data to file. Default is False.
    label : :obj:`str` or :obj:`None`, optional
        Label for output directory. Default is None.
    fixed_seed : :obj:`int`, optional
        Seeded value for ICA, for reproducibility.
    """

    global catd, ref_img

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    catd, ref_img = load_data(data, n_echos=n_echos)
    n_samp, n_echos, n_vols = catd.shape

    if fout:
        fout = ref_img
    else:
        fout = None

    kdaw, rdaw = float(kdaw), float(rdaw)

    if label is not None:
        out_dir = 'TED.{0}'.format(label)
    else:
        out_dir = 'TED'
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    if mixm is not None and op.isfile(mixm):
        shutil.copyfile(mixm, op.join(out_dir, 'meica_mix.1D'))
        shutil.copyfile(mixm, op.join(out_dir, op.basename(mixm)))
    elif mixm is not None:
        raise IOError('Argument "mixm" must be an existing file.')

    if ctab is not None and op.isfile(ctab):
        shutil.copyfile(ctab, op.join(out_dir, 'comp_table.txt'))
        shutil.copyfile(ctab, op.join(out_dir, op.basename(ctab)))
    elif ctab is not None:
        raise IOError('Argument "ctab" must be an existing file.')

    os.chdir(out_dir)

    lgr.info('++ Computing Mask')
    global mask
    mask, masksum = makeadmask(catd, minimum=False, getsum=True)

    lgr.info('++ Computing T2* map')
    global t2s, s0, t2sG
    t2s, s0, t2ss, s0s, t2sG, s0G = t2sadmap(catd, tes,
                                             mask, masksum,
                                             start_echo=1)

    # set a hard cap for the T2* map
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s.flatten(), 99.5,
                                      interpolation_method='lower')
    t2s[t2s > cap_t2s * 10] = cap_t2s
    filewrite(s0, op.join(out_dir, 's0v'), ref_img)
    filewrite(t2s, op.join(out_dir, 't2sv'), ref_img)
    filewrite(t2ss, op.join(out_dir, 't2ss'), ref_img)
    filewrite(s0s, op.join(out_dir, 's0vs'), ref_img)
    filewrite(s0G, op.join(out_dir, 's0vG'), ref_img)
    filewrite(t2sG, op.join(out_dir, 't2svG'), ref_img)

    # optimally combine data
    global OCcatd
    OCcatd = make_optcom(catd, t2sG, tes, mask, combmode)

    # regress out global signal unless explicitly not desired
    if not no_gscontrol:
        catd, OCcatd = gscontrol_raw(catd, OCcatd, n_echos, ref_img)

    if mixm is None:
        lgr.info("++ Doing ME-PCA and ME-ICA")
        n_components, dd = tedpca(catd, combmode, mask, stabilize, ref_img,
                                  tes=tes, kdaw=kdaw, rdaw=rdaw, ste=ste)
        mmix_orig = tedica(n_components, dd, conv, fixed_seed, cost=initcost,
                           final_cost=finalcost)
        np.savetxt(op.join(out_dir, '__meica_mix.1D'), mmix_orig)
        seldict, comptable, betas, mmix = fitmodels_direct(catd, mmix_orig,
                                                           mask, t2s, t2sG,
                                                           tes, combmode,
                                                           ref_img,
                                                           fout=fout,
                                                           reindex=True)
        np.savetxt(op.join(out_dir, 'meica_mix.1D'), mmix)

        acc, rej, midk, empty = selcomps(seldict, mmix, ref_img, manacc, n_echos,
                                         strict_mode=strict,
                                         filecsdata=filecsdata)
    else:
        mmix_orig = np.loadtxt(op.join(out_dir, 'meica_mix.1D'))
        seldict, comptable, betas, mmix = fitmodels_direct(catd, mmix_orig,
                                                           mask, t2s, t2sG,
                                                           tes, combmode,
                                                           ref_img,
                                                           fout=fout)
        if ctab is None:
            acc, rej, midk, empty = selcomps(seldict, mmix, ref_img, manacc,
                                             n_echos,
                                             filecsdata=filecsdata,
                                             strict_mode=strict)
        else:
            acc, rej, midk, empty = ctabsel(ctab)

    if len(acc) == 0:
        lgr.info('** WARNING! No BOLD components detected!!! \n'
                 '** Please check data and results!')

    writeresults(OCcatd, comptable, mmix, n_vols, acc, rej, midk, empty, ref_img)
    gscontrol_mmix(mmix, acc, rej, midk, empty, ref_img)
    if dne:
        writeresults_echoes(acc, rej, midk, ref_img, comptable, mmix, n_echos)
