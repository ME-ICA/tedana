"""
Signal decomposition methods for tedana
"""
import pickle
import logging
import os.path as op

import numpy as np
from scipy import stats

from tedana import model, utils
from tedana.decomposition._utils import eimask
from tedana.selection._utils import (getelbow_cons, getelbow_mod)

LGR = logging.getLogger(__name__)

F_MAX = 500
Z_MAX = 8


def tedpca(catd, OCcatd, combmode, mask, t2s, t2sG, stabilize,
           ref_img, tes, kdaw, rdaw, ste=0, mlepca=True):
    """
    Use principal components analysis (PCA) to identify and remove thermal
    noise from multi-echo data.

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input functional data
    OCcatd : (S x T) array_like
        Optimally-combined time series data
    combmode : {'t2s', 'ste'} str
        How optimal combination of echos should be made, where 't2s' indicates
        using the method of Posse 1999 and 'ste' indicates using the method of
        Poser 2006
    mask : (S,) array_like
        Boolean mask array
    stabilize : bool
        Whether to attempt to stabilize convergence of ICA by returning
        dimensionally-reduced data from PCA and component selection.
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    tes : list
        List of echo times associated with `catd`, in milliseconds
    kdaw : float
        Dimensionality augmentation weight for Kappa calculations
    rdaw : float
        Dimensionality augmentation weight for Rho calculations
    ste : int or list-of-int, optional
        Which echos to use in PCA. Values -1 and 0 are special, where a value
        of -1 will indicate using all the echos and 0 will indicate using the
        optimal combination of the echos. A list can be provided to indicate
        a subset of echos. Default: 0
    mlepca : bool, optional
        Whether to use the method originally explained in Minka, NIPS 2000 for
        guessing PCA dimensionality instead of a traditional SVD. Default: True

    Returns
    -------
    n_components : int
        Number of components retained from PCA decomposition
    dd : (S x E x T) :obj:`numpy.ndarray`
        Dimensionally-reduced functional data

    Notes
    -----
    ======================    =================================================
    Notation                  Meaning
    ======================    =================================================
    :math:`\\kappa`            Component pseudo-F statistic for TE-dependent
                              (BOLD) model.
    :math:`\\rho`              Component pseudo-F statistic for TE-independent
                              (artifact) model.
    :math:`v`                 Voxel
    :math:`V`                 Total number of voxels in mask
    :math:`\\zeta`             Something
    :math:`c`                 Component
    :math:`p`                 Something else
    ======================    =================================================

    Steps:

    1.  Variance normalize either multi-echo or optimally combined data,
        depending on settings.
    2.  Decompose normalized data using PCA or SVD.
    3.  Compute :math:`{\\kappa}` and :math:`{\\rho}`:

            .. math::
                {\\kappa}_c = \\frac{\sum_{v}^V {\\zeta}_{c,v}^p * \
                      F_{c,v,R_2^*}}{\sum {\\zeta}_{c,v}^p}

                {\\rho}_c = \\frac{\sum_{v}^V {\\zeta}_{c,v}^p * \
                      F_{c,v,S_0}}{\sum {\\zeta}_{c,v}^p}

    4.  Some other stuff. Something about elbows.
    5.  Classify components as thermal noise if they meet both of the
        following criteria:

            - Nonsignificant :math:`{\\kappa}` and :math:`{\\rho}`.
            - Nonsignificant variance explained.
    """

    n_samp, n_echos, n_vols = catd.shape
    ste = np.array([int(ee) for ee in str(ste).split(',')])

    if len(ste) == 1 and ste[0] == -1:
        LGR.info('Computing PCA of optimally combined multi-echo data')
        d = OCcatd[utils.make_min_mask(OCcatd[:, np.newaxis, :])][:, np.newaxis, :]
    elif len(ste) == 1 and ste[0] == 0:
        LGR.info('Computing PCA of spatially concatenated multi-echo data')
        d = catd[mask].astype('float64')
    else:
        LGR.info('Computing PCA of echo #%s' % ','.join([str(ee) for ee in ste]))
        d = np.stack([catd[mask, ee] for ee in ste - 1], axis=1).astype('float64')

    eim = np.squeeze(eimask(d))
    d = np.squeeze(d[eim])

    dz = ((d.T - d.T.mean(axis=0)) / d.T.std(axis=0)).T  # var normalize ts
    dz = (dz - dz.mean()) / dz.std()  # var normalize everything

    if not op.exists('pcastate.pkl'):
        # do PC dimension selection and get eigenvalue cutoff
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
        LGR.info('Making initial component selection guess from PCA results')
        _, ctb, betasv, v_T = model.fitmodels_direct(catd, v.T, eimum, t2s, t2sG,
                                                     tes, combmode, ref_img,
                                                     mmixN=vTmixN, full_sel=False)
        ctb = ctb[ctb[:, 0].argsort(), :]
        ctb = np.vstack([ctb.T[:3], sp]).T

        # Save state
        fname = op.abspath('pcastate.pkl')
        LGR.info('Saving PCA results to: {}'.format(fname))
        pcastate = {'u': u, 's': s, 'v': v, 'ctb': ctb,
                    'eigelb': eigelb, 'spmin': spmin, 'spcum': spcum}
        try:
            with open(fname, 'wb') as handle:
                pickle.dump(pcastate, handle)
        except TypeError:
            LGR.warning('Could not save PCA solution')

    else:  # if loading existing state
        LGR.info('Loading PCA from: {}'.format('pcastate.pkl'))
        with open('pcastate.pkl', 'rb') as handle:
            pcastate = pickle.load(handle)
        u, s, v = pcastate['u'], pcastate['s'], pcastate['v']
        ctb, eigelb = pcastate['ctb'], pcastate['eigelb']
        spmin, spcum = pcastate['spmin'], pcastate['spcum']

    np.savetxt('comp_table_pca.txt', ctb[ctb[:, 1].argsort(), :][::-1])
    np.savetxt('mepca_mix.1D', v[ctb[:, 1].argsort()[::-1], :].T)

    kappas = ctb[ctb[:, 1].argsort(), 1]
    rhos = ctb[ctb[:, 2].argsort(), 2]
    fmin, fmid, fmax = utils.getfbounds(n_echos)
    kappa_thr = np.average(sorted([fmin, getelbow_mod(kappas, val=True)/2, fmid]),
                           weights=[kdaw, 1, 1])
    rho_thr = np.average(sorted([fmin, getelbow_cons(rhos, val=True)/2, fmid]),
                         weights=[rdaw, 1, 1])
    if int(kdaw) == -1:
        kappas_lim = kappas[utils.andb([kappas < fmid, kappas > fmin]) == 2]
        kappa_thr = kappas_lim[getelbow_mod(kappas_lim)]
        rhos_lim = rhos[utils.andb([rhos < fmid, rhos > fmin]) == 2]
        rho_thr = rhos_lim[getelbow_mod(rhos_lim)]
        stabilize = True
    if int(kdaw) != -1 and int(rdaw) == -1:
        rhos_lim = rhos[utils.andb([rhos < fmid, rhos > fmin]) == 2]
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
    LGR.info('Selected {0} components with Kappa threshold: {1:.02f}, '
             'Rho threshold: {2:.02f}'.format(n_components, kappa_thr, rho_thr))

    dd = stats.zscore(dd.T, axis=0).T  # variance normalize timeseries
    dd = stats.zscore(dd, axis=None)  # variance normalize everything

    return n_components, dd


def tedica(n_components, dd, conv, fixed_seed, cost, final_cost, verbose=False):
    """
    Performs ICA on `dd` and returns mixing matrix

    Parameters
    ----------
    n_components : int
        Number of components retained from PCA decomposition
    dd : (S x E x T) :obj:`numpy.ndarray`
        Dimensionally-reduced functional data, where `S` is samples, `E` is
        echos, and `T` is time
    conv : float
        Convergence limit for ICA
    fixed_seed : int
        Seed for ensuring reproducibility of ICA results
    initcost : {'tanh', 'pow3', 'gaus', 'skew'} str, optional
        Initial cost function for ICA
    finalcost : {'tanh', 'pow3', 'gaus', 'skew'} str, optional
        Final cost function for ICA
    verbose : bool, optional
        Whether to print messages regarding convergence process. Default: False

    Returns
    -------
    mmix : (C x T) :obj:`numpy.ndarray`
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `dd`

    Notes
    -----
    Uses `mdp` implementation of FastICA for decomposition
    """

    import mdp
    climit = float(conv)
    mdp.numx_rand.seed(fixed_seed)
    icanode = mdp.nodes.FastICANode(white_comp=n_components, approach='symm', g=cost,
                                    fine_g=final_cost, coarse_limit=climit*100,
                                    limit=climit, verbose=verbose)
    icanode.train(dd)
    smaps = icanode.execute(dd)  # noqa
    mmix = icanode.get_recmatrix().T
    mmix = stats.zscore(mmix, axis=0)
    return mmix
