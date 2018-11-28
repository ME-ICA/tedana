"""
Signal decomposition methods for tedana
"""
import pickle
import logging
import os.path as op

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from tedana import model, utils, io
from tedana.decomposition._utils import eimask, dwtmat, idwtmat
from tedana.selection._utils import (getelbow_cons, getelbow)
from tedana.due import due, BibTeX

LGR = logging.getLogger(__name__)

F_MAX = 500
Z_MAX = 8


@due.dcite(BibTeX(
    """
    @inproceedings{minka2001automatic,
      title={Automatic choice of dimensionality for PCA},
      author={Minka, Thomas P},
      booktitle={Advances in neural information processing systems},
      pages={598--604},
      year={2001}
    }
    """),
    description='Introduces method for choosing PCA dimensionality '
                'automatically')
def run_svd(data):
    """
    Run Singular Value Decomposition (SVD) on input data.

    Parameters
    ----------
    data : (S [*E] x T) array_like
        Optimally combined (S x T) or full multi-echo (S*E x T) data.

    Returns
    -------
    u : (S [*E] x C) array_like
        Component weight map for each component.
    s : (C,) array_like
        Variance explained for each component.
    v : (C x T) array_like
        Component timeseries.
    """
    # do PC dimension selection and get eigenvalue cutoff
    ppca = PCA(n_components='mle', svd_solver='full')
    ppca.fit(data)
    v = ppca.components_
    s = ppca.explained_variance_
    u = np.dot(np.dot(data, v.T), np.diag(1. / s))
    return u, s, v


def tedpca(catd, OCcatd, combmode, mask, t2s, t2sG, stabilize,
           ref_img, tes, kdaw, rdaw, ste=0, wvpca=False):
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
    stabilize : :obj:`bool`
        Whether to attempt to stabilize convergence of ICA by returning
        dimensionally-reduced data from PCA and component selection.
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    tes : :obj:`list`
        List of echo times associated with `catd`, in milliseconds
    kdaw : :obj:`float`
        Dimensionality augmentation weight for Kappa calculations
    rdaw : :obj:`float`
        Dimensionality augmentation weight for Rho calculations
    ste : :obj:`int` or :obj:`list` of :obj:`int`, optional
        Which echos to use in PCA. Values -1 and 0 are special, where a value
        of -1 will indicate using all the echos and 0 will indicate using the
        optimal combination of the echos. A list can be provided to indicate
        a subset of echos. Default: 0
    wvpca : :obj:`bool`, optional
        Whether to apply wavelet denoising to data. Default: False

    Returns
    -------
    n_components : :obj:`int`
        Number of components retained from PCA decomposition
    dd : (S x T) :obj:`numpy.ndarray`
        Dimensionally reduced optimally combined functional data

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

    Outputs:

    This function writes out several files:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    pcastate.pkl              Values from PCA results.
    comp_table_pca.txt        PCA component table.
    mepca_mix.1D              PCA mixing matrix.
    ======================    =================================================
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

    if wvpca:
        dz, cAl = dwtmat(dz)

    if not op.exists('pcastate.pkl'):
        voxel_comp_weights, varex, comp_ts = run_svd(dz)

        # actual variance explained (normalized)
        varex_norm = varex / varex.sum()
        eigenvalue_elbow = getelbow(varex_norm, return_val=True)

        diff_varex_norm = np.abs(np.diff(varex_norm))
        lower_diff_varex_norm = diff_varex_norm[(len(diff_varex_norm)//2):]
        varex_norm_thr = np.mean([lower_diff_varex_norm.max(),
                                  diff_varex_norm.min()])
        varex_norm_min = varex_norm[
            (len(diff_varex_norm)//2) +
            np.arange(len(lower_diff_varex_norm))[lower_diff_varex_norm >= varex_norm_thr][0] + 1]
        varex_norm_cum = np.cumsum(varex_norm)

        # Compute K and Rho for PCA comps
        eimum = np.atleast_2d(eim)
        eimum = np.transpose(eimum, np.argsort(eimum.shape)[::-1])
        eimum = eimum.prod(axis=1)
        o = np.zeros((mask.shape[0], *eimum.shape[1:]))
        o[mask] = eimum
        eimum = np.squeeze(o).astype(bool)

        vTmix = comp_ts.T
        vTmixN = ((vTmix.T - vTmix.T.mean(0)) / vTmix.T.std(0)).T
        LGR.info('Making initial component selection guess from PCA results')
        _, ct_df, betasv, v_T = model.fitmodels_direct(catd, comp_ts.T, eimum,
                                                       t2s, t2sG,
                                                       tes, combmode, ref_img,
                                                       mmixN=vTmixN,
                                                       full_sel=False)
        # varex_norm overrides normalized varex computed by fitmodels_direct
        ct_df['normalized variance explained'] = varex_norm

        # Save state
        fname = op.abspath('pcastate.pkl')
        LGR.info('Saving PCA results to: {}'.format(fname))
        pcastate = {'voxel_comp_weights': voxel_comp_weights,
                    'varex': varex,
                    'comp_ts': comp_ts,
                    'comptable': ct_df,
                    'eigenvalue_elbow': eigenvalue_elbow,
                    'varex_norm_min': varex_norm_min,
                    'varex_norm_cum': varex_norm_cum}
        try:
            with open(fname, 'wb') as handle:
                pickle.dump(pcastate, handle)
        except TypeError:
            LGR.warning('Could not save PCA solution')

    else:  # if loading existing state
        LGR.info('Loading PCA from: pcastate.pkl')
        with open('pcastate.pkl', 'rb') as handle:
            pcastate = pickle.load(handle)
        voxel_comp_weights, varex = pcastate['voxel_comp_weights'], pcastate['varex']
        comp_ts = pcastate['comp_ts']
        ct_df = pcastate['comptable']
        eigenvalue_elbow = pcastate['eigenvalue_elbow']
        varex_norm_min = pcastate['varex_norm_min']
        varex_norm_cum = pcastate['varex_norm_cum']

    np.savetxt('mepca_mix.1D', comp_ts.T)

    # write component maps to 4D image
    comp_maps = np.zeros((OCcatd.shape[0], comp_ts.shape[0]))
    for i_comp in range(comp_ts.shape[0]):
        temp_comp_ts = comp_ts[i_comp, :][:, None]
        comp_map = utils.unmask(model.computefeats2(OCcatd, temp_comp_ts, mask), mask)
        comp_maps[:, i_comp] = np.squeeze(comp_map)
    io.filewrite(comp_maps, 'mepca_OC_components.nii', ref_img)

    fmin, fmid, fmax = utils.getfbounds(n_echos)
    kappa_thr = np.average(sorted([fmin, getelbow(ct_df['kappa'], return_val=True)/2, fmid]),
                           weights=[kdaw, 1, 1])
    rho_thr = np.average(sorted([fmin, getelbow_cons(ct_df['rho'], return_val=True)/2, fmid]),
                         weights=[rdaw, 1, 1])
    if int(kdaw) == -1:
        lim_idx = utils.andb([ct_df['kappa'] < fmid,
                              ct_df['kappa'] > fmin]) == 2
        kappa_lim = ct_df.loc[lim_idx, 'kappa'].values
        kappa_thr = kappa_lim[getelbow(kappa_lim)]

        lim_idx = utils.andb([ct_df['rho'] < fmid, ct_df['rho'] > fmin]) == 2
        rho_lim = ct_df.loc[lim_idx, 'rho'].values
        rho_thr = rho_lim[getelbow(rho_lim)]
        stabilize = True
    elif int(rdaw) == -1:
        lim_idx = utils.andb([ct_df['rho'] < fmid, ct_df['rho'] > fmin]) == 2
        rho_lim = ct_df.loc[lim_idx, 'rho'].values
        rho_thr = rho_lim[getelbow(rho_lim)]

    # Add new columns to comptable for classification
    ct_df['classification'] = 'accepted'
    ct_df['rationale'] = ''

    # Reject if low Kappa, Rho, and variance explained
    is_lowk = ct_df['kappa'] <= kappa_thr
    is_lowr = ct_df['rho'] <= rho_thr
    is_lowe = ct_df['normalized variance explained'] <= eigenvalue_elbow
    is_lowkre = is_lowk & is_lowr & is_lowe
    ct_df.loc[is_lowkre, 'classification'] = 'rejected'
    ct_df.loc[is_lowkre, 'rationale'] += 'low rho, kappa, and varex;'

    # Reject if low variance explained
    is_lows = ct_df['normalized variance explained'] <= varex_norm_min
    ct_df.loc[is_lows, 'classification'] = 'rejected'
    ct_df.loc[is_lows, 'rationale'] += 'low variance explained;'

    # Reject if Kappa over limit
    is_fmax1 = ct_df['kappa'] == F_MAX
    ct_df.loc[is_fmax1, 'classification'] = 'rejected'
    ct_df.loc[is_fmax1, 'rationale'] += 'kappa equals fmax;'

    # Reject if Rho over limit
    is_fmax2 = ct_df['rho'] == F_MAX
    ct_df.loc[is_fmax2, 'classification'] = 'rejected'
    ct_df.loc[is_fmax2, 'rationale'] += 'rho equals fmax;'

    if stabilize:
        temp7 = varex_norm_cum >= 0.95
        ct_df.loc[temp7, 'classification'] = 'rejected'
        ct_df.loc[temp7, 'rationale'] += 'cumulative var. explained above 95%;'
        under_fmin1 = ct_df['kappa'] <= fmin
        ct_df.loc[under_fmin1, 'classification'] = 'rejected'
        ct_df.loc[under_fmin1, 'rationale'] += 'kappa below fmin;'
        under_fmin2 = ct_df['rho'] <= fmin
        ct_df.loc[under_fmin2, 'classification'] = 'rejected'
        ct_df.loc[under_fmin2, 'rationale'] += 'rho below fmin;'

    ct_df.to_csv('comp_table_pca.txt', sep='\t', index=True,
                 index_label='component', float_format='%.6f')

    sel_idx = ct_df['classification'] == 'accepted'
    n_components = np.sum(sel_idx)
    voxel_kept_comp_weighted = (voxel_comp_weights[:, sel_idx] *
                                varex[None, sel_idx])
    kept_data = np.dot(voxel_kept_comp_weighted, comp_ts[sel_idx, :])

    if wvpca:
        kept_data = idwtmat(kept_data, cAl)

    LGR.info('Selected {0} components with Kappa threshold: {1:.02f}, '
             'Rho threshold: {2:.02f}'.format(n_components, kappa_thr,
                                              rho_thr))

    kept_data = stats.zscore(kept_data, axis=1)  # variance normalize timeseries
    kept_data = stats.zscore(kept_data, axis=None)  # variance normalize everything

    return n_components, kept_data


def tedica(n_components, dd, conv, fixed_seed, cost='logcosh'):
    """
    Performs ICA on `dd` and returns mixing matrix

    Parameters
    ----------
    n_components : :obj:`int`
        Number of components retained from PCA decomposition
    dd : (S x T) :obj:`numpy.ndarray`
        Dimensionally reduced optimally combined functional data, where `S` is
        samples and `T` is time
    conv : :obj:`float`
        Convergence limit for ICA
    cost : {'logcosh', 'exp', 'cube'} str, optional
        Cost function for ICA
    fixed_seed : int
        Seed for ensuring reproducibility of ICA results

    Returns
    -------
    mmix : (C x T) :obj:`numpy.ndarray`
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `dd`

    Notes
    -----
    Uses `sklearn` implementation of FastICA for decomposition
    """

    from sklearn.decomposition import FastICA

    if cost not in ('logcosh', 'cube', 'exp'):
        LGR.error('ICA cost function not understood')
        raise

    climit = float(conv)
    if fixed_seed == -1:
        fixed_seed = np.random.randint(low=1, high=1000)
    rand_state = np.random.RandomState(seed=fixed_seed)
    ica = FastICA(n_components=n_components, algorithm='parallel',
                  fun=cost, tol=climit, random_state=rand_state)
    ica.fit(dd)
    mmix = ica.mixing_
    mmix = stats.zscore(mmix, axis=0)
    return mmix, fixed_seed
