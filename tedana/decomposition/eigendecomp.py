"""
Signal decomposition methods for tedana
"""
import pickle
import logging
import warnings
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
def run_mlepca(data):
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


def kundu_tedpca(ct_df, n_echos, kdaw, rdaw, stabilize=False):
    """
    Select PCA components using Kundu's decision tree approach.

    Parameters
    ----------
    ct_df : :obj:`pandas.DataFrame`
        Component table with relevant metrics: kappa, rho, and normalized
        variance explained.
    n_echos : :obj:`int`
        Number of echoes in dataset.
    kdaw : :obj:`float`
        Kappa dimensionality augmentation weight. Must be a non-negative float,
        or -1 (a special value).
    rdaw : :obj:`float`
        Rho dimensionality augmentation weight. Must be a non-negative float,
        or -1 (a special value).
    stabilize : :obj:`bool`, optional
        Whether to stabilize convergence by reducing dimensionality, for low
        quality data. Default is False.

    Returns
    -------
    ct_df : :obj:`pandas.DataFrame`
        Component table with components classified as 'accepted', 'rejected',
        or 'ignored'.
    """
    eigenvalue_elbow = getelbow(ct_df['normalized variance explained'],
                                return_val=True)

    diff_varex_norm = np.abs(np.diff(ct_df['normalized variance explained']))
    lower_diff_varex_norm = diff_varex_norm[(len(diff_varex_norm)//2):]
    varex_norm_thr = np.mean([lower_diff_varex_norm.max(),
                              diff_varex_norm.min()])
    varex_norm_min = ct_df['normalized variance explained'][
        (len(diff_varex_norm)//2) +
        np.arange(len(lower_diff_varex_norm))[lower_diff_varex_norm >= varex_norm_thr][0] + 1]
    varex_norm_cum = np.cumsum(ct_df['normalized variance explained'])

    fmin, fmid, fmax = utils.getfbounds(n_echos)
    if int(kdaw) == -1:
        lim_idx = utils.andb([ct_df['kappa'] < fmid,
                              ct_df['kappa'] > fmin]) == 2
        kappa_lim = ct_df.loc[lim_idx, 'kappa'].values
        kappa_thr = kappa_lim[getelbow(kappa_lim)]

        lim_idx = utils.andb([ct_df['rho'] < fmid, ct_df['rho'] > fmin]) == 2
        rho_lim = ct_df.loc[lim_idx, 'rho'].values
        rho_thr = rho_lim[getelbow(rho_lim)]
        stabilize = True
        LGR.info('kdaw set to -1. Switching TEDPCA method to '
                 'kundu-stabilize')
    elif int(rdaw) == -1:
        lim_idx = utils.andb([ct_df['rho'] < fmid, ct_df['rho'] > fmin]) == 2
        rho_lim = ct_df.loc[lim_idx, 'rho'].values
        rho_thr = rho_lim[getelbow(rho_lim)]
    else:
        kappa_thr = np.average(
            sorted([fmin, getelbow(ct_df['kappa'], return_val=True)/2, fmid]),
            weights=[kdaw, 1, 1])
        rho_thr = np.average(
            sorted([fmin, getelbow_cons(ct_df['rho'], return_val=True)/2, fmid]),
            weights=[rdaw, 1, 1])

    # Reject if low Kappa, Rho, and variance explained
    is_lowk = ct_df['kappa'] <= kappa_thr
    is_lowr = ct_df['rho'] <= rho_thr
    is_lowe = ct_df['normalized variance explained'] <= eigenvalue_elbow
    is_lowkre = is_lowk & is_lowr & is_lowe
    ct_df.loc[is_lowkre, 'classification'] = 'rejected'
    ct_df.loc[is_lowkre, 'rationale'] += 'P001;'

    # Reject if low variance explained
    is_lows = ct_df['normalized variance explained'] <= varex_norm_min
    ct_df.loc[is_lows, 'classification'] = 'rejected'
    ct_df.loc[is_lows, 'rationale'] += 'P002;'

    # Reject if Kappa over limit
    is_fmax1 = ct_df['kappa'] == F_MAX
    ct_df.loc[is_fmax1, 'classification'] = 'rejected'
    ct_df.loc[is_fmax1, 'rationale'] += 'P003;'

    # Reject if Rho over limit
    is_fmax2 = ct_df['rho'] == F_MAX
    ct_df.loc[is_fmax2, 'classification'] = 'rejected'
    ct_df.loc[is_fmax2, 'rationale'] += 'P004;'

    if stabilize:
        temp7 = varex_norm_cum >= 0.95
        ct_df.loc[temp7, 'classification'] = 'rejected'
        ct_df.loc[temp7, 'rationale'] += 'P005;'
        under_fmin1 = ct_df['kappa'] <= fmin
        ct_df.loc[under_fmin1, 'classification'] = 'rejected'
        ct_df.loc[under_fmin1, 'rationale'] += 'P006;'
        under_fmin2 = ct_df['rho'] <= fmin
        ct_df.loc[under_fmin2, 'classification'] = 'rejected'
        ct_df.loc[under_fmin2, 'rationale'] += 'P007;'

    n_components = ct_df.loc[ct_df['classification'] == 'accepted'].shape[0]
    LGR.info('Selected {0} components with Kappa threshold: {1:.02f}, Rho '
             'threshold: {2:.02f}'.format(n_components, kappa_thr, rho_thr))
    return ct_df


def tedpca(catd, OCcatd, combmode, mask, t2s, t2sG,
           ref_img, tes, method='mle', ste=-1, kdaw=10., rdaw=1., wvpca=False,
           verbose=False):
    """
    Use principal components analysis (PCA) to identify and remove thermal
    noise from multi-echo data.

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input functional data
    OCcatd : (S x T) array_like
        Optimally combined time series data
    combmode : {'t2s', 'ste'} str
        How optimal combination of echos should be made, where 't2s' indicates
        using the method of Posse 1999 and 'ste' indicates using the method of
        Poser 2006
    mask : (S,) array_like
        Boolean mask array
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    tes : :obj:`list`
        List of echo times associated with `catd`, in milliseconds
    kdaw : :obj:`float`
        Dimensionality augmentation weight for Kappa calculations
    rdaw : :obj:`float`
        Dimensionality augmentation weight for Rho calculations
    method : {'mle', 'kundu', 'kundu-stabilize'}, optional
        Method with which to select components in TEDPCA. Default is 'mle'.
    ste : :obj:`int` or :obj:`list` of :obj:`int`, optional
        Which echos to use in PCA. Values -1 and 0 are special, where a value
        of -1 will indicate using the optimal combination of the echos
        and 0  will indicate using all the echos. A list can be provided
        to indicate a subset of echos.
        Default: -1
    wvpca : :obj:`bool`, optional
        Whether to apply wavelet denoising to data. Default: False
    verbose : :obj:`bool`, optional
        Whether to output files from fitmodels_direct or not. Default: False

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
        d = OCcatd[mask, :][:, np.newaxis, :]
    elif len(ste) == 1 and ste[0] == 0:
        LGR.info('Computing PCA of spatially concatenated multi-echo data')
        d = catd[mask, ...]
    else:
        LGR.info('Computing PCA of echo #%s' % ','.join([str(ee) for ee in ste]))
        d = np.stack([catd[mask, ee, :] for ee in ste - 1], axis=1)

    eim = np.squeeze(eimask(d))
    d = np.squeeze(d[eim])

    dz = ((d.T - d.T.mean(axis=0)) / d.T.std(axis=0)).T  # var normalize ts
    dz = (dz - dz.mean()) / dz.std()  # var normalize everything

    if wvpca:
        dz, cAl = dwtmat(dz)

    fname = op.abspath('pcastate.pkl')
    if op.exists('pcastate.pkl'):
        LGR.info('Loading PCA from: pcastate.pkl')
        with open('pcastate.pkl', 'rb') as handle:
            pcastate = pickle.load(handle)

        if pcastate['method'] != method:
            LGR.warning('Method from PCA state file ({0}) does not match '
                        'requested method ({1}).'.format(pcastate['method'],
                                                         method))
            state_found = False
        else:
            state_found = True
    else:
        state_found = False

    if not state_found:
        if method == 'mle':
            voxel_comp_weights, varex, comp_ts = run_mlepca(dz)
        else:
            ppca = PCA()
            ppca.fit(dz)
            comp_ts = ppca.components_
            varex = ppca.explained_variance_
            voxel_comp_weights = np.dot(np.dot(dz, comp_ts.T),
                                        np.diag(1. / varex))

        # actual variance explained (normalized)
        varex_norm = varex / varex.sum()

        # Compute K and Rho for PCA comps
        eimum = np.atleast_2d(eim)
        eimum = np.transpose(eimum, np.argsort(eimum.shape)[::-1])
        eimum = eimum.prod(axis=1)
        o = np.zeros((mask.shape[0], *eimum.shape[1:]))
        o[mask, ...] = eimum
        eimum = np.squeeze(o).astype(bool)

        vTmix = comp_ts.T
        vTmixN = ((vTmix.T - vTmix.T.mean(0)) / vTmix.T.std(0)).T
        LGR.info('Making initial component selection guess from PCA results')
        _, ct_df, betasv, v_T = model.fitmodels_direct(
                    catd, comp_ts.T, eimum, t2s, t2sG, tes, combmode, ref_img,
                    mmixN=vTmixN, full_sel=False, label='mepca_',
                    verbose=verbose)
        # varex_norm overrides normalized varex computed by fitmodels_direct
        ct_df['normalized variance explained'] = varex_norm

        pcastate = {'method': method,
                    'voxel_comp_weights': voxel_comp_weights,
                    'varex': varex,
                    'comp_ts': comp_ts,
                    'comptable': ct_df}

        # Save state
        LGR.info('Saving PCA results to: {}'.format(fname))

        try:
            with open(fname, 'wb') as handle:
                pickle.dump(pcastate, handle)
        except TypeError:
            LGR.warning('Could not save PCA solution')
    else:  # if loading existing state
        voxel_comp_weights = pcastate['voxel_comp_weights']
        varex = pcastate['varex']
        comp_ts = pcastate['comp_ts']
        ct_df = pcastate['comptable']

    np.savetxt('mepca_mix.1D', comp_ts.T)

    # write component maps to 4D image
    comp_maps = np.zeros((OCcatd.shape[0], comp_ts.shape[0]))
    for i_comp in range(comp_ts.shape[0]):
        temp_comp_ts = comp_ts[i_comp, :][:, None]
        comp_map = utils.unmask(model.computefeats2(OCcatd, temp_comp_ts, mask), mask)
        comp_maps[:, i_comp] = np.squeeze(comp_map)
    io.filewrite(comp_maps, 'mepca_OC_components.nii', ref_img)

    # Add new columns to comptable for classification
    ct_df['classification'] = 'accepted'
    ct_df['rationale'] = ''

    # Select components using decision tree
    if method == 'kundu':
        ct_df = kundu_tedpca(ct_df, n_echos, kdaw, rdaw, stabilize=False)
    elif method == 'kundu-stabilize':
        ct_df = kundu_tedpca(ct_df, n_echos, kdaw, rdaw, stabilize=True)
    elif method == 'mle':
        LGR.info('Selected {0} components with MLE dimensionality '
                 'detection'.format(ct_df.shape[0]))

    ct_df.to_csv('comp_table_pca.txt', sep='\t', index=True,
                 index_label='component', float_format='%.6f')

    sel_idx = ct_df['classification'] == 'accepted'
    n_components = np.sum(sel_idx)
    voxel_kept_comp_weighted = (voxel_comp_weights[:, sel_idx] *
                                varex[None, sel_idx])
    kept_data = np.dot(voxel_kept_comp_weighted, comp_ts[sel_idx, :])

    if wvpca:
        kept_data = idwtmat(kept_data, cAl)

    kept_data = stats.zscore(kept_data, axis=1)  # variance normalize time series
    kept_data = stats.zscore(kept_data, axis=None)  # variance normalize everything

    return n_components, kept_data


def tedica(n_components, dd, fixed_seed):
    """
    Performs ICA on `dd` and returns mixing matrix

    Parameters
    ----------
    n_components : :obj:`int`
        Number of components retained from PCA decomposition
    dd : (S x T) :obj:`numpy.ndarray`
        Dimensionally reduced optimally combined functional data, where `S` is
        samples and `T` is time
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
    warnings.filterwarnings(action='ignore', module='scipy',
                            message='^internal gelsd')

    if fixed_seed == -1:
        fixed_seed = np.random.randint(low=1, high=1000)
    rand_state = np.random.RandomState(seed=fixed_seed)
    ica = FastICA(n_components=n_components, algorithm='parallel',
                  fun='logcosh', max_iter=5000, random_state=rand_state)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter('always')

        ica.fit(dd)

        w = list(filter(lambda i: issubclass(i.category, UserWarning), w))
        if len(w):
            LGR.warning('ICA failed to converge')

    mmix = ica.mixing_
    mmix = stats.zscore(mmix, axis=0)
    return mmix, fixed_seed
