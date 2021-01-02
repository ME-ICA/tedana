"""
PCA and related signal decomposition methods for tedana
"""
import logging
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from tedana import metrics, utils, io
from tedana.decomposition import ma_pca
from tedana.stats import computefeats2
from tedana.selection import kundu_tedpca

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def low_mem_pca(data):
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
    from sklearn.decomposition import IncrementalPCA
    ppca = IncrementalPCA(n_components=(data.shape[-1] - 1))
    ppca.fit(data)
    v = ppca.components_.T
    s = ppca.explained_variance_
    u = np.dot(np.dot(data, v), np.diag(1. / s))
    return u, s, v


def tedpca(data_cat, data_oc, combmode, mask, adaptive_mask, t2sG,
           ref_img, tes, algorithm='mdl', kdaw=10., rdaw=1.,
           out_dir='.', verbose=False, low_mem=False):
    """
    Use principal components analysis (PCA) to identify and remove thermal
    noise from multi-echo data.

    Parameters
    ----------
    data_cat : (S x E x T) array_like
        Input functional data
    data_oc : (S x T) array_like
        Optimally combined time series data
    combmode : {'t2s', 'paid'} str
        How optimal combination of echos should be made, where 't2s' indicates
        using the method of Posse 1999 and 'paid' indicates using the method of
        Poser 2006
    mask : (S,) array_like
        Boolean mask array
    adaptive_mask : (S,) array_like
        Adaptive mask of the data indicating the number of echos with signal at each voxel
    t2sG : (S,) array_like
        Map of voxel-wise T2* estimates.
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    tes : :obj:`list`
        List of echo times associated with `data_cat`, in milliseconds
    algorithm : {'kundu', 'kundu-stabilize', 'mdl', 'aic', 'kic'}, optional
        Method with which to select components in TEDPCA. Default is 'mdl'. PCA
        decomposition with the mdl, kic and aic options are based on a Moving Average
        (stationary Gaussian) process and are ordered from most to least aggresive.
        See (Li et al., 2007).
    kdaw : :obj:`float`, optional
        Dimensionality augmentation weight for Kappa calculations. Must be a
        non-negative float, or -1 (a special value). Default is 10.
    rdaw : :obj:`float`, optional
        Dimensionality augmentation weight for Rho calculations. Must be a
        non-negative float, or -1 (a special value). Default is 1.
    out_dir : :obj:`str`, optional
        Output directory.
    verbose : :obj:`bool`, optional
        Whether to output files from fitmodels_direct or not. Default: False
    low_mem : :obj:`bool`, optional
        Whether to use incremental PCA (for low-memory systems) or not.
        Default: False

    Returns
    -------
    kept_data : (S x T) :obj:`numpy.ndarray`
        Dimensionally reduced optimally combined functional data
    n_components : :obj:`int`
        Number of components retained from PCA decomposition

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
                {\\kappa}_c = \\frac{\\sum_{v}^V {\\zeta}_{c,v}^p * \
                      F_{c,v,R_2^*}}{\\sum {\\zeta}_{c,v}^p}

                {\\rho}_c = \\frac{\\sum_{v}^V {\\zeta}_{c,v}^p * \
                      F_{c,v,S_0}}{\\sum {\\zeta}_{c,v}^p}

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
    pca_decomposition.json    PCA component table.
    pca_mixing.tsv            PCA mixing matrix.
    pca_components.nii.gz     Component weight maps.
    ======================    =================================================
    """
    if algorithm == 'kundu':
        alg_str = ("followed by the Kundu component selection decision "
                   "tree (Kundu et al., 2013)")
        RefLGR.info("Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., "
                    "Vértes, P. E., Inati, S. J., ... & Bullmore, E. T. "
                    "(2013). Integrated strategy for improving functional "
                    "connectivity mapping using multiecho fMRI. Proceedings "
                    "of the National Academy of Sciences, 110(40), "
                    "16187-16192.")
    elif algorithm == 'kundu-stabilize':
        alg_str = ("followed by the 'stabilized' Kundu component "
                   "selection decision tree (Kundu et al., 2013)")
        RefLGR.info("Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., "
                    "Vértes, P. E., Inati, S. J., ... & Bullmore, E. T. "
                    "(2013). Integrated strategy for improving functional "
                    "connectivity mapping using multiecho fMRI. Proceedings "
                    "of the National Academy of Sciences, 110(40), "
                    "16187-16192.")
    else:
        alg_str = ("based on the PCA component estimation with a Moving Average"
                   "(stationary Gaussian) process (Li et al., 2007)")
        RefLGR.info("Li, Y.O., Adalı, T. and Calhoun, V.D., (2007). "
                    "Estimating the number of independent components for "
                    "functional magnetic resonance imaging data. "
                    "Human brain mapping, 28(11), pp.1251-1266.")

    RepLGR.info("Principal component analysis {0} was applied to "
                "the optimally combined data for dimensionality "
                "reduction.".format(alg_str))

    n_samp, n_echos, n_vols = data_cat.shape

    LGR.info('Computing PCA of optimally combined multi-echo data')
    data = data_oc[mask, :]

    data_z = ((data.T - data.T.mean(axis=0)) / data.T.std(axis=0)).T  # var normalize ts
    data_z = (data_z - data_z.mean()) / data_z.std()  # var normalize everything

    if algorithm in ['mdl', 'aic', 'kic']:
        data_img = io.new_nii_like(ref_img, utils.unmask(data, mask))
        mask_img = io.new_nii_like(ref_img, mask.astype(int))
        voxel_comp_weights, varex, varex_norm, comp_ts = ma_pca.ma_pca(
            data_img, mask_img, algorithm)
    elif low_mem:
        voxel_comp_weights, varex, comp_ts = low_mem_pca(data_z)
        varex_norm = varex / varex.sum()
    else:
        ppca = PCA(copy=False, n_components=(n_vols - 1))
        ppca.fit(data_z)
        comp_ts = ppca.components_.T
        varex = ppca.explained_variance_
        voxel_comp_weights = np.dot(np.dot(data_z, comp_ts),
                                    np.diag(1. / varex))
        varex_norm = varex / varex.sum()

    # Compute Kappa and Rho for PCA comps
    # Normalize each component's time series
    vTmixN = stats.zscore(comp_ts, axis=0)
    comptable, _, _, _ = metrics.dependence_metrics(
                data_cat, data_oc, comp_ts, adaptive_mask, tes, ref_img,
                reindex=False, mmixN=vTmixN, algorithm=None,
                label='mepca_', out_dir=out_dir, verbose=verbose)

    # varex_norm from PCA overrides varex_norm from dependence_metrics,
    # but we retain the original
    comptable['estimated normalized variance explained'] = \
        comptable['normalized variance explained']
    comptable['normalized variance explained'] = varex_norm

    # write component maps to 4D image
    comp_ts_z = stats.zscore(comp_ts, axis=0)
    comp_maps = utils.unmask(computefeats2(data_oc, comp_ts_z, mask), mask)
    io.filewrite(comp_maps, op.join(out_dir, 'pca_components.nii.gz'), ref_img)

    # Select components using decision tree
    if algorithm == 'kundu':
        comptable = kundu_tedpca(comptable, n_echos, kdaw, rdaw, stabilize=False)
    elif algorithm == 'kundu-stabilize':
        comptable = kundu_tedpca(comptable, n_echos, kdaw, rdaw, stabilize=True)
    elif algorithm in ['mdl', 'aic', 'kic']:
        LGR.info('Selected {0} components with {1} dimensionality '
                 'detection'.format(comptable.shape[0], algorithm))
        comptable['classification'] = 'accepted'
        comptable['rationale'] = ''

    # Save decomposition
    comp_names = [io.add_decomp_prefix(comp, prefix='pca', max_value=comptable.index.max())
                  for comp in comptable.index.values]

    mixing_df = pd.DataFrame(data=comp_ts, columns=comp_names)
    mixing_df.to_csv(op.join(out_dir, 'pca_mixing.tsv'), sep='\t', index=False)

    comptable['Description'] = 'PCA fit to optimally combined data.'
    mmix_dict = {}
    mmix_dict['Method'] = ('Principal components analysis implemented by '
                           'sklearn. Components are sorted by variance '
                           'explained in descending order. '
                           'Component signs are flipped to best match the '
                           'data.')
    io.save_comptable(comptable, op.join(out_dir, 'pca_decomposition.json'),
                      label='pca', metadata=mmix_dict)

    acc = comptable[comptable.classification == 'accepted'].index.values
    n_components = acc.size
    voxel_kept_comp_weighted = (voxel_comp_weights[:, acc] * varex[None, acc])
    kept_data = np.dot(voxel_kept_comp_weighted, comp_ts[:, acc].T)

    kept_data = stats.zscore(kept_data, axis=1)  # variance normalize time series
    kept_data = stats.zscore(kept_data, axis=None)  # variance normalize everything

    return kept_data, n_components
