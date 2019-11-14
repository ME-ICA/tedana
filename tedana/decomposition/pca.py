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
from tedana.decomposition import (gift_pca, _utils)
from tedana.stats import computefeats2
from tedana.selection import kundu_tedpca
from tedana.due import due, BibTeX

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


@due.dcite(BibTeX("""
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
    Run Singular Value Decomposition (SVD) on input data,
    automatically select components on MLE variance cut-off.

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
    v : (T x C) array_like
        Component timeseries.
    """
    # do PC dimension selection and get eigenvalue cutoff
    ppca = PCA(n_components='mle', svd_solver='full', copy=False)
    ppca.fit(data)
    v = ppca.components_.T
    s = ppca.explained_variance_
    u = np.dot(np.dot(data, v), np.diag(1. / s))
    varex_norm = ppca.explained_variance_ratio_
    return u, s, varex_norm, v


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


def tedpca(data_cat,
           data_oc,
           combmode,
           mask,
           t2s,
           t2sG,
           ref_img,
           tes,
           algorithm='mdl',
           source_tes=-1,
           kdaw=10.,
           rdaw=1.,
           out_dir='.',
           verbose=False,
           low_mem=False):
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
    t2s : (S,) array_like
        Map of voxel-wise T2* estimates.
    t2sG : (S,) array_like
        Map of voxel-wise T2* estimates.
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    tes : :obj:`list`
        List of echo times associated with `data_cat`, in milliseconds
    algorithm : {'mle', 'kundu', 'kundu-stabilize', 'mdl', 'aic', 'kic'}, optional
        Method with which to select components in TEDPCA. Default is 'mdl'.
    source_tes : :obj:`int` or :obj:`list` of :obj:`int`, optional
        Which echos to use in PCA. Values -1 and 0 are special, where a value
        of -1 will indicate using the optimal combination of the echos
        and 0  will indicate using all the echos. A list can be provided
        to indicate a subset of echos.
        Default: -1
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
    if low_mem and algorithm == 'mle':
        LGR.warning('Low memory option is not compatible with MLE '
                    'dimensionality estimation. Switching to Kundu decision '
                    'tree.')
        algorithm = 'kundu'

    if algorithm == 'mle':
        alg_str = "using MLE dimensionality estimation (Minka, 2001)"
        RefLGR.info("Minka, T. P. (2001). Automatic choice of dimensionality "
                    "for PCA. In Advances in neural information processing "
                    "systems (pp. 598-604).")
    elif algorithm == 'kundu':
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
        alg_str = ("based on the PCA components estimation in the GIFT software "
                   "(Li et al., 2007)")
        RefLGR.info("Li, Y.O., Adalı, T. and Calhoun, V.D., (2007). "
                    "Estimating the number of independent components for "
                    "functional magnetic resonance imaging data. "
                    "Human brain mapping, 28(11), pp.1251-1266.")

    if source_tes == -1:
        dat_str = "the optimally combined data"
    elif source_tes == 0:
        dat_str = "the z-concatenated multi-echo data"
    else:
        dat_str = "a z-concatenated subset of echoes from the input data"

    RepLGR.info("Principal component analysis {0} was applied to "
                "{1} for dimensionality reduction.".format(alg_str, dat_str))

    n_samp, n_echos, n_vols = data_cat.shape
    source_tes = np.array([int(ee) for ee in str(source_tes).split(',')])

    if len(source_tes) == 1 and source_tes[0] == -1:
        LGR.info('Computing PCA of optimally combined multi-echo data')
        data = data_oc[mask, :][:, np.newaxis, :]
    elif len(source_tes) == 1 and source_tes[0] == 0:
        LGR.info('Computing PCA of spatially concatenated multi-echo data')
        data = data_cat[mask, ...]
    else:
        LGR.info('Computing PCA of echo #{0}'.format(','.join([str(ee) for ee in source_tes])))
        data = np.stack([data_cat[mask, ee, :] for ee in source_tes - 1], axis=1)

    eim = np.squeeze(_utils.eimask(data))
    data = np.squeeze(data[eim])

    data_z = ((data.T - data.T.mean(axis=0)) / data.T.std(axis=0)).T  # var normalize ts
    data_z = (data_z - data_z.mean()) / data_z.std()  # var normalize everything

    if algorithm in ['mdl', 'aic', 'kic']:
        data_img = io.new_nii_like(
            ref_img, utils.unmask(utils.unmask(data, eim), mask))
        mask_img = io.new_nii_like(ref_img,
                                   utils.unmask(eim, mask).astype(int))
        voxel_comp_weights, varex, varex_norm, comp_ts = gift_pca.run_gift_pca(
            data_img, mask_img, algorithm)
    elif algorithm == 'mle':
        voxel_comp_weights, varex, varex_norm, comp_ts = run_mlepca(data_z)
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
    eimum = np.atleast_2d(eim)
    eimum = np.transpose(eimum, np.argsort(eimum.shape)[::-1])
    eimum = eimum.prod(axis=1)
    o = np.zeros((mask.shape[0], *eimum.shape[1:]))
    o[mask, ...] = eimum
    eimum = np.squeeze(o).astype(bool)

    # Normalize each component's time series
    vTmixN = stats.zscore(comp_ts, axis=0)
    comptable, _, _, _ = metrics.dependence_metrics(data_cat,
                                                    data_oc,
                                                    comp_ts,
                                                    t2s,
                                                    tes,
                                                    ref_img,
                                                    reindex=False,
                                                    mmixN=vTmixN,
                                                    algorithm=None,
                                                    label='mepca_',
                                                    out_dir=out_dir,
                                                    verbose=verbose)

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
    elif algorithm == 'mle':
        LGR.info('Selected {0} components with MLE dimensionality '
                 'detection'.format(comptable.shape[0]))
        comptable['classification'] = 'accepted'
        comptable['rationale'] = ''

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

    data_type = 'optimally combined data' if source_tes == -1 else 'z-concatenated data'
    comptable['Description'] = 'PCA fit to {0}.'.format(data_type)
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

    kept_data = stats.zscore(kept_data,
                             axis=1)  # variance normalize time series
    kept_data = stats.zscore(kept_data,
                             axis=None)  # variance normalize everything

    return kept_data, n_components
