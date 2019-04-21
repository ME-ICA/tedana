"""
Signal decomposition methods for tedana
"""
import logging

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from tedana import model, utils, io
from tedana.decomposition._utils import eimask
from tedana.selection import kundu_tedpca
from tedana.due import due, BibTeX

LGR = logging.getLogger(__name__)


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
    v : (T x C) array_like
        Component timeseries.
    """
    # do PC dimension selection and get eigenvalue cutoff
    ppca = PCA(n_components='mle', svd_solver='full')
    ppca.fit(data)
    v = ppca.components_.T
    s = ppca.explained_variance_
    u = np.dot(np.dot(data, v), np.diag(1. / s))
    return u, s, v


def tedpca(data_cat, data_oc, combmode, mask, t2s, t2sG, ref_img, tes, method='mle',
           ste=-1, kdaw=10., rdaw=1., out_dir='.', verbose=False):
    """
    Use principal components analysis (PCA) to identify and remove thermal
    noise from multi-echo data.

    Parameters
    ----------
    data_cat : (S x E x T) array_like
        Input functional data
    data_oc : (S x T) array_like
        Optimally combined time series data
    combmode : :obj:`str`
        Combination method
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
    method : {'mle', 'kundu', 'kundu-stabilize'}, optional
        Method with which to select components in TEDPCA. Default is 'mle'.
    ste : :obj:`int` or :obj:`list` of :obj:`int`, optional
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

    Returns
    -------
    n_components : :obj:`int`
        Number of components retained from PCA decomposition
    kept_data : (S x T) :obj:`numpy.ndarray`
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

    n_samp, n_echos, n_vols = data_cat.shape
    ste = np.array([int(ee) for ee in str(ste).split(',')])

    if len(ste) == 1 and ste[0] == -1:
        LGR.info('Computing PCA of optimally combined multi-echo data')
        data = data_oc[mask, :][:, np.newaxis, :]
    elif len(ste) == 1 and ste[0] == 0:
        LGR.info('Computing PCA of spatially concatenated multi-echo data')
        data = data_cat[mask, ...]
    else:
        LGR.info('Computing PCA of echo #{0}'.format(','.join([str(ee) for ee in ste])))
        data = np.stack([data_cat[mask, ee, :] for ee in ste - 1], axis=1)

    eim = np.squeeze(eimask(data))
    data = np.squeeze(data[eim])

    data_z = ((data.T - data.T.mean(axis=0)) / data.T.std(axis=0)).T  # var normalize ts
    data_z = (data_z - data_z.mean()) / data_z.std()  # var normalize everything

    if method == 'mle':
        voxel_comp_weights, varex, comp_ts = run_mlepca(data_z)
    else:
        ppca = PCA()
        ppca.fit(data_z)
        comp_ts = ppca.components_.T
        varex = ppca.explained_variance_
        voxel_comp_weights = np.dot(np.dot(data_z, comp_ts),
                                    np.diag(1. / varex))

    # actual variance explained (normalized)
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
    _, comptable, _, _ = model.fitmodels_direct(
                data_cat, comp_ts, eimum, t2s, t2sG, tes, combmode, ref_img,
                reindex=False, mmixN=vTmixN, full_sel=False,
                label='mepca_', out_dir=out_dir, verbose=verbose)
    # varex_norm overrides normalized varex computed by fitmodels_direct
    comptable['real normalized variance explained'] = varex_norm

    np.savetxt('mepca_mix.1D', comp_ts)

    # write component maps to 4D image
    comp_maps = np.zeros((data_oc.shape[0], comp_ts.shape[1]))
    for i_comp in range(comp_ts.shape[1]):
        temp_comp_ts = comp_ts[:, i_comp][:, None]
        comp_map = utils.unmask(model.computefeats2(data_oc, temp_comp_ts, mask), mask)
        comp_maps[:, i_comp] = np.squeeze(comp_map)
    io.filewrite(comp_maps, 'mepca_OC_components.nii', ref_img)

    # Select components using decision tree
    if method == 'kundu':
        comptable = kundu_tedpca(comptable, n_echos, kdaw, rdaw, stabilize=False)
    elif method == 'kundu-stabilize':
        comptable = kundu_tedpca(comptable, n_echos, kdaw, rdaw, stabilize=True)
    elif method == 'mle':
        LGR.info('Selected {0} components with MLE dimensionality '
                 'detection'.format(comptable.shape[0]))
        comptable['classification'] = 'accepted'
        comptable['rationale'] = ''

    comptable.to_csv('comp_table_pca.txt', sep='\t', index=True,
                     index_label='component', float_format='%.6f')

    acc = comptable[comptable.classification == 'accepted'].index.values
    n_components = acc.size
    voxel_kept_comp_weighted = (voxel_comp_weights[:, acc] *
                                varex[None, acc])
    kept_data = np.dot(voxel_kept_comp_weighted, comp_ts[:, acc].T)

    kept_data = stats.zscore(kept_data, axis=1)  # variance normalize time series
    kept_data = stats.zscore(kept_data, axis=None)  # variance normalize everything

    return kept_data, n_components
