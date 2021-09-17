"""
PCA and related signal decomposition methods for tedana
"""
import logging
from numbers import Number

import numpy as np
import pandas as pd
from mapca import ma_pca
from scipy import stats
from sklearn.decomposition import PCA

from tedana import io, metrics, utils
from tedana.selection import kundu_tedpca
from tedana.stats import computefeats2

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")


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
    u = np.dot(np.dot(data, v), np.diag(1.0 / s))
    return u, s, v


def tedpca(
    data_cat,
    data_oc,
    combmode,
    mask,
    adaptive_mask,
    t2sG,
    io_generator,
    tes,
    algorithm="mdl",
    kdaw=10.0,
    rdaw=1.0,
    verbose=False,
    low_mem=False,
):
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
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    t2sG : (S,) array_like
        Map of voxel-wise T2* estimates.
    io_generator : :obj:`tedana.io.OutputGenerator`
        The output generation object for this workflow
    tes : :obj:`list`
        List of echo times associated with `data_cat`, in milliseconds
    algorithm : {'kundu', 'kundu-stabilize', 'mdl', 'aic', 'kic', float}, optional
        Method with which to select components in TEDPCA. PCA
        decomposition with the mdl, kic and aic options are based on a Moving Average
        (stationary Gaussian) process and are ordered from most to least aggressive
        (see Li et al., 2007).
        If a float is provided, then it is assumed to represent percentage of variance
        explained (0-1) to retain from PCA.
        Default is 'mdl'.
    kdaw : :obj:`float`, optional
        Dimensionality augmentation weight for Kappa calculations. Must be a
        non-negative float, or -1 (a special value). Default is 10.
    rdaw : :obj:`float`, optional
        Dimensionality augmentation weight for Rho calculations. Must be a
        non-negative float, or -1 (a special value). Default is 1.
    verbose : :obj:`bool`, optional
        Whether to output files from fitmodels_direct or not. Default: False
    low_mem : :obj:`bool`, optional
        Whether to use incremental PCA (for low-memory systems) or not.
        This is only compatible with the "kundu" or "kundu-stabilize" algorithms.
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
    ===========================    =============================================
    Default Filename               Content
    ===========================    =============================================
    desc-PCA_metrics.tsv           PCA component table
    desc-PCA_metrics.json          Metadata sidecar file describing the
                                   component table
    desc-PCA_mixing.tsv            PCA mixing matrix
    desc-PCA_components.nii.gz     Component weight maps
    desc-PCA_decomposition.json    Metadata sidecar file describing the PCA
                                   decomposition
    ===========================    =============================================

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create
        the ``adaptive_mask` parameter.
    :py:mod:`tedana.constants` : The module describing the filenames for
        various naming conventions
    """
    if algorithm == "kundu":
        alg_str = "followed by the Kundu component selection decision tree (Kundu et al., 2013)"
        RefLGR.info(
            "Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., "
            "Vértes, P. E., Inati, S. J., ... & Bullmore, E. T. "
            "(2013). Integrated strategy for improving functional "
            "connectivity mapping using multiecho fMRI. Proceedings "
            "of the National Academy of Sciences, 110(40), "
            "16187-16192."
        )
    elif algorithm == "kundu-stabilize":
        alg_str = (
            "followed by the 'stabilized' Kundu component "
            "selection decision tree (Kundu et al., 2013)"
        )
        RefLGR.info(
            "Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., "
            "Vértes, P. E., Inati, S. J., ... & Bullmore, E. T. "
            "(2013). Integrated strategy for improving functional "
            "connectivity mapping using multiecho fMRI. Proceedings "
            "of the National Academy of Sciences, 110(40), "
            "16187-16192."
        )
    elif isinstance(algorithm, Number):
        alg_str = (
            "in which the number of components was determined based on a "
            "variance explained threshold"
        )
    else:
        alg_str = (
            "based on the PCA component estimation with a Moving Average"
            "(stationary Gaussian) process (Li et al., 2007)"
        )
        RefLGR.info(
            "Li, Y.O., Adalı, T. and Calhoun, V.D., (2007). "
            "Estimating the number of independent components for "
            "functional magnetic resonance imaging data. "
            "Human brain mapping, 28(11), pp.1251-1266."
        )

    RepLGR.info(
        "Principal component analysis {0} was applied to "
        "the optimally combined data for dimensionality "
        "reduction.".format(alg_str)
    )

    n_samp, n_echos, n_vols = data_cat.shape

    LGR.info("Computing PCA of optimally combined multi-echo data")
    data = data_oc[mask, :]

    data_z = ((data.T - data.T.mean(axis=0)) / data.T.std(axis=0)).T  # var normalize ts
    data_z = (data_z - data_z.mean()) / data_z.std()  # var normalize everything

    if algorithm in ["mdl", "aic", "kic"]:
        data_img = io.new_nii_like(io_generator.reference_img, utils.unmask(data, mask))
        mask_img = io.new_nii_like(io_generator.reference_img, mask.astype(int))
        voxel_comp_weights, varex, varex_norm, comp_ts = ma_pca(
            data_img, mask_img, algorithm, normalize=True
        )
    elif isinstance(algorithm, Number):
        ppca = PCA(copy=False, n_components=algorithm, svd_solver="full")
        ppca.fit(data_z)
        comp_ts = ppca.components_.T
        varex = ppca.explained_variance_
        voxel_comp_weights = np.dot(np.dot(data_z, comp_ts), np.diag(1.0 / varex))
        varex_norm = varex / varex.sum()
    elif low_mem:
        voxel_comp_weights, varex, comp_ts = low_mem_pca(data_z)
        varex_norm = varex / varex.sum()
    else:
        ppca = PCA(copy=False, n_components=(n_vols - 1))
        ppca.fit(data_z)
        comp_ts = ppca.components_.T
        varex = ppca.explained_variance_
        voxel_comp_weights = np.dot(np.dot(data_z, comp_ts), np.diag(1.0 / varex))
        varex_norm = varex / varex.sum()

    # Compute Kappa and Rho for PCA comps
    required_metrics = [
        "kappa",
        "rho",
        "countnoise",
        "countsigFT2",
        "countsigFS0",
        "dice_FT2",
        "dice_FS0",
        "signal-noise_t",
        "variance explained",
        "normalized variance explained",
        "d_table_score",
    ]
    comptable = metrics.collect.generate_metrics(
        data_cat,
        data_oc,
        comp_ts,
        adaptive_mask,
        tes,
        io_generator,
        "PCA",
        metrics=required_metrics,
    )

    # varex_norm from PCA overrides varex_norm from dependence_metrics,
    # but we retain the original
    comptable["estimated normalized variance explained"] = comptable[
        "normalized variance explained"
    ]
    comptable["normalized variance explained"] = varex_norm

    # write component maps to 4D image
    comp_maps = utils.unmask(computefeats2(data_oc, comp_ts, mask), mask)
    io_generator.save_file(comp_maps, "z-scored PCA components img")

    # Select components using decision tree
    if algorithm == "kundu":
        comptable, metric_metadata = kundu_tedpca(
            comptable,
            n_echos,
            kdaw,
            rdaw,
            stabilize=False,
        )
    elif algorithm == "kundu-stabilize":
        comptable, metric_metadata = kundu_tedpca(
            comptable,
            n_echos,
            kdaw,
            rdaw,
            stabilize=True,
        )
    else:
        alg_str = "variance explained-based" if isinstance(algorithm, Number) else algorithm
        LGR.info(
            "Selected {0} components with {1} dimensionality "
            "detection".format(comptable.shape[0], alg_str)
        )
        comptable["classification"] = "accepted"
        comptable["rationale"] = ""

    # Save decomposition files
    comp_names = [
        io.add_decomp_prefix(comp, prefix="pca", max_value=comptable.index.max())
        for comp in comptable.index.values
    ]

    mixing_df = pd.DataFrame(data=comp_ts, columns=comp_names)
    io_generator.save_file(mixing_df, "PCA mixing tsv")

    # Save component table and associated json
    io_generator.save_file(comptable, "PCA metrics tsv")

    metric_metadata = metrics.collect.get_metadata(comptable)
    io_generator.save_file(metric_metadata, "PCA metrics json")

    decomp_metadata = {
        "Method": (
            "Principal components analysis implemented by sklearn. "
            "Components are sorted by variance explained in descending order. "
        ),
    }
    for comp_name in comp_names:
        decomp_metadata[comp_name] = {
            "Description": "PCA fit to optimally combined data.",
            "Method": "tedana",
        }
    io_generator.save_file(decomp_metadata, "PCA decomposition json")

    acc = comptable[comptable.classification == "accepted"].index.values
    n_components = acc.size
    voxel_kept_comp_weighted = voxel_comp_weights[:, acc] * varex[None, acc]
    kept_data = np.dot(voxel_kept_comp_weighted, comp_ts[:, acc].T)

    kept_data = stats.zscore(kept_data, axis=1)  # variance normalize time series
    kept_data = stats.zscore(kept_data, axis=None)  # variance normalize everything

    return kept_data, n_components
