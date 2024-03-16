"""PCA and related signal decomposition methods for tedana."""

import logging
from numbers import Number

import numpy as np
import pandas as pd
from mapca import MovingAveragePCA
from scipy import stats
from sklearn.decomposition import PCA

from tedana import io, metrics, utils
from tedana.reporting import pca_results as plot_pca_results
from tedana.selection import kundu_tedpca
from tedana.stats import computefeats2

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def low_mem_pca(data):
    """Run Singular Value Decomposition (SVD) on input data.

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
    varex_norm : array-like, shape (n_components,)
        Explained variance ratio.
    v : (C x T) array_like
        Component timeseries.
    """
    from sklearn.decomposition import IncrementalPCA

    ppca = IncrementalPCA(n_components=(data.shape[-1] - 1))
    ppca.fit(data)
    v = ppca.components_.T
    s = ppca.explained_variance_
    u = np.dot(np.dot(data, v), np.diag(1.0 / s))
    varex_norm = ppca.explained_variance_ratio_
    return u, s, varex_norm, v


def tedpca(
    data_cat,
    data_oc,
    mask,
    adaptive_mask,
    io_generator,
    tes,
    algorithm="aic",
    kdaw=10.0,
    rdaw=1.0,
    low_mem=False,
):
    r"""Use principal components analysis (PCA) to identify and remove thermal noise from data.

    Parameters
    ----------
    data_cat : (S x E x T) array_like
        Input functional data
    data_oc : (S x T) array_like
        Optimally combined time series data
    mask : (S,) array_like
        Boolean mask array
    adaptive_mask : (S,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    io_generator : :obj:`tedana.io.OutputGenerator`
        The output generation object for this workflow
    tes : :obj:`list`
        List of echo times associated with `data_cat`, in milliseconds
    algorithm : {'kundu', 'kundu-stabilize', 'mdl', 'aic', 'kic', float}, optional
        Method with which to select components in TEDPCA. PCA
        decomposition with the mdl, kic and aic options are based on a Moving Average
        (stationary Gaussian) process and are ordered from most to least aggressive
        (see :footcite:p:`li2007estimating`).
        If a float is provided, then it is assumed to represent percentage of variance
        explained (0-1) to retain from PCA.
        If an int is provided, then it is assumed to be the number of components
        to select
        Default is 'aic'.
    kdaw : :obj:`float`, optional
        Dimensionality augmentation weight for Kappa calculations when `algorithm` is
        'kundu'. Must be a non-negative float, or -1 (a special value). Default is 10.
    rdaw : :obj:`float`, optional
        Dimensionality augmentation weight for Rho calculations when `algorithm` is
        'kundu'. Must be a non-negative float, or -1 (a special value). Default is 1.
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
    :math:`\kappa`            Component pseudo-F statistic for TE-dependent
                              (BOLD) model.
    :math:`\rho`              Component pseudo-F statistic for TE-independent
                              (artifact) model.
    :math:`v`                 Voxel
    :math:`V`                 Total number of voxels in mask
    :math:`\zeta`             Something
    :math:`c`                 Component
    :math:`p`                 Something else
    ======================    =================================================

    Steps:

    1.  Variance normalize either multi-echo or optimally combined data,
        depending on settings.
    2.  Decompose normalized data using PCA or SVD.
    3.  Compute :math:`{\kappa}` and :math:`{\rho}`:

            .. math::
                {\kappa}_c = \frac{\sum_{v}^V {\zeta}_{c,v}^p * \
                    F_{c,v,R_2^*}}{\sum {\zeta}_{c,v}^p}

                {\rho}_c = \frac{\sum_{v}^V {\zeta}_{c,v}^p * \
                    F_{c,v,S_0}}{\sum {\zeta}_{c,v}^p}

    4.  Some other stuff. Something about elbows.
    5.  Classify components as thermal noise if they meet both of the
        following criteria:

            - Nonsignificant :math:`{\kappa}` and :math:`{\rho}`.
            - Nonsignificant variance explained.
    Generated Files
    ---------------

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

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create
        the ``adaptive_mask`` parameter.
    :py:mod:`tedana.constants` : The module describing the filenames for
        various naming conventions
    """
    if algorithm == "kundu":
        alg_str = (
            "followed by the Kundu component selection decision tree \\citep{kundu2013integrated}"
        )
    elif algorithm == "kundu-stabilize":
        alg_str = (
            "followed by the 'stabilized' Kundu component "
            "selection decision tree \\citep{kundu2013integrated}"
        )
    elif isinstance(algorithm, Number):
        if isinstance(algorithm, float):
            alg_str = (
                "in which the number of components was determined based on a "
                "variance explained threshold"
            )
        else:
            alg_str = "in which the number of components is pre-defined"
    else:
        alg_str = (
            "based on the PCA component estimation with a Moving Average"
            "(stationary Gaussian) process \\citep{li2007estimating}"
        )

    RepLGR.info(
        f"Principal component analysis {alg_str} was applied to "
        "the optimally combined data for dimensionality "
        "reduction."
    )

    n_samp, n_echos, n_vols = data_cat.shape

    LGR.info(
        f"Computing PCA of optimally combined multi-echo data with selection criteria: {algorithm}"
    )
    data = data_oc[mask, :]

    data_z = ((data.T - data.T.mean(axis=0)) / data.T.std(axis=0)).T  # var normalize ts
    data_z = (data_z - data_z.mean()) / data_z.std()  # var normalize everything

    if algorithm in ["mdl", "aic", "kic"]:
        data_img = io.new_nii_like(io_generator.reference_img, utils.unmask(data, mask))
        mask_img = io.new_nii_like(io_generator.reference_img, mask.astype(int))
        ma_pca = MovingAveragePCA(criterion=algorithm, normalize=True)
        _ = ma_pca.fit_transform(data_img, mask_img)

        # Extract results from maPCA
        voxel_comp_weights = ma_pca.u_
        varex = ma_pca.explained_variance_
        varex_norm = ma_pca.explained_variance_ratio_
        comp_ts = ma_pca.components_.T
        aic = ma_pca.aic_
        kic = ma_pca.kic_
        mdl = ma_pca.mdl_
        varex_90 = ma_pca.varexp_90_
        varex_95 = ma_pca.varexp_95_
        all_comps = ma_pca.all_

        # Extract number of components and variance explained for logging and plotting
        n_aic = aic["n_components"]
        aic_varexp = np.round(aic["explained_variance_total"], 3)
        n_kic = kic["n_components"]
        kic_varexp = np.round(kic["explained_variance_total"], 3)
        n_mdl = mdl["n_components"]
        mdl_varexp = np.round(mdl["explained_variance_total"], 3)
        n_varex_90 = varex_90["n_components"]
        varex_90_varexp = np.round(varex_90["explained_variance_total"], 3)
        n_varex_95 = varex_95["n_components"]
        varex_95_varexp = np.round(varex_95["explained_variance_total"], 3)
        all_varex = np.round(all_comps["explained_variance_total"], 3)

        # Print out the results
        LGR.info("Optimal number of components based on different criteria:")
        LGR.info(
            f"AIC: {n_aic} | KIC: {n_kic} | MDL: {n_mdl} | 90% varexp: {n_varex_90} "
            f"| 95% varexp: {n_varex_95}"
        )

        LGR.info("Explained variance based on different criteria:")
        LGR.info(
            f"AIC: {aic_varexp}% | KIC: {kic_varexp}% | MDL: {mdl_varexp}% | "
            f"90% varexp: {varex_90_varexp}% | 95% varexp: {varex_95_varexp}%"
        )

        pca_optimization_curves = np.array([aic["value"], kic["value"], mdl["value"]])
        pca_criteria_components = np.array(
            [
                n_aic,
                n_kic,
                n_mdl,
                n_varex_90,
                n_varex_95,
            ]
        )

        # Plot maPCA optimization curves
        LGR.info("Plotting maPCA optimization curves")
        plot_pca_results(pca_optimization_curves, pca_criteria_components, all_varex, io_generator)

        # Save maPCA results into a dictionary
        mapca_results = {
            "aic": {
                "n_components": n_aic,
                "explained_variance_total": aic_varexp,
                "curve": aic["value"],
            },
            "kic": {
                "n_components": n_kic,
                "explained_variance_total": kic_varexp,
                "curve": kic["value"],
            },
            "mdl": {
                "n_components": n_mdl,
                "explained_variance_total": mdl_varexp,
                "curve": mdl["value"],
            },
            "varex_90": {
                "n_components": n_varex_90,
                "explained_variance_total": varex_90_varexp,
            },
            "varex_95": {
                "n_components": n_varex_95,
                "explained_variance_total": varex_95_varexp,
            },
        }
        if "subsampling_" in dir(ma_pca):
            # Since older version of MAPCA did not log these values
            # Check before trying to access the values. This will be
            # unnecessary and removal once logging these values gets
            # a new version number in MAPCA and tedana updates its
            # minimum MAPCA version
            mapca_results["MAPCA_subsampling"] = {
                "calculated_IID_subsample_depth": ma_pca.subsampling_[
                    "calculated_IID_subsample_depth"
                ],
                "calculated_IID_subsample_mean": ma_pca.subsampling_[
                    "calculated_IID_subsample_mean"
                ],
                "effective_num_IID_samples": ma_pca.subsampling_["effective_num_IID_samples"],
                "total_num_samples": ma_pca.subsampling_["total_num_samples"],
            }

        # Save dictionary
        io_generator.save_file(mapca_results, "PCA cross component metrics json")

    elif isinstance(algorithm, Number):
        ppca = PCA(copy=False, n_components=algorithm, svd_solver="full")
        ppca.fit(data_z)
        comp_ts = ppca.components_.T
        varex = ppca.explained_variance_
        voxel_comp_weights = np.dot(np.dot(data_z, comp_ts), np.diag(1.0 / varex))
        varex_norm = ppca.explained_variance_ratio_
    elif low_mem:
        voxel_comp_weights, varex, varex_norm, comp_ts = low_mem_pca(data_z)
    else:
        # If algorithm is kundu or kundu-stablize component metrics
        # are calculated without dimensionality estimation and
        # reduction and then kundu identifies components that are
        # to be accepted or rejected
        ppca = PCA(copy=False, n_components=(n_vols - 1))
        ppca.fit(data_z)
        comp_ts = ppca.components_.T
        varex = ppca.explained_variance_
        voxel_comp_weights = np.dot(np.dot(data_z, comp_ts), np.diag(1.0 / varex))
        varex_norm = ppca.explained_variance_ratio_

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
        if isinstance(algorithm, float):
            alg_str = "variance explained-based"
        elif isinstance(algorithm, int):
            alg_str = "a fixed number of components and no"
        else:
            alg_str = algorithm
        LGR.info(
            f"Selected {comptable.shape[0]} components with {round(100 * varex_norm.sum(), 2)}% "
            f"normalized variance explained using {alg_str} dimensionality estimate"
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
