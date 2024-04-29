"""Global signal control methods."""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import lpmv

from tedana import io, utils

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def gscontrol_raw(
    *,
    data_cat: np.ndarray,
    data_optcom: np.ndarray,
    n_echos: int,
    io_generator: io.OutputGenerator,
    dtrank: int = 4,
):
    """Remove global signal from individual echo ``data_cat`` and ``data_optcom`` time series.

    This function uses the spatial global signal estimation approach to
    to removal global signal out of individual echo time series datasets. The
    spatial global signal is estimated from the optimally combined data after
    detrending with a Legendre polynomial basis of ``order = 0`` and
    ``degree = dtrank``.

    Parameters
    ----------
    data_cat : (S x E x T) array_like
        Input functional data
    data_optcom : (S x T) array_like
        Optimally combined functional data (i.e., the output of `make_optcom`)
    n_echos : :obj:`int`
        Number of echos in data. Should be the same as ``E`` dimension of ``data_cat``
    io_generator : :obj:`tedana.io.OutputGenerator`
        The output generator for this workflow
    dtrank : :obj:`int`, optional
        Specifies degree of Legendre polynomial basis function for estimating
        spatial global signal. Default: 4

    Returns
    -------
    data_cat_nogs : (S x E x T) array_like
        Input ``data_cat`` with global signal removed from time series.
    data_optcom_nogs : (S x T) array_like
        Input ``data_optcom`` with global signal removed from time series.
    """
    LGR.info("Applying amplitude-based T1 equilibration correction")
    RepLGR.info(
        "Global signal regression was applied to the multi-echo and optimally combined datasets."
    )
    if data_cat.shape[0] != data_optcom.shape[0]:
        raise ValueError(
            f"First dimensions of data_cat ({data_cat.shape[0]}) and "
            f"data_optcom ({data_optcom.shape[0]}) do not match"
        )
    elif data_cat.shape[1] != n_echos:
        raise ValueError(
            f"Second dimension of data_cat ({data_cat.shape[1]}) "
            f"does not match n_echos ({n_echos})"
        )
    elif data_cat.shape[2] != data_optcom.shape[1]:
        raise ValueError(
            f"Third dimension of data_cat ({data_cat.shape[2]}) does not match second dimension "
            f"of data_optcom ({data_optcom.shape[1]})"
        )

    io_generator.save_file(data_optcom, "has gs combined img")

    # Legendre polynomial basis for denoising
    bounds = np.linspace(-1, 1, data_optcom.shape[-1])
    legendre_arr = np.column_stack([lpmv(0, vv, bounds) for vv in range(dtrank)])

    # compute mean, std, mask local to this function
    # inefficient, but makes this function a bit more modular
    temporal_mean = data_optcom.mean(axis=-1)  # temporal mean
    temporal_mean_mask = temporal_mean != 0

    # Mean-center optimally combined data over time
    data_optcom_masked = (
        data_optcom[temporal_mean_mask] - temporal_mean[temporal_mean_mask][:, np.newaxis]
    )
    # Detrend the data using the Legendre basis functions
    betas = np.linalg.lstsq(legendre_arr, data_optcom_masked.T, rcond=None)[0]
    optcom_detr = data_optcom_masked - np.dot(betas.T, legendre_arr.T)[0]
    # The spatial global signal is the minimum of the detrended data
    gs_spatial = (optcom_detr).min(axis=1)
    gs_spatial -= gs_spatial.mean()
    io_generator.save_file(utils.unmask(gs_spatial, temporal_mean_mask), "gs img")

    # find time course of the spatial global signal
    # make basis with the Legendre basis
    gs_ts = np.linalg.lstsq(np.atleast_2d(gs_spatial).T, data_optcom_masked, rcond=None)[0]
    gs_ts = stats.zscore(gs_ts, axis=None)

    glsig_df = pd.DataFrame(data=gs_ts.T, columns=["global_signal"])
    io_generator.save_file(glsig_df, "global signal time series tsv")
    glbase = np.hstack([legendre_arr, gs_ts.T])

    # Project global signal out of optimally combined data
    betas = np.linalg.lstsq(np.atleast_2d(glbase), data_optcom_masked.T, rcond=None)[0]
    gs_fitted = np.dot(np.atleast_2d(betas[dtrank]).T, np.atleast_2d(glbase.T[dtrank]))
    data_optcom_nogs = (
        data_optcom_masked - gs_fitted + temporal_mean[temporal_mean_mask][:, np.newaxis]
    )
    data_optcom_nogs = utils.unmask(data_optcom_nogs, temporal_mean_mask)
    io_generator.save_file(data_optcom_nogs, "removed gs combined img")

    # Project glbase out of each echo
    data_cat_nogs = data_cat.copy()  # don't overwrite data_cat
    for echo in range(n_echos):
        data_echo_masked = data_cat_nogs[:, echo, :][temporal_mean_mask]
        betas = np.linalg.lstsq(np.atleast_2d(glbase), data_echo_masked.T, rcond=None)[0]
        echo_nogs = data_echo_masked - np.dot(
            np.atleast_2d(betas[dtrank]).T,
            np.atleast_2d(glbase.T[dtrank]),
        )
        data_cat_nogs[:, echo, :] = utils.unmask(echo_nogs, temporal_mean_mask)

    return data_cat_nogs, data_optcom_nogs


def minimum_image_regression(
    *,
    data_optcom: np.ndarray,
    mixing: np.ndarray,
    mask: np.ndarray,
    comptable: pd.DataFrame,
    classification_tags: list,
    io_generator: io.OutputGenerator,
):
    """Perform minimum image regression (MIR) to remove T1-like effects from BOLD-like components.

    While this method has not yet been described in detail in any publications,
    we recommend that users cite :footcite:t:`kundu2013integrated`.

    Parameters
    ----------
    data_optcom : (S x T) array_like
        Optimally combined time series data
    mixing : (T x C) array_like
        Mixing matrix for converting input data to component space, where ``C``
        is components and ``T`` is the same as in ``data_optcom``
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    classification_tags : :obj:`list` of :obj:`str`
        List of classification tags used in the decision tree.
        This is used to separate "accepted" and "ignored" components.
    io_generator : :obj:`tedana.io.OutputGenerator`
        The output generating object for this workflow

    Notes
    -----
    Minimum image regression operates by constructing a amplitude-normalized form of the multi-echo
    high Kappa (MEHK) time series from BOLD-like ICA components,
    and then taking the voxel-wise minimum over time.
    This "minimum map" serves as a voxel-wise estimate of the T1-like effect in the time series.
    From this minimum map, a T1-like global signal (i.e., a 1D time series) is estimated.
    The component time series in the mixing matrix are then corrected for the T1-like effect by
    regressing out the global signal time series from each.
    Finally, the multi-echo denoised (MEDN) and MEHK time series are reconstructed from the
    corrected mixing matrix and are written out to new files.

    This function writes out several files:

    ========================================    =================================================
    IOGenerator Label                           Content
    ========================================    =================================================
    "t1 like img"                               T1-like effect
    "mir denoised img"                          Denoised version of T1-corrected time series
    "ICA MIR mixing tsv"                        T1 global signal-corrected mixing matrix

    if io_generator.verbose==True
    "ICA accepted mir denoised img"             T1-corrected BOLD (high-Kappa) time series
    "ICA accepted mir component weights img"    T1 global signal-corrected components
    ========================================    =================================================

    References
    ----------
    .. footbibliography::
    """
    LGR.info("Performing minimum image regression to remove spatially-diffuse noise")
    RepLGR.info(
        "Minimum image regression was then applied to the data in order to remove spatially "
        "diffuse noise \\citep{kundu2013integrated}."
    )

    all_comps = comptable.index.values
    # Get accepted and ignored components
    # Tedana has removed the "ignored" classification,
    # so we must separate "accepted" components based on the classification tag(s).
    ignore_tags = ["low variance", "accept borderline"]
    if not any(tag in classification_tags for tag in ignore_tags):
        LGR.warning(
            "Decision tree does not contain classification tags indicating low variance "
            f"components ({', '.join(ignore_tags)})."
        )
        ign = np.array([], dtype=int)
    else:
        pattern = "|".join(ignore_tags)  # Create a pattern that matches any of the ignore tags

        # Select rows where the 'classification_tags' column contains any of the ignore tags
        ign = comptable[
            comptable.classification_tags.str.contains(pattern, na=False, regex=True)
        ].index.values

    acc = comptable[comptable.classification == "accepted"].index.values
    # Ignored components are classified as "accepted", so we need to remove them from the list
    acc = sorted(np.setdiff1d(acc, ign))
    not_ign = sorted(np.setdiff1d(all_comps, ign))

    data_optcom_masked = data_optcom[mask, :]
    optcom_mean = data_optcom_masked.mean(axis=-1)[:, np.newaxis]
    optcom_std = data_optcom_masked.std(axis=-1)[:, np.newaxis]

    # Compute temporal regression
    data_optcom_z = stats.zscore(data_optcom_masked, axis=-1)
    # component parameter estimates
    comp_pes = np.linalg.lstsq(mixing, data_optcom_z.T, rcond=None)[0].T
    # Get residuals (ignored/low-variance components and unmodeled noise)
    resid = data_optcom_z - np.dot(comp_pes[:, not_ign], mixing[:, not_ign].T)

    # Build time series of just BOLD-like components (i.e., MEHK) and save T1-like effect
    mehk_ts = np.dot(comp_pes[:, acc], mixing[:, acc].T)
    t1_map = mehk_ts.min(axis=-1)  # map of T1-like effect
    t1_map -= t1_map.mean()
    io_generator.save_file(utils.unmask(t1_map, mask), "t1 like img")
    t1_map = t1_map[:, np.newaxis]

    # Find the global signal based on the T1-like effect
    gs_ts = np.linalg.lstsq(t1_map, data_optcom_z, rcond=None)[0]

    # Remove T1-like global signal from MEHK time series
    mehk_no_t1_gs = mehk_ts - np.dot(
        np.linalg.lstsq(gs_ts.T, mehk_ts.T, rcond=None)[0].T,
        gs_ts,
    )

    # Make denoised version of T1-corrected time series
    medn_ts = optcom_mean + ((mehk_no_t1_gs + resid) * optcom_std)
    io_generator.save_file(utils.unmask(medn_ts, mask), "mir denoised img")

    # Orthogonalize mixing matrix w.r.t. T1-GS
    mixing_not1gs = mixing.T - np.dot(np.linalg.lstsq(gs_ts.T, mixing, rcond=None)[0].T, gs_ts)
    mixing_not1gs_z = stats.zscore(mixing_not1gs, axis=-1)
    mixing_not1gs_z = np.vstack((np.atleast_2d(np.ones(max(gs_ts.shape))), gs_ts, mixing_not1gs_z))

    # Write T1-corrected components and mixing matrix
    mixing_df = pd.DataFrame(data=mixing_not1gs.T, columns=comptable["Component"].values)
    io_generator.save_file(mixing_df, "ICA MIR mixing tsv")

    if io_generator.verbose:
        hik_ts = mehk_no_t1_gs * optcom_std  # rescale
        io_generator.save_file(utils.unmask(hik_ts, mask), "ICA accepted mir denoised img")

        comp_pes_norm = np.linalg.lstsq(mixing_not1gs_z.T, data_optcom_z.T, rcond=None)[0].T
        io_generator.save_file(
            utils.unmask(comp_pes_norm[:, 2:], mask),
            "ICA accepted mir component weights img",
        )
