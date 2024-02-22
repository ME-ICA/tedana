"""Global signal control methods."""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import lpmv

from tedana import utils

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def gscontrol_raw(catd, optcom, n_echos, io_generator, dtrank=4):
    """Remove global signal from individual echo ``catd`` and ``optcom`` time series.

    This function uses the spatial global signal estimation approach to
    to removal global signal out of individual echo time series datasets. The
    spatial global signal is estimated from the optimally combined data after
    detrending with a Legendre polynomial basis of `order = 0` and
    `degree = dtrank`.

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input functional data
    optcom : (S x T) array_like
        Optimally combined functional data (i.e., the output of `make_optcom`)
    n_echos : :obj:`int`
        Number of echos in data. Should be the same as `E` dimension of `catd`
    io_generator : :obj:`tedana.io.OutputGenerator`
        The output generator for this workflow
    dtrank : :obj:`int`, optional
        Specifies degree of Legendre polynomial basis function for estimating
        spatial global signal. Default: 4

    Returns
    -------
    dm_catd : (S x E x T) array_like
        Input `catd` with global signal removed from time series
    dm_optcom : (S x T) array_like
        Input `optcom` with global signal removed from time series
    """
    LGR.info("Applying amplitude-based T1 equilibration correction")
    RepLGR.info(
        "Global signal regression was applied to the multi-echo "
        "and optimally combined datasets."
    )
    if catd.shape[0] != optcom.shape[0]:
        raise ValueError(
            f"First dimensions of catd ({catd.shape[0]}) and optcom ({optcom.shape[0]}) do not "
            "match"
        )
    elif catd.shape[1] != n_echos:
        raise ValueError(
            f"Second dimension of catd ({catd.shape[1]}) does not match n_echos ({n_echos})"
        )
    elif catd.shape[2] != optcom.shape[1]:
        raise ValueError(
            f"Third dimension of catd ({catd.shape[2]}) does not match second dimension of optcom "
            f"({optcom.shape[1]})"
        )

    # Legendre polynomial basis for denoising
    bounds = np.linspace(-1, 1, optcom.shape[-1])
    legendre_arr = np.column_stack([lpmv(0, vv, bounds) for vv in range(dtrank)])

    # compute mean, std, mask local to this function
    # inefficient, but makes this function a bit more modular
    temporal_mean = optcom.mean(axis=-1)  # temporal mean
    temporal_mean_mask = temporal_mean != 0

    # find spatial global signal
    dat = optcom[temporal_mean_mask] - temporal_mean[temporal_mean_mask][:, np.newaxis]
    sol = np.linalg.lstsq(legendre_arr, dat.T, rcond=None)[0]  # Legendre basis for detrending
    detr = dat - np.dot(sol.T, legendre_arr.T)[0]
    sphis = (detr).min(axis=1)
    sphis -= sphis.mean()
    io_generator.save_file(utils.unmask(sphis, temporal_mean_mask), "gs img")

    # find time course ofc the spatial global signal
    # make basis with the Legendre basis
    glsig = np.linalg.lstsq(np.atleast_2d(sphis).T, dat, rcond=None)[0]
    glsig = stats.zscore(glsig, axis=None)

    glsig_df = pd.DataFrame(data=glsig.T, columns=["global_signal"])
    io_generator.save_file(glsig_df, "global signal time series tsv")
    glbase = np.hstack([legendre_arr, glsig.T])

    # Project global signal out of optimally combined data
    sol = np.linalg.lstsq(np.atleast_2d(glbase), dat.T, rcond=None)[0]
    tsoc_nogs = (
        dat
        - np.dot(np.atleast_2d(sol[dtrank]).T, np.atleast_2d(glbase.T[dtrank]))
        + temporal_mean[temporal_mean_mask][:, np.newaxis]
    )

    io_generator.save_file(optcom, "has gs combined img")
    dm_optcom = utils.unmask(tsoc_nogs, temporal_mean_mask)
    io_generator.save_file(dm_optcom, "removed gs combined img")

    # Project glbase out of each echo
    dm_catd = catd.copy()  # don't overwrite catd
    for echo in range(n_echos):
        dat = dm_catd[:, echo, :][temporal_mean_mask]
        sol = np.linalg.lstsq(np.atleast_2d(glbase), dat.T, rcond=None)[0]
        e_nogs = dat - np.dot(np.atleast_2d(sol[dtrank]).T, np.atleast_2d(glbase.T[dtrank]))
        dm_catd[:, echo, :] = utils.unmask(e_nogs, temporal_mean_mask)

    return dm_catd, dm_optcom


def minimum_image_regression(optcom_ts, mmix, mask, comptable, io_generator):
    """Perform minimum image regression (MIR) to remove T1-like effects from BOLD-like components.

    While this method has not yet been described in detail in any publications,
    we recommend that users cite :footcite:t:`kundu2013integrated`.

    Parameters
    ----------
    optcom_ts : (S x T) array_like
        Optimally combined time series data
    mmix : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `optcom_ts`
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    io_generator : :obj:`tedana.io.OutputGenerator`
        The output generating object for this workflow

    Notes
    -----
    Minimum image regression operates by constructing a amplitude-normalized
    form of the multi-echo high Kappa (MEHK) time series from BOLD-like ICA
    components, and then taking voxel-wise minimum over time.
    This "minimum map" serves as a voxel-wise estimate of the T1-like effect
    in the time series.
    From this minimum map, a T1-like global signal (i.e., a 1D time series)
    is estimated.
    The component time series in the mixing matrix are then corrected for the
    T1-like effect by regressing out the global signal time series from each.
    Finally, the multi-echo denoising (MEDN) and MEHK time series are
    reconstructed from the corrected mixing matrix and are written out to new
    files.

    This function writes out several files:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    sphis_hik.nii             T1-like effect
    hik_ts_OC_MIR.nii         T1-corrected BOLD (high-Kappa) time series
    dn_ts_OC_MIR.nii          Denoised version of T1-corrected time series
    betas_hik_OC_MIR.nii      T1 global signal-corrected components
    meica_mix_MIR.1D          T1 global signal-corrected mixing matrix
    ======================    =================================================

    References
    ----------
    .. footbibliography::
    """
    LGR.info("Performing minimum image regression to remove spatially-diffuse noise")
    RepLGR.info(
        "Minimum image regression was then applied to the "
        "data in order to remove spatially diffuse noise \\citep{kundu2013integrated}."
    )

    all_comps = comptable.index.values
    acc = comptable[comptable.classification == "accepted"].index.values
    ign = comptable[comptable.classification == "ignored"].index.values
    not_ign = sorted(np.setdiff1d(all_comps, ign))

    optcom_masked = optcom_ts[mask, :]
    optcom_mean = optcom_masked.mean(axis=-1)[:, np.newaxis]
    optcom_std = optcom_masked.std(axis=-1)[:, np.newaxis]

    # Compute temporal regression
    optcom_z = stats.zscore(optcom_masked, axis=-1)
    comp_pes = np.linalg.lstsq(mmix, optcom_z.T, rcond=None)[0].T  # component parameter estimates
    resid = optcom_z - np.dot(comp_pes[:, not_ign], mmix[:, not_ign].T)

    # Build time series of just BOLD-like components (i.e., MEHK) and save T1-like effect
    mehk_ts = np.dot(comp_pes[:, acc], mmix[:, acc].T)
    t1_map = mehk_ts.min(axis=-1)  # map of T1-like effect
    t1_map -= t1_map.mean()
    io_generator.save_file(utils.unmask(t1_map, mask), "t1 like img")
    t1_map = t1_map[:, np.newaxis]

    # Find the global signal based on the T1-like effect
    glob_sig = np.linalg.lstsq(t1_map, optcom_z, rcond=None)[0]

    # Remove T1-like global signal from MEHK time series
    mehk_no_t1_gs = mehk_ts - np.dot(
        np.linalg.lstsq(glob_sig.T, mehk_ts.T, rcond=None)[0].T, glob_sig
    )
    hik_ts = mehk_no_t1_gs * optcom_std  # rescale
    io_generator.save_file(utils.unmask(hik_ts, mask), "ICA accepted mir denoised img")

    # Make denoised version of T1-corrected time series
    medn_ts = optcom_mean + ((mehk_no_t1_gs + resid) * optcom_std)
    io_generator.save_file(utils.unmask(medn_ts, mask), "mir denoised img")

    # Orthogonalize mixing matrix w.r.t. T1-GS
    mmix_no_t1_gs = mmix.T - np.dot(np.linalg.lstsq(glob_sig.T, mmix, rcond=None)[0].T, glob_sig)
    mmix_no_t1_gs_z = stats.zscore(mmix_no_t1_gs, axis=-1)
    mmix_no_t1_gs_z = np.vstack(
        (np.atleast_2d(np.ones(max(glob_sig.shape))), glob_sig, mmix_no_t1_gs_z)
    )

    # Write T1-corrected components and mixing matrix
    comp_pes_norm = np.linalg.lstsq(mmix_no_t1_gs_z.T, optcom_z.T, rcond=None)[0].T
    io_generator.save_file(
        utils.unmask(comp_pes_norm[:, 2:], mask),
        "ICA accepted mir component weights img",
    )
    mixing_df = pd.DataFrame(data=mmix_no_t1_gs.T, columns=comptable["Component"].values)
    io_generator.save_file(mixing_df, "ICA MIR mixing tsv")
