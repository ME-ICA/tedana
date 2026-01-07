"""Functions to estimate S0 and T2* from multi-echo data."""

import logging
import os
from typing import List, Literal, Tuple

import numpy as np
import numpy.matlib
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy import stats
from tqdm.auto import tqdm

from tedana import utils

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def _apply_t2s_floor(t2s, echo_times):
    """Apply a floor to T2* values to prevent zero division errors during optimal combination.

    Parameters
    ----------
    t2s : (S,) array_like
        T2* estimates.
    echo_times : (E,) array_like
        Echo times in milliseconds.

    Returns
    -------
    t2s_corrected : (S,) array_like
        T2* estimates with very small, positive values replaced with a floor value.
    """
    t2s_corrected = t2s.copy()
    echo_times = np.asarray(echo_times)
    if echo_times.ndim == 1:
        echo_times = echo_times[:, None]

    eps = np.finfo(dtype=t2s.dtype).eps  # smallest value for datatype
    nonzerovox = t2s != 0
    # Exclude values where t2s is 0 when dividing by t2s.
    # These voxels are also excluded from bad_voxel_idx
    temp_arr = np.zeros((len(echo_times), len(t2s)))
    temp_arr[:, nonzerovox] = np.exp(-echo_times / t2s[nonzerovox])  # (E x V) array
    bad_voxel_idx = np.any(temp_arr == 0, axis=0) & (t2s != 0)
    n_bad_voxels = np.sum(bad_voxel_idx)
    if n_bad_voxels > 0:
        n_voxels = temp_arr.size
        floor_percent = 100 * n_bad_voxels / n_voxels
        LGR.debug(
            f"T2* values for {n_bad_voxels}/{n_voxels} voxels ({floor_percent:.2f}%) have been "
            "identified as close to zero and have been adjusted"
        )
    t2s_corrected[bad_voxel_idx] = np.min(-echo_times) / np.log(eps)
    return t2s_corrected


def monoexponential(tes, s0, t2star):
    """Specify a monoexponential model for use with scipy curve fitting.

    Parameters
    ----------
    tes : (E,) :obj:`list`
        Echo times
    s0 : :obj:`float`
        Initial signal parameter
    t2star : :obj:`float`
        T2* parameter

    Returns
    -------
    : obj:`float`
        Predicted signal
    """
    return s0 * np.exp(-tes / t2star)


def _fit_single_voxel(voxel, echo_times_1d, data_column, s0_init, t2s_init, bounds):
    """Fit monoexponential model for a single voxel.

    Parameters
    ----------
    voxel : int
        Voxel index
    echo_times_1d : (E*T,) array_like
        Echo times repeated for each timepoint
    data_column : (E*T,) array_like
        Data for this voxel across all echoes and timepoints
    s0_init : float
        Initial S0 estimate
    t2s_init : float
        Initial T2* estimate
    bounds : tuple
        Bounds for curve_fit

    Returns
    -------
    result : tuple or None
        If successful: (voxel, s0, t2s, False)
        If failed: (voxel, None, None, True)
    """
    try:
        popt, cov = scipy.optimize.curve_fit(
            monoexponential,
            echo_times_1d,
            data_column,
            p0=(s0_init, t2s_init),
            bounds=bounds,
        )
        return (voxel, popt[0], popt[1], False, cov[0, 0], cov[1, 1], cov[0, 1])
    except (RuntimeError, ValueError):
        return (voxel, None, None, True, None, None, None)


def fit_monoexponential(data_cat, echo_times, adaptive_mask, report=True, n_threads=1):
    """Fit monoexponential decay model with nonlinear curve-fitting.

    Parameters
    ----------
    data_cat : (S x E x T) :obj:`numpy.ndarray`
        Multi-echo data.
    echo_times : (E,) array_like
        Echo times in milliseconds.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.
    n_threads : int, optional
        Number of threads to use. Default is 1. If None or <= 0, uses the number
        of available CPU cores.

    Returns
    -------
    t2s, s0 : (S,) :obj:`numpy.ndarray`
        T2* and S0 estimate maps.
        These maps include T2*/S0 estimates for all voxels with adaptive mask >= 1.
        For voxels with adaptive mask == 1, the T2*/S0 estimates are from the first two echoes.
        These voxels should be replaced with zeros in the full T2*/S0 maps.
    failures : (S,) :obj:`numpy.ndarray`
        Boolean array indicating samples that failed to fit the model.
    t2s_var : (S,) :obj:`numpy.ndarray`
        Variance of the T2* estimates.
    s0_var : (S,) :obj:`numpy.ndarray`
        Variance of the S0 estimates.
    t2s_s0_covar : (S,) :obj:`numpy.ndarray`
        Covariance of the T2* and S0 estimates.

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
        parameter.

    Notes
    -----
    This method is slower, but more accurate, than the log-linear approach.
    """
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    if report:
        RepLGR.info(
            "A monoexponential model was fit to the data at each voxel "
            "using nonlinear model fitting in order to estimate T2* and S0 "
            "maps, using T2*/S0 estimates from a log-linear fit as "
            "initial values. For each voxel, the value from the adaptive "
            "mask was used to determine which echoes would be used to "
            "estimate T2* and S0. In cases of model fit failure, T2*/S0 "
            "estimates from the log-linear fit were retained instead."
        )
    n_samp, _, n_vols = data_cat.shape

    # Currently unused
    # fit_data = np.mean(data_cat, axis=2)
    # fit_sigma = np.std(data_cat, axis=2)

    t2s_init, s0_init = fit_loglinear(
        data_cat=data_cat,
        echo_times=echo_times,
        adaptive_mask=adaptive_mask,
        report=False,
    )

    echos_to_run = np.unique(adaptive_mask)
    # When there is one good echo, use two
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    t2s_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    failures_asc_maps = np.zeros([n_samp, len(echos_to_run)], dtype=bool)
    t2s_var_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_var_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    t2s_s0_covar_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    echo_masks = np.zeros([n_samp, len(echos_to_run)], dtype=bool)

    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
            # Use the first two echoes for cases where there are
            # either one or two good echoes
            voxel_idx = np.where(adaptive_mask <= echo_num)[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        # Create echo masks to assign values to limited vs full maps later
        echo_mask = np.squeeze(echo_masks[..., i_echo])
        echo_mask[adaptive_mask == echo_num] = True
        echo_masks[..., i_echo] = echo_mask

        data_2d = data_cat[:, :echo_num, :].reshape(len(data_cat), -1).T
        echo_times_1d = np.repeat(echo_times[:echo_num], n_vols)

        # perform a monoexponential fit of echo times against MR signal
        # using loglin estimates as initial starting points for fit
        # parallelize the curve_fit calls across voxels
        results = Parallel(n_jobs=n_threads)(
            delayed(_fit_single_voxel)(
                voxel=voxel,
                echo_times_1d=echo_times_1d,
                data_column=data_2d[:, voxel],
                s0_init=s0_init[voxel],
                t2s_init=t2s_init[voxel],
                bounds=((np.min(data_2d[:, voxel]), 0), (np.inf, np.inf)),
            )
            for voxel in tqdm(voxel_idx, desc=f"{echo_num}-echo monoexponential")
        )

        # Update results and count failures
        fail_count = 0
        for (
            voxel,
            s0_voxel,
            t2s_voxel,
            failure,
            t2s_var_voxel,
            s0_var_voxel,
            t2s_s0_covar_voxel,
        ) in results:
            if failure:
                failures_asc_maps[voxel, i_echo] = True
                fail_count += 1
            else:
                s0_init[voxel] = s0_voxel
                t2s_init[voxel] = t2s_voxel
                t2s_var_asc_maps[voxel, i_echo] = t2s_var_voxel
                s0_var_asc_maps[voxel, i_echo] = s0_var_voxel
                t2s_s0_covar_asc_maps[voxel, i_echo] = t2s_s0_covar_voxel

        if fail_count:
            fail_percent = 100 * fail_count / len(voxel_idx)
            LGR.debug(
                f"With {echo_num} echoes, monoexponential fit failed on "
                f"{fail_count}/{len(voxel_idx)} ({fail_percent:.2f}%) voxel(s), "
                "used log linear estimate instead"
            )

        t2s_asc_maps[:, i_echo] = t2s_init
        s0_asc_maps[:, i_echo] = s0_init

    # create full T2* and S0 maps
    t2s = utils.unmask(t2s_asc_maps[echo_masks], adaptive_mask > 1)
    s0 = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask > 1)
    failures = utils.unmask(failures_asc_maps[echo_masks], adaptive_mask > 1)
    t2s_var = utils.unmask(t2s_var_asc_maps[echo_masks], adaptive_mask > 1)
    s0_var = utils.unmask(s0_var_asc_maps[echo_masks], adaptive_mask > 1)
    t2s_s0_covar = utils.unmask(t2s_s0_covar_asc_maps[echo_masks], adaptive_mask > 1)

    # create full T2* maps with S0 estimation errors
    t2s[adaptive_mask == 1] = t2s_asc_maps[adaptive_mask == 1, 0]
    s0[adaptive_mask == 1] = s0_asc_maps[adaptive_mask == 1, 0]

    return t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar


def fit_loglinear(data_cat, echo_times, adaptive_mask, report=True):
    """Fit monoexponential decay model with log-linear regression.

    The monoexponential decay function is fitted to all values for a given
    voxel across TRs, per TE, to estimate voxel-wise :math:`S_0` and :math:`T_2^*`.
    At a given voxel, only those echoes with "good signal", as indicated by the
    value of the voxel in the adaptive mask, are used.
    Therefore, for a voxel with an adaptive mask value of five, the first five
    echoes would be used to estimate T2* and S0.

    Parameters
    ----------
    data_cat : (M x E x T) :obj:`numpy.ndarray`
        Multi-echo data. M is samples in mask, E is echoes, and T is timepoints.
    echo_times : (E,) array_like
        Echo times in milliseconds.
    adaptive_mask : (M,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    report : :obj:`bool`, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    t2s, s0 : (M,) :obj:`numpy.ndarray`
        "Full" T2* and S0 maps without floors or ceilings applied.
        This includes T2* and S0 estimates for all voxels with adaptive mask >= 1.
        Voxels with adaptive mask == 1 have T2* and S0 estimates from the first two echoes.

    Notes
    -----
    The approach used in this function involves transforming the raw signal values
    (:math:`log(|data| + 1)`) and then fitting a line to the transformed data using
    ordinary least squares.
    This results in two parameter estimates: one for the slope  and one for the intercept.
    The slope estimate is inverted (i.e., 1 / slope) to get  :math:`T_2^*`,
    while the intercept estimate is exponentiated (i.e., e^intercept) to get :math:`S_0`.

    This method is faster, but less accurate, than the nonlinear approach.
    """
    if report:
        RepLGR.info(
            "A monoexponential model was fit to the data at each voxel "
            "using log-linear regression in order to estimate T2* and S0 "
            "maps. For each voxel, the value from the adaptive mask was "
            "used to determine which echoes would be used to estimate T2* "
            "and S0."
        )
    n_samp, _, n_vols = data_cat.shape

    echos_to_run = np.unique(adaptive_mask)
    # When there is one good echo, use two
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    t2s_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    echo_masks = np.zeros([n_samp, len(echos_to_run)], dtype=bool)

    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
            # Use the first two echoes for cases where there are
            # either one or two good echoes
            voxel_idx = np.where(np.logical_and(adaptive_mask > 0, adaptive_mask <= echo_num))[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        # Create echo masks to assign values to limited vs full maps later
        echo_mask = np.squeeze(echo_masks[..., i_echo])
        echo_mask[adaptive_mask == echo_num] = True
        echo_masks[..., i_echo] = echo_mask

        # perform log linear fit of echo times against MR signal
        # make DV matrix: samples x (time series * echos)
        data_2d = data_cat[voxel_idx, :echo_num, :].reshape(len(voxel_idx), -1).T
        log_data = np.log(np.abs(data_2d) + 1)

        # make IV matrix: intercept/TEs x (time series * echos)
        x = np.column_stack([np.ones(echo_num), [-te for te in echo_times[:echo_num]]])
        iv_arr = np.repeat(x, n_vols, axis=0)

        # Log-linear fit
        betas = np.linalg.lstsq(iv_arr, log_data, rcond=None)[0]
        t2s = 1.0 / betas[1, :].T
        s0 = np.exp(betas[0, :]).T

        t2s_asc_maps[voxel_idx, i_echo] = t2s
        s0_asc_maps[voxel_idx, i_echo] = s0

    # create full T2* and S0 maps with S0 estimation errors
    t2s = utils.unmask(t2s_asc_maps[echo_masks], adaptive_mask > 1)
    s0 = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask > 1)
    t2s[adaptive_mask == 1] = t2s_asc_maps[adaptive_mask == 1, 0]
    s0[adaptive_mask == 1] = s0_asc_maps[adaptive_mask == 1, 0]

    return t2s, s0


def fit_decay(data, tes, adaptive_mask, fittype, report=True, n_threads=1):
    """Fit voxel-wise monoexponential decay models to ``data``.

    Parameters
    ----------
    data : (M x E [x T]) array_like
        Multi-echo data array, where `M` is samples in mask, `E` is echos, and `T` is time.
    tes : (E,) :obj:`list`
        Echo times in milliseconds.
    adaptive_mask : (M,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    fittype : {loglin, curvefit}
        The type of model fit to use
    report : bool, optional
        Whether to log a description of this step or not. Default is True.
    n_threads : int, optional
        Number of threads to use. Default is 1. If None or <= 0, uses the number
        of available CPU cores.

    Returns
    -------
    t2s : (M,) :obj:`numpy.ndarray`
        "Full" T2* map without floors or ceilings applied.
        This includes T2* estimates for all voxels with adaptive mask >= 1.
        Voxels with adaptive mask == 1 have T2* estimates from the first two echoes.
    s0 : (M,) :obj:`numpy.ndarray`
        "Full" S0 map without floors or ceilings applied.
        This includes S0 estimates for all voxels with adaptive mask >= 1.
        Voxels with adaptive mask == 1 have S0 estimates from the first two echoes.
    failures : (M,) :obj:`numpy.ndarray` or None
        Boolean array indicating samples that failed to fit the model.
        None if fittype is not "curvefit".
    t2s_var : (M,) :obj:`numpy.ndarray` or None
        Variance of the T2* estimates.
        None if fittype is not "curvefit".
    s0_var : (M,) :obj:`numpy.ndarray` or None
        Variance of the S0 estimates.
        None if fittype is not "curvefit".
    t2s_s0_covar : (M,) :obj:`numpy.ndarray` or None
        Covariance of the T2* and S0 estimates.
        None if fittype is not "curvefit".

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                              parameter.
    """
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    if data.shape[1] != len(tes):
        raise ValueError(
            f"Second dimension of data ({data.shape[1]}) does not match number "
            f"of echoes provided (tes; {len(tes)})"
        )
    elif not (data.shape[0] == adaptive_mask.shape[0]):
        raise ValueError(
            f"First dimensions (number of samples) of data ({data.shape[0]}), "
            f"and adaptive_mask ({adaptive_mask.shape[0]}) do not match"
        )

    if data.ndim == 2:
        data = data[:, :, None]

    failures, t2s_var, s0_var, t2s_s0_covar = None, None, None, None
    if fittype == "loglin":
        t2s, s0 = fit_loglinear(
            data_cat=data,
            echo_times=tes,
            adaptive_mask=adaptive_mask,
            report=report,
        )
    elif fittype == "curvefit":
        (
            t2s,
            s0,
            failures,
            t2s_var,
            s0_var,
            t2s_s0_covar,
        ) = fit_monoexponential(
            data_cat=data,
            echo_times=tes,
            adaptive_mask=adaptive_mask,
            report=report,
            n_threads=n_threads,
        )
    else:
        raise ValueError(f"Unknown fittype option: {fittype}")

    return t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar


def fit_decay_ts(data, tes, adaptive_mask, fittype, n_threads=1):
    """Fit voxel- and timepoint-wise monoexponential decay models to ``data``.

    Parameters
    ----------
    data : (M x E x T) array_like
        Multi-echo data array, where `M` is samples in mask, `E` is echos, and `T` is time.
    tes : (E,) :obj:`list`
        Echo times
    adaptive_mask : (M,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    fittype : :obj: `str`
        The type of model fit to use
    n_threads : int, optional
        Number of threads to use. Default is 1. If None or <= 0, uses the number
        of available CPU cores.

    Returns
    -------
    t2s : (M x T) :obj:`numpy.ndarray`
        Limited T2* map. The limited map only keeps the T2* values for data
        where there are at least two echos with good signal.
    s0 : (M x T) :obj:`numpy.ndarray`
        Limited S0 map.  The limited map only keeps the S0 values for data
        where there are at least two echos with good signal.
    failures : (M x T) :obj:`numpy.ndarray` or None
        Boolean array indicating samples that failed to fit the model.
        None if fittype is not "curvefit".
    t2s_var : (M x T) :obj:`numpy.ndarray` or None
        Variance of the T2* estimates.
        None if fittype is not "curvefit".
    s0_var : (M x T) :obj:`numpy.ndarray` or None
        Variance of the S0 estimates.
        None if fittype is not "curvefit".
    t2s_s0_covar : (M x T) :obj:`numpy.ndarray` or None
        Covariance of the T2* and S0 estimates.
        None if fittype is not "curvefit".

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
        parameter.
    """
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    n_samples, _, n_vols = data.shape
    tes = np.array(tes)

    t2s = np.zeros([n_samples, n_vols])
    s0 = np.zeros([n_samples, n_vols])
    failures = None
    t2s_var = None
    s0_var = None
    t2s_s0_covar = None
    if fittype == "curvefit":
        failures = np.zeros([n_samples, n_vols], dtype=bool)
        t2s_var = np.zeros([n_samples, n_vols])
        s0_var = np.zeros([n_samples, n_vols])
        t2s_s0_covar = np.zeros([n_samples, n_vols])

    report = True
    for vol in range(n_vols):
        (
            t2s_vol,
            s0_vol,
            failures_vol,
            t2s_var_vol,
            s0_var_vol,
            t2s_s0_covar_vol
        ) = fit_decay(
            data=data[:, :, vol][:, :, None],
            tes=tes,
            adaptive_mask=adaptive_mask,
            fittype=fittype,
            report=report,
            n_threads=n_threads,
        )
        t2s[:, vol] = t2s_vol
        s0[:, vol] = s0_vol
        if fittype == "curvefit":
            failures[:, vol] = failures_vol
            t2s_var[:, vol] = t2s_var_vol
            s0_var[:, vol] = s0_var_vol
            t2s_s0_covar[:, vol] = t2s_s0_covar_vol

        report = False

    return (
        t2s,
        s0,
        failures,
        t2s_var,
        s0_var,
        t2s_s0_covar,
    )


def modify_t2s_s0_maps(t2s, s0, adaptive_mask, tes):
    """Modify T2* and S0 maps to include estimates for voxels with adaptive mask == 1.

    Parameters
    ----------
    t2s : (S,) :obj:`numpy.ndarray`
        "Full" T2* map.
        This includes T2* estimates for all voxels with adaptive mask >= 1.
    s0 : (S,) :obj:`numpy.ndarray`
        "Full" S0 map.
        This includes S0 estimates for all voxels with adaptive mask >= 1.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Adaptive mask array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    tes : (E,) :obj:`list`
        Echo times in milliseconds.

    Returns
    -------
    t2s : (S,) :obj:`numpy.ndarray`
        "Full" T2* map with floors and ceilings applied.
        This includes T2* estimates for all voxels with adaptive mask >= 1.
    s0 : (S,) :obj:`numpy.ndarray`
        "Full" S0 map with floors and ceilings applied.
        This includes S0 estimates for all voxels with adaptive mask >= 1.
    t2s_limited : (S,) :obj:`numpy.ndarray`
        "Limited" T2* map.
        This includes T2* estimates for all voxels with adaptive mask > 1.
        Voxels with adaptive mask == 1 are set to 0.
    s0_limited : (S,) :obj:`numpy.ndarray`
        "Limited" S0 map.
        This includes S0 estimates for all voxels with adaptive mask > 1.
        Voxels with adaptive mask == 1 are set to 0.

    Notes
    -----
    This function replaces infinite values in the :math:`T_2^*` map with 500 and
    :math:`T_2^*` values less than or equal to zero with 1.
    Additionally, very small :math:`T_2^*` values above zero are replaced with a floor
    value to prevent zero-division errors later on in the workflow.
    It also replaces NaN values in the :math:`S_0` map with 0.
    """
    # Apply floors and ceilings to the T2* and S0 maps
    t2s[np.isinf(t2s)] = 500.0  # why 500?
    t2s[t2s <= 0] = 1.0  # let's get rid of negative values!
    t2s = _apply_t2s_floor(t2s, tes)
    s0[np.isnan(s0)] = 0.0  # why 0?

    t2s_limited = t2s.copy()
    s0_limited = s0.copy()
    t2s_limited[adaptive_mask == 1] = 0
    s0_limited[adaptive_mask == 1] = 0

    # set a hard cap for the T2* map
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s_limited.flatten(), 99.5, interpolation_method="lower")
    LGR.debug(f"Setting cap on T2* map at {cap_t2s * 10:.5f}")
    t2s_limited[t2s_limited > cap_t2s * 10] = cap_t2s

    return t2s, s0, t2s_limited, s0_limited


def rmse_of_fit_decay_ts(
    *,
    data: np.ndarray,
    tes: List[float],
    adaptive_mask: np.ndarray,
    t2s: np.ndarray,
    s0: np.ndarray,
    fitmode: Literal["all", "ts"],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate model fit of voxel- and timepoint-wise monoexponential decay models to ``data``.

    Parameters
    ----------
    data : (S x E x T) :obj:`numpy.ndarray`
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is time.
    tes : (E,) :obj:`list`
        Echo times.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal for that voxel.
        This mask may be thresholded; for example, with values less than 3 set to 0.
        For more information on thresholding, see :func:`~tedana.utils.make_adaptive_mask`.
    t2s : (S [x T]) :obj:`numpy.ndarray`
        Voxel-wise (and possibly volume-wise) T2* estimates from
        :func:`~tedana.decay.fit_decay_ts`.
    s0 : (S [x T]) :obj:`numpy.ndarray`
        Voxel-wise (and possibly volume-wise) S0 estimates from :func:`~tedana.decay.fit_decay_ts`.
    fitmode : {"fit", "all"}
        Whether the T2* and S0 estimates are volume-wise ("fit") or not ("all").

    Returns
    -------
    rmse_map : (S,) :obj:`numpy.ndarray`
        Mean root mean squared error of the model fit across all volumes at each voxel.
    rmse_df : :obj:`pandas.DataFrame`
        Each column is the root mean squared error of the model fit at each timepoint.
        Columns are mean, standard deviation, and percentiles across voxels. Column labels are
        "rmse_mean", "rmse_std", "rmse_min", "rmse_percentile02", "rmse_percentile25",
        "rmse_median", "rmse_percentile75", "rmse_percentile98", and "rmse_max"
    """
    n_samples, _, n_vols = data.shape
    tes = np.array(tes)

    rmse = np.full([n_samples, n_vols], np.nan, dtype=np.float32)
    # n_good_echoes interates from 2 through the number of echoes
    #   0 and 1 are excluded because there aren't T2* and S0 estimates
    #   for less than 2 good echoes. 2 echoes will have a bad estimate so consider
    #   how/if we want to distinguish those
    for n_good_echoes in range(2, len(tes) + 1):
        # a boolean mask for voxels with a specific num of good echoes
        use_vox = adaptive_mask == n_good_echoes
        data_echo = data[use_vox, :n_good_echoes, :]
        if fitmode == "all":
            s0_echo = numpy.matlib.repmat(s0[use_vox].T, n_vols, 1).T
            t2s_echo = numpy.matlib.repmat(t2s[use_vox], n_vols, 1).T
        elif fitmode == "ts":
            s0_echo = s0[use_vox, :]
            t2s_echo = t2s[use_vox, :]
        else:
            raise ValueError(f"Unknown fitmode option {fitmode}")

        predicted_data = np.full([use_vox.sum(), n_good_echoes, n_vols], np.nan, dtype=np.float32)
        # Need to loop by echo since monoexponential can take either single vals for s0 and t2star
        #   or a single TE value.
        # We could expand that func, but this is a functional solution
        for echo_num in range(n_good_echoes):
            predicted_data[:, echo_num, :] = monoexponential(
                tes=tes[echo_num],
                s0=s0_echo,
                t2star=t2s_echo,
            )
        rmse[use_vox, :] = np.sqrt(np.mean((data_echo - predicted_data) ** 2, axis=1))

    rmse_map = np.nanmean(rmse, axis=1)
    rmse_timeseries = np.nanmean(rmse, axis=0)
    rmse_sd_timeseries = np.nanstd(rmse, axis=0)
    rmse_percentiles_timeseries = np.nanpercentile(rmse, [0, 2, 25, 50, 75, 98, 100], axis=0)

    rmse_df = pd.DataFrame(
        columns=[
            "rmse_mean",
            "rmse_std",
            "rmse_min",
            "rmse_percentile02",
            "rmse_percentile25",
            "rmse_median",
            "rmse_percentile75",
            "rmse_percentile98",
            "rmse_max",
        ],
        data=np.column_stack(
            (
                rmse_timeseries,
                rmse_sd_timeseries,
                rmse_percentiles_timeseries.T,
            )
        ),
    )

    return rmse_map, rmse_df
