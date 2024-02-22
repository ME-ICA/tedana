"""Functions to estimate S0 and T2* from multi-echo data."""

import logging

import numpy as np
import scipy
from scipy import stats

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


def fit_monoexponential(data_cat, echo_times, adaptive_mask, report=True):
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

    Returns
    -------
    t2s_limited, s0_limited, t2s_full, s0_full : (S,) :obj:`numpy.ndarray`
        T2* and S0 estimate maps.

    See Also
    --------
    : func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
        parameter.

    Notes
    -----
    This method is slower, but more accurate, than the log-linear approach.
    """
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
    n_samp, n_echos, n_vols = data_cat.shape

    # Currently unused
    # fit_data = np.mean(data_cat, axis=2)
    # fit_sigma = np.std(data_cat, axis=2)

    t2s_limited, s0_limited, t2s_full, s0_full = fit_loglinear(
        data_cat, echo_times, adaptive_mask, report=False
    )

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
        fail_count = 0
        for voxel in voxel_idx:
            try:
                popt, cov = scipy.optimize.curve_fit(
                    monoexponential,
                    echo_times_1d,
                    data_2d[:, voxel],
                    p0=(s0_full[voxel], t2s_full[voxel]),
                    bounds=((np.min(data_2d[:, voxel]), 0), (np.inf, np.inf)),
                )
                s0_full[voxel] = popt[0]
                t2s_full[voxel] = popt[1]
            except (RuntimeError, ValueError):
                # If curve_fit fails to converge, fall back to loglinear estimate
                fail_count += 1

        if fail_count:
            fail_percent = 100 * fail_count / len(voxel_idx)
            LGR.debug(
                f"With {echo_num} echoes, monoexponential fit failed on "
                f"{fail_count}/{len(voxel_idx)} ({fail_percent:.2f}%) voxel(s), "
                "used log linear estimate instead"
            )

        t2s_asc_maps[:, i_echo] = t2s_full
        s0_asc_maps[:, i_echo] = s0_full

    # create limited T2* and S0 maps
    t2s_limited = utils.unmask(t2s_asc_maps[echo_masks], adaptive_mask > 1)
    s0_limited = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask > 1)

    # create full T2* maps with S0 estimation errors
    t2s_full, s0_full = t2s_limited.copy(), s0_limited.copy()
    t2s_full[adaptive_mask == 1] = t2s_asc_maps[adaptive_mask == 1, 0]
    s0_full[adaptive_mask == 1] = s0_asc_maps[adaptive_mask == 1, 0]

    return t2s_limited, s0_limited, t2s_full, s0_full


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
    data_cat : (S x E x T) :obj:`numpy.ndarray`
        Multi-echo data. S is samples, E is echoes, and T is timepoints.
    echo_times : (E,) array_like
        Echo times in milliseconds.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    report : :obj:`bool`, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    t2s_limited, s0_limited, t2s_full, s0_full : (S,) :obj:`numpy.ndarray`
        T2* and S0 estimate maps.

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
    n_samp, n_echos, n_vols = data_cat.shape

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

    # create limited T2* and S0 maps
    t2s_limited = utils.unmask(t2s_asc_maps[echo_masks], adaptive_mask > 1)
    s0_limited = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask > 1)

    # create full T2* maps with S0 estimation errors
    t2s_full, s0_full = t2s_limited.copy(), s0_limited.copy()
    t2s_full[adaptive_mask == 1] = t2s_asc_maps[adaptive_mask == 1, 0]
    s0_full[adaptive_mask == 1] = s0_asc_maps[adaptive_mask == 1, 0]

    return t2s_limited, s0_limited, t2s_full, s0_full


def fit_decay(data, tes, mask, adaptive_mask, fittype, report=True):
    """Fit voxel-wise monoexponential decay models to ``data``.

    Parameters
    ----------
    data : (S x E [x T]) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E,) :obj:`list`
        Echo times
    mask : (S,) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    adaptive_mask : (S,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    fittype : {loglin, curvefit}
        The type of model fit to use
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    t2s_limited : (S,) :obj:`numpy.ndarray`
        Limited T2* map. The limited map only keeps the T2* values for data
        where there are at least two echos with good signal.
    s0_limited : (S,) :obj:`numpy.ndarray`
        Limited S0 map.  The limited map only keeps the S0 values for data
        where there are at least two echos with good signal.
    t2s_full : (S,) :obj:`numpy.ndarray`
        Full T2* map. For voxels affected by dropout, with good signal from
        only one echo, the full map uses the T2* estimate from the first two
        echoes.
    s0_full : (S,) :obj:`numpy.ndarray`
        Full S0 map. For voxels affected by dropout, with good signal from
        only one echo, the full map uses the S0 estimate from the first two
        echoes.

    See Also
    --------
    : func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                                       parameter.

    Notes
    -----
    This function replaces infinite values in the :math:`T_2^*` map with 500 and
    :math:`T_2^*` values less than or equal to zero with 1.
    Additionally, very small :math:`T_2^*` values above zero are replaced with a floor
    value to prevent zero-division errors later on in the workflow.
    It also replaces NaN values in the :math:`S_0` map with 0.
    """
    if data.shape[1] != len(tes):
        raise ValueError(
            f"Second dimension of data ({data.shape[1]}) does not match number "
            f"of echoes provided (tes; {len(tes)})"
        )
    elif not (data.shape[0] == mask.shape[0] == adaptive_mask.shape[0]):
        raise ValueError(
            f"First dimensions (number of samples) of data ({data.shape[0]}), "
            f"mask ({mask.shape[0]}), and adaptive_mask ({adaptive_mask.shape[0]}) do not match"
        )

    if data.ndim == 2:
        data = data[:, :, None]

    # Mask the inputs
    data_masked = data[mask, :, :]
    adaptive_mask_masked = adaptive_mask[mask]

    if fittype == "loglin":
        t2s_limited, s0_limited, t2s_full, s0_full = fit_loglinear(
            data_masked, tes, adaptive_mask_masked, report=report
        )
    elif fittype == "curvefit":
        t2s_limited, s0_limited, t2s_full, s0_full = fit_monoexponential(
            data_masked, tes, adaptive_mask_masked, report=report
        )
    else:
        raise ValueError(f"Unknown fittype option: {fittype}")

    t2s_limited[np.isinf(t2s_limited)] = 500.0  # why 500?
    # let's get rid of negative values, but keep zeros where limited != full
    t2s_limited[(adaptive_mask_masked > 1) & (t2s_limited <= 0)] = 1.0
    t2s_limited = _apply_t2s_floor(t2s_limited, tes)
    s0_limited[np.isnan(s0_limited)] = 0.0  # why 0?
    t2s_full[np.isinf(t2s_full)] = 500.0  # why 500?
    t2s_full[t2s_full <= 0] = 1.0  # let's get rid of negative values!
    t2s_full = _apply_t2s_floor(t2s_full, tes)
    s0_full[np.isnan(s0_full)] = 0.0  # why 0?

    t2s_limited = utils.unmask(t2s_limited, mask)
    s0_limited = utils.unmask(s0_limited, mask)
    t2s_full = utils.unmask(t2s_full, mask)
    s0_full = utils.unmask(s0_full, mask)

    # set a hard cap for the T2* map
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s_limited.flatten(), 99.5, interpolation_method="lower")
    LGR.debug(f"Setting cap on T2* map at {cap_t2s * 10:.5f}")
    t2s_limited[t2s_limited > cap_t2s * 10] = cap_t2s

    return t2s_limited, s0_limited, t2s_full, s0_full


def fit_decay_ts(data, tes, mask, adaptive_mask, fittype):
    """Fit voxel- and timepoint-wise monoexponential decay models to ``data``.

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E,) :obj:`list`
        Echo times
    mask : (S,) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    adaptive_mask : (S,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    fittype : :obj: `str`
        The type of model fit to use

    Returns
    -------
    t2s_limited_ts : (S x T) :obj:`numpy.ndarray`
        Limited T2* map. The limited map only keeps the T2* values for data
        where there are at least two echos with good signal.
    s0_limited_ts : (S x T) :obj:`numpy.ndarray`
        Limited S0 map.  The limited map only keeps the S0 values for data
        where there are at least two echos with good signal.
    t2s_full_ts : (S x T) :obj:`numpy.ndarray`
        Full T2* timeseries.  For voxels affected by dropout, with good signal
        from only one echo, the full timeseries uses the single echo's value
        at that voxel/volume.
    s0_full_ts : (S x T) :obj:`numpy.ndarray`
        Full S0 timeseries. For voxels affected by dropout, with good signal
        from only one echo, the full timeseries uses the single echo's value
        at that voxel/volume.

    See Also
    --------
    : func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                                          parameter.
    """
    n_samples, _, n_vols = data.shape
    tes = np.array(tes)

    t2s_limited_ts = np.zeros([n_samples, n_vols])
    s0_limited_ts = np.copy(t2s_limited_ts)
    t2s_full_ts = np.copy(t2s_limited_ts)
    s0_full_ts = np.copy(t2s_limited_ts)

    report = True
    for vol in range(n_vols):
        t2s_limited, s0_limited, t2s_full, s0_full = fit_decay(
            data[:, :, vol][:, :, None], tes, mask, adaptive_mask, fittype, report=report
        )
        t2s_limited_ts[:, vol] = t2s_limited
        s0_limited_ts[:, vol] = s0_limited
        t2s_full_ts[:, vol] = t2s_full
        s0_full_ts[:, vol] = s0_full
        report = False

    return t2s_limited_ts, s0_limited_ts, t2s_full_ts, s0_full_ts
