"""
Functions to estimate S0 and T2* from multi-echo data.
"""
import logging
import scipy
import numpy as np
from tedana import utils

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def monoexponential(tes, s0, t2star):
    """
    Specifies a monoexponential model for use with scipy curve fitting

    Parameters
    ----------
    tes : (E,) :obj:`list`
        Echo times
    s0 : :obj:`float`
        Initial signal parameter
    t2star : :oj:`float`
        T2* parameter

    """
    return s0 * np.exp(-tes / t2star)


def fit_monoexponential(data_cat, echo_times, report=True):
    """
    Fit monoexponential decay model with nonlinear curve-fitting.

    Parameters
    ----------
    data_cat
    echo_times
    adaptive_mask

    Returns
    -------
    t2s_limited, s0_limited, t2s_full, s0_full
    """
    if report:
        RepLGR.info("A monoexponential model was fit to the data at each voxel "
                    "using nonlinear model fitting in order to estimate T2* and S0 "
                    "maps, using T2*/S0 estimates from a log-linear fit as "
                    "initial values. For each voxel, the value from the adaptive "
                    "mask was used to determine which echoes would be used to "
                    "estimate T2* and S0. In cases of model fit failure, T2*/S0 "
                    "estimates from the log-linear fit were retained instead.")
    n_samp, n_echos, n_vols = data_cat.shape

    # Currently unused
    # fit_data = np.mean(data_cat, axis=2)
    # fit_sigma = np.std(data_cat, axis=2)

    t2s, s0, r_squared = fit_loglinear(data_cat, echo_times, report=False)

    data_2d = data_cat.reshape(n_samp, -1).T
    echo_times_1d = np.repeat(echo_times, n_vols)

    # perform a monoexponential fit of echo times against MR signal
    # using loglin estimates as initial starting points for fit
    fail_count = 0
    for voxel in range(n_samp):
        try:
            popt, cov = scipy.optimize.curve_fit(
                monoexponential, echo_times_1d, data_2d[:, voxel],
                p0=(s0[voxel], t2s[voxel]),
                bounds=((np.min(data_2d[:, voxel]), 0),
                        (np.inf, np.inf)))
            s0[voxel] = popt[0]
            t2s[voxel] = popt[1]
        except (RuntimeError, ValueError):
            # If curve_fit fails to converge, fall back to loglinear estimate
            fail_count += 1

    if fail_count:
        fail_percent = 100 * fail_count / n_samp
        LGR.debug('With {0} echoes, monoexponential fit failed on {1}/{2} '
                  '({3:.2f}%) voxel(s), used log linear estimate '
                  'instead'.format(n_echos, fail_count, n_samp, fail_percent))

    # Determine model fit
    mean_data = np.mean(data_cat, axis=2)  # avg echo-wise data over time
    r_squared = calculate_r_squared(mean_data, echo_times, s0, t2s, report=report)
    return t2s, s0, r_squared


def fit_loglinear(data_cat, echo_times, report=True):
    """
    """
    if report:
        RepLGR.info("A monoexponential model was fit to the data at each voxel "
                    "using log-linear regression in order to estimate T2* and S0 "
                    "maps. For each voxel, the value from the adaptive mask was "
                    "used to determine which echoes would be used to estimate T2* "
                    "and S0.")
    n_samp, n_echos, n_vols = data_cat.shape

    # perform log linear fit of echo times against MR signal
    # make DV matrix: samples x (time series * echos)
    data_2d = data_cat.reshape(n_samp, -1).T
    log_data = np.log(np.abs(data_2d) + 1)

    # make IV matrix: intercept/TEs x (time series * echos)
    x = np.column_stack([np.ones(n_echos), [-te for te in echo_times]])
    X = np.repeat(x, n_vols, axis=0)

    # Log-linear fit
    betas = np.linalg.lstsq(X, log_data, rcond=None)[0]
    t2s = 1. / betas[1, :].T
    s0 = np.exp(betas[0, :]).T

    # Determine model fit
    mean_data = np.mean(data_cat, axis=2)  # avg echo-wise data over time
    r_squared = calculate_r_squared(mean_data, echo_times, s0, t2s, report=report)
    return t2s, s0, r_squared


def calculate_r_squared(data, echo_times, s0, t2s, report=True):
    """
    Calculate R^2 from data and T2*/S0 estimates.
    """
    assert t2s.ndim == s0.ndim == (data.ndim - 1)

    if report:
        RepLGR.info("Model fit, calculated as R-squared, was evaluated by "
                    "comparing predicted data from a monoexponential model "
                    "using the estimated T2* and S0 values against the data, "
                    "averaged over time.")

    if data.ndim == 2:
        data = data[:, :, None]

    if t2s.ndim == 1:
        t2s = t2s[:, None, None]
    else:
        t2s = t2s[:, None, :]

    if s0.ndim == 1:
        s0 = s0[:, None, None]
    else:
        s0 = s0[:, None, :]

    n_samp, n_echos, n_vols = data.shape
    echo_times_rep = np.tile(echo_times, (n_samp, n_vols, 1)).swapaxes(2, 1)
    s_pred = s0 * np.exp(-echo_times_rep / t2s)  # monoexp
    # Not sure if the sums here are correct
    ss_resid = np.sum((data - s_pred) ** 2, axis=1)
    var = np.var(data, axis=1)
    var[var == 0] = np.spacing(1)
    ss_total = (n_echos - 1) * var
    r_squared = 1 - (ss_resid / ss_total)
    r_squared = np.mean(r_squared, axis=1)
    return r_squared


def fit_decay(data, tes, mask, adaptive_mask, fittype, report=True):
    """
    Fit voxel-wise monoexponential decay models to `data`

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
        Valued array indicating number of echos that have sufficient signal in
        given sample
    fittype : {loglin, curvefit}
        The type of model fit to use

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

    Notes
    -----
    1.  Fit monoexponential decay function to all values for a given voxel
        across TRs, per TE, to estimate voxel-wise :math:`S_0` and
        :math:`T_2^*`:

        .. math::
            S(TE) = S_0 * exp(-R_2^* * TE)

            T_2^* = 1 / R_2^*

    2.  Replace infinite values in :math:`T_2^*` map with 500 and NaN values
        in :math:`S_0` map with 0.
    3.  Generate limited :math:`T_2^*` and :math:`S_0` maps by doing something.
    """
    if data.shape[1] != len(tes):
        raise ValueError('Second dimension of data ({0}) does not match number '
                         'of echoes provided (tes; {1})'.format(data.shape[1], len(tes)))
    elif not (data.shape[0] == mask.shape[0] == adaptive_mask.shape[0]):
        raise ValueError('First dimensions (number of samples) of data ({0}), '
                         'mask ({1}), and adaptive_mask ({2}) do not '
                         'match'.format(data.shape[0], mask.shape[0], adaptive_mask.shape[0]))

    data = data.copy()
    if data.ndim == 2:
        data = data[:, :, None]

    # Mask the inputs
    adaptive_mask = adaptive_mask.copy()
    data_masked = data[mask, :, :]
    adaptive_mask_masked = adaptive_mask[mask]

    n_samp, n_echos, n_vols = data_masked.shape

    # Start that loop
    echos_to_run = np.arange(3, n_echos + 1)

    t2s_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    r_squared_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    for i_echo, echo_num in enumerate(echos_to_run):
        temp_data = data_masked[:, :echo_num, :]
        temp_echo_times = tes[:echo_num]

        if (i_echo == 0) and report:
            report = True
        else:
            report = False

        if fittype == 'loglin':
            t2s, s0, r_squared = fit_loglinear(temp_data, temp_echo_times, report=report)
        elif fittype == 'curvefit':
            t2s, s0, r_squared = fit_monoexponential(temp_data, temp_echo_times, report=report)
        else:
            raise ValueError('Unknown fittype option: {}'.format(fittype))

        t2s_asc_maps[:, i_echo] = t2s
        s0_asc_maps[:, i_echo] = s0
        r_squared_asc_maps[:, i_echo] = r_squared

    # determine r_squared sitch
    adaptive_mask_r2 = np.zeros(n_samp, int)
    r_squared = np.zeros(n_samp)
    good_idx = np.vstack(np.where(r_squared_asc_maps >= 0.8))

    # split idx by voxel
    split_idx = np.split(
        good_idx,
        np.where(np.diff(good_idx[0, :]))[0]+1,
        axis=1
    )
    for i_voxel in range(len(split_idx)):
        voxel_idx = split_idx[i_voxel][0, 0]
        echo_idx = split_idx[i_voxel][1, :].max()
        adaptive_mask_r2[voxel_idx] = echo_idx + 3  # set to proper echo number
        r_squared[voxel_idx] = r_squared_asc_maps[voxel_idx, echo_idx]

    # Grab full r-squared for voxels with poor fit at all subsets
    bad_idx = np.where(adaptive_mask_r2 == 0)[0]
    r_squared[bad_idx] = r_squared_asc_maps[bad_idx, -1]

    # Grab T2*/S0 from the appropriate combination of echoes
    adaptive_mask = np.minimum(adaptive_mask_r2, adaptive_mask_masked)
    echo_masks = np.zeros([n_samp, len(echos_to_run)], dtype=bool)
    for i_echo, echo_num in enumerate(echos_to_run):
        # Create echo masks to assign values to limited vs full maps later
        echo_masks[adaptive_mask == echo_num, i_echo] = True

    # create limited T2* and S0 maps
    t2s_limited = utils.unmask(t2s_asc_maps[echo_masks], adaptive_mask >= 3)
    s0_limited = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask >= 3)

    # Use T2*/S0 from first three echoes for bad voxels
    t2s_full, s0_full = t2s_limited.copy(), s0_limited.copy()
    t2s_full[adaptive_mask < 3] = t2s_asc_maps[adaptive_mask < 3, 0]
    s0_full[adaptive_mask < 3] = s0_asc_maps[adaptive_mask < 3, 0]

    # Restrict calculated values
    # let's get rid of negative values, but keep zeros where limited != full
    bad_idx = (adaptive_mask >= 3) & (t2s_limited <= 0)
    adaptive_mask[bad_idx] = 0
    t2s_limited[bad_idx] = 1.
    t2s_full[t2s_full <= 0] = 1.  # let's get rid of negative values!

    bad_idx = r_squared < 0
    adaptive_mask[bad_idx] = 0
    r_squared[bad_idx] = 0  # r^2 can be negative when fit is awful

    t2s_limited[np.isinf(t2s_limited)] = 500.  # why 500?
    t2s_full[np.isinf(t2s_full)] = 500.  # why 500?
    s0_limited[np.isnan(s0_limited)] = 0.  # why 0?
    s0_full[np.isnan(s0_full)] = 0.  # why 0?

    t2s_limited = utils.unmask(t2s_limited, mask)
    s0_limited = utils.unmask(s0_limited, mask)
    t2s_full = utils.unmask(t2s_full, mask)
    s0_full = utils.unmask(s0_full, mask)
    r_squared = utils.unmask(r_squared, mask)
    adaptive_mask = utils.unmask(adaptive_mask, mask)
    return t2s_limited, s0_limited, t2s_full, s0_full, r_squared, adaptive_mask


def fit_decay_ts(data, tes, mask, adaptive_mask, fittype, report=True):
    """
    Fit voxel- and timepoint-wise monoexponential decay models to `data`

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
        Valued array indicating number of echos that have sufficient signal in
        given sample
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
    """
    # Mask the inputs
    adaptive_mask = adaptive_mask.copy()
    data_masked = data[mask, :, :]
    adaptive_mask_masked = adaptive_mask[mask]

    n_samp, n_echos, n_vols = data_masked.shape

    # Start that loop
    echos_to_run = np.arange(3, n_echos + 1)

    t2s_asc_maps = np.zeros([n_samp, len(echos_to_run), n_vols])
    s0_asc_maps = np.zeros([n_samp, len(echos_to_run), n_vols])
    r_squared_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    for i_echo, echo_num in enumerate(echos_to_run):
        temp_echo_times = tes[:echo_num]
        for j_vol in range(n_vols):
            temp_data = data_masked[:, :echo_num, j_vol][:, :, None]

            if (i_echo == 0) and (j_vol == 0) and report:
                report = True
            else:
                report = False

            if fittype == 'loglin':
                t2s, s0, _ = fit_loglinear(temp_data, temp_echo_times, report=report)
            elif fittype == 'curvefit':
                t2s, s0, _ = fit_monoexponential(temp_data, temp_echo_times, report=report)
            else:
                raise ValueError('Unknown fittype option: {}'.format(fittype))

            t2s_asc_maps[:, i_echo, j_vol] = t2s
            s0_asc_maps[:, i_echo, j_vol] = s0

        r_squared_asc_maps[:, i_echo] = calculate_r_squared(
            temp_data, temp_echo_times,
            s0_asc_maps[:, i_echo, :],
            t2s_asc_maps[:, i_echo, :],
            report=False
        )

    # determine r_squared sitch
    adaptive_mask_r2 = np.zeros(n_samp, int)
    r_squared = np.zeros(n_samp)
    good_idx = np.vstack(np.where(r_squared_asc_maps >= 0.8))

    # split idx by voxel
    split_idx = np.split(
        good_idx,
        np.where(np.diff(good_idx[0, :]))[0]+1,
        axis=1
    )
    for i_voxel in range(len(split_idx)):
        voxel_idx = split_idx[i_voxel][0, 0]
        echo_idx = split_idx[i_voxel][1, :].max()
        adaptive_mask_r2[voxel_idx] = echo_idx + 3  # set to proper echo number
        r_squared[voxel_idx] = r_squared_asc_maps[voxel_idx, echo_idx]

    # Grab full r-squared for voxels with poor fit at all subsets
    bad_idx = np.where(adaptive_mask_r2 == 0)[0]
    r_squared[bad_idx] = r_squared_asc_maps[bad_idx, -1]

    # Grab T2*/S0 from the appropriate combination of echoes
    adaptive_mask = np.minimum(adaptive_mask_r2, adaptive_mask_masked)
    echo_masks = np.zeros([n_samp, len(echos_to_run)], dtype=bool)
    for i_echo, echo_num in enumerate(echos_to_run):
        # Create echo masks to assign values to limited vs full maps later
        echo_masks[adaptive_mask == echo_num, i_echo] = True

    # create limited T2* and S0 maps
    t2s_limited = utils.unmask(t2s_asc_maps[echo_masks], adaptive_mask >= 3)
    s0_limited = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask >= 3)

    # Use T2*/S0 from first three echoes for bad voxels
    t2s_full, s0_full = t2s_limited.copy(), s0_limited.copy()
    t2s_full[adaptive_mask < 3] = t2s_asc_maps[adaptive_mask < 3, 0]
    s0_full[adaptive_mask < 3] = s0_asc_maps[adaptive_mask < 3, 0]

    t2s_limited = utils.unmask(t2s_limited, mask)
    s0_limited = utils.unmask(s0_limited, mask)
    t2s_full = utils.unmask(t2s_full, mask)
    s0_full = utils.unmask(s0_full, mask)
    r_squared = utils.unmask(r_squared, mask)
    adaptive_mask = utils.unmask(adaptive_mask, mask)
    return t2s_limited, s0_limited, t2s_full, s0_full, r_squared, adaptive_mask
