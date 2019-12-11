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


def fit_monoexponential(data_cat, echo_times, adaptive_mask):
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

    t2s_limited, s0_limited, t2s_full, s0_full = fit_loglinear(
        data_cat, echo_times, adaptive_mask, report=False)

    echos_to_run = np.unique(adaptive_mask)
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    t2s_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    echo_masks = np.zeros([n_samp, len(echos_to_run)], dtype=bool)

    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
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
                    monoexponential, echo_times_1d, data_2d[:, voxel],
                    p0=(s0_full[voxel], t2s_full[voxel]),
                    bounds=((np.min(data_2d[:, voxel]), 0),
                            (np.inf, np.inf)))
                s0_full[voxel] = popt[0]
                t2s_full[voxel] = popt[1]
            except (RuntimeError, ValueError):
                # If curve_fit fails to converge, fall back to loglinear estimate
                fail_count += 1

        if fail_count:
            fail_percent = 100 * fail_count / len(voxel_idx)
            LGR.debug('With {0} echoes, monoexponential fit failed on {1}/{2} '
                      '({3:.2f}%) voxel(s), used log linear estimate '
                      'instead'.format(echo_num, fail_count, len(voxel_idx), fail_percent))

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
    """
    """
    if report:
        RepLGR.info("A monoexponential model was fit to the data at each voxel "
                    "using log-linear regression in order to estimate T2* and S0 "
                    "maps. For each voxel, the value from the adaptive mask was "
                    "used to determine which echoes would be used to estimate T2* "
                    "and S0.")
    n_samp, n_echos, n_vols = data_cat.shape

    echos_to_run = np.unique(adaptive_mask)
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    t2s_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    echo_masks = np.zeros([n_samp, len(echos_to_run)], dtype=bool)

    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
            voxel_idx = np.where(adaptive_mask <= echo_num)[0]
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
        X = np.repeat(x, n_vols, axis=0)

        # Log-linear fit
        betas = np.linalg.lstsq(X, log_data, rcond=None)[0]
        t2s = 1. / betas[1, :].T
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


def fit_decay(data, tes, mask, adaptive_mask, fittype):
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
    data_masked = data[mask, :, :]
    adaptive_mask_masked = adaptive_mask[mask]

    if fittype == 'loglin':
        t2s_limited, s0_limited, t2s_full, s0_full = fit_loglinear(
            data_masked, tes, adaptive_mask_masked)
    elif fittype == 'curvefit':
        t2s_limited, s0_limited, t2s_full, s0_full = fit_monoexponential(
            data_masked, tes, adaptive_mask_masked)
    else:
        raise ValueError('Unknown fittype option: {}'.format(fittype))

    t2s_limited[np.isinf(t2s_limited)] = 500.  # why 500?
    # let's get rid of negative values, but keep zeros where limited != full
    t2s_limited[(adaptive_mask_masked > 1) & (t2s_limited <= 0)] = 1.
    s0_limited[np.isnan(s0_limited)] = 0.  # why 0?
    t2s_full[np.isinf(t2s_full)] = 500.  # why 500?
    t2s_full[t2s_full <= 0] = 1.  # let's get rid of negative values!
    s0_full[np.isnan(s0_full)] = 0.  # why 0?

    t2s_limited = utils.unmask(t2s_limited, mask)
    s0_limited = utils.unmask(s0_limited, mask)
    t2s_full = utils.unmask(t2s_full, mask)
    s0_full = utils.unmask(s0_full, mask)

    return t2s_limited, s0_limited, t2s_full, s0_full


def fit_decay_ts(data, tes, mask, adaptive_mask, fittype):
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
    n_samples, _, n_vols = data.shape
    tes = np.array(tes)

    t2s_limited_ts = np.zeros([n_samples, n_vols])
    s0_limited_ts = np.copy(t2s_limited_ts)
    t2s_full_ts = np.copy(t2s_limited_ts)
    s0_full_ts = np.copy(t2s_limited_ts)

    for vol in range(n_vols):
        t2s_limited, s0_limited, t2s_full, s0_full = fit_decay(
            data[:, :, vol][:, :, None], tes, mask, adaptive_mask, fittype)
        t2s_limited_ts[:, vol] = t2s_limited
        s0_limited_ts[:, vol] = s0_limited
        t2s_full_ts[:, vol] = t2s_full
        s0_full_ts[:, vol] = s0_full

    return t2s_limited_ts, s0_limited_ts, t2s_full_ts, s0_full_ts
