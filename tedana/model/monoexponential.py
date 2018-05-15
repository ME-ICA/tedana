"""
Functions to estimate S0 and T2* from multi-echo data.
"""
import numpy as np
from tedana import utils


def fit_decay(data, tes, mask, masksum, start_echo):
    """
    Fit voxel-wise monoexponential decay models to `data`

    Parameters
    ----------
    data : (S x E [x T]) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E, ) list
        Echo times
    mask : (S, ) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    masksum : (S, ) array_like
        Valued array indicating number of echos that have sufficient signal in
        given sample
    start_echo : int
        First echo to consider

    Returns
    -------
    t2sa : (S x E) :obj:`numpy.ndarray`
        Limited T2* map
    s0va : (S x E) :obj:`numpy.ndarray`
        Limited S0 map
    t2ss : (S x E) :obj:`numpy.ndarray`
        ???
    s0vs : (S x E) :obj:`numpy.ndarray`
        ???
    t2saf : (S x E) :obj:`numpy.ndarray`
        Full T2* map
    s0vaf : (S x E) :obj:`numpy.ndarray`
        Full S0 map

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
    if len(data.shape) == 3:
        n_samp, n_echos, n_vols = data.shape
    else:
        n_samp, n_echos = data.shape
        n_vols = 1

    data = data[mask]
    t2ss = np.zeros([n_samp, n_echos - 1])
    s0vs = np.zeros([n_samp, n_echos - 1])

    for echo in range(start_echo, n_echos + 1):
        # perform log linear fit of echo times against MR signal
        # make DV matrix: samples x (time series * echos)
        B = np.log((np.abs(data[:, :echo, :]) + 1).reshape(len(data), -1).T)
        # make IV matrix: intercept/TEs x (time series * echos)
        x = np.column_stack([np.ones(echo), [-te for te in tes[:echo]]])
        X = np.repeat(x, n_vols, axis=0)

        beta = np.linalg.lstsq(X, B, rcond=None)[0]
        t2s = 1. / beta[1, :].T
        s0 = np.exp(beta[0, :]).T

        t2s[np.isinf(t2s)] = 500.  # why 500?
        s0[np.isnan(s0)] = 0.      # why 0?

        t2ss[..., echo - 2] = np.squeeze(utils.unmask(t2s, mask))
        s0vs[..., echo - 2] = np.squeeze(utils.unmask(s0, mask))

    # create limited T2* and S0 maps
    fl = np.zeros([n_samp, len(tes) - 1], dtype=bool)
    for echo in range(n_echos - 1):
        fl_ = np.squeeze(fl[..., echo])
        fl_[masksum == echo + 2] = True
        fl[..., echo] = fl_
    t2sa = utils.unmask(t2ss[fl], masksum > 1)
    s0va = utils.unmask(s0vs[fl], masksum > 1)

    # create full T2* maps with S0 estimation errors
    t2saf, s0vaf = t2sa.copy(), s0va.copy()
    t2saf[masksum == 1] = t2ss[masksum == 1, 0]
    s0vaf[masksum == 1] = s0vs[masksum == 1, 0]

    return t2sa, s0va, t2ss, s0vs, t2saf, s0vaf


def fit_decay_ts(data, mask, tes, masksum, start_echo):
    """
    Fit voxel- and timepoint-wise monoexponential decay models to `data`

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E, ) list
        Echo times
    mask : (S, ) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    masksum : (S, ) array_like
        Valued array indicating number of echos that have sufficient signal in
        given sample
    start_echo : int
        First echo to consider

    Returns
    -------
    t2sa_ts : (S x E x T) :obj:`numpy.ndarray`
        Limited T2* map
    s0va_ts : (S x E x T) :obj:`numpy.ndarray`
        Limited S0 map
    t2saf_ts : (S x E x T) :obj:`numpy.ndarray`
        Full T2* map
    s0vaf_ts : (S x E x T) :obj:`numpy.ndarray`
        Full S0 map
    """
    n_samples, _, n_trs = data.shape
    echodata = data[mask]
    tes = np.array(tes)

    t2sa_ts = np.zeros([n_samples, n_trs])
    s0va_ts = np.copy(t2sa_ts)
    t2saf_ts = np.copy(t2sa_ts)
    s0vaf_ts = np.copy(t2sa_ts)

    for vol in range(echodata.shape[-1]):

        [t2sa, s0va, _, _, t2saf, s0vaf] = fit_decay(
            data, mask, tes, masksum, start_echo)

        t2sa_ts[:, vol] = t2sa
        s0va_ts[:, vol] = s0va
        t2saf_ts[:, vol] = t2saf
        s0vaf_ts[:, vol] = s0vaf

    return t2sa_ts, s0va_ts, t2saf_ts, s0vaf_ts
