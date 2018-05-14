"""
Functions to estimate S0 and T2* from multi-echo data.
"""
import logging

import numpy as np

from tedana import utils

logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
LGR = logging.getLogger(__name__)


def fit_decay(data, tes, mask, masksum, start_echo):
    """
    Fit voxel-wise monoexponential decay models to estimate T2* and S0 maps.

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
    dim : The dimensions we wish to consider - e.g. if dim is 4 we only loop over
          x by y by z by E. If dim is 5 we loop over x by y by z by E by T.

    Returns
    -------
    t2sa : (S x E x T) :obj:`numpy.ndarray`
        Limited T2* map
    s0va : (S x E x T) :obj:`numpy.ndarray`
        Limited S0 map
    t2ss : (S x E x T) :obj:`numpy.ndarray`
        ???
    s0vs : (S x E x T) :obj:`numpy.ndarray`
        ???
    t2saf : (S x E x T) :obj:`numpy.ndarray`
        Full T2* map
    s0vaf : (S x E x T) :obj:`numpy.ndarray`
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
        n_samples, n_echoes, n_trs = data.shape
    else:
        n_samples, n_echoes = data.shape
        n_trs = 1

    data = data[mask]
    n_voxels = data.shape[0]
    tes = np.array(tes)

    t2ss = np.zeros([n_samples, n_echoes - 1])
    s0vs = t2ss.copy()

    # Fit monoexponential decay first for first echo only,
    # then first two echoes, etc.
    for i_echo in range(start_echo, n_echoes + 1):
        # Do Log Linear fit
        B = np.reshape(np.abs(data[:, :i_echo, :]) + 1,
                       (n_voxels, i_echo*n_trs)).transpose()
        B = np.log(B)
        neg_tes = -1 * tes[:i_echo]

        # First row is constant, second is TEs for decay curve
        # Independent variables for least-squares model
        x = np.array([np.ones(i_echo), neg_tes])
        X = np.tile(x, (1, n_trs))
        X = np.sort(X)[:, ::-1].transpose()

        beta, _, _, _ = np.linalg.lstsq(X, B)
        t2s = 1. / beta[1, :].transpose()
        s0 = np.exp(beta[0, :]).transpose()

        t2s[np.isinf(t2s)] = 500.
        s0[np.isnan(s0)] = 0.

        t2ss[..., i_echo-2] = np.squeeze(utils.unmask(t2s, mask))
        s0vs[..., i_echo-2] = np.squeeze(utils.unmask(s0, mask))

    # Limited T2* and S0 maps
    fl = np.zeros([n_samples, len(tes)-1], bool)
    for i_echo in range(n_echoes - 1):
        fl_ = np.squeeze(fl[..., i_echo])
        fl_[masksum == i_echo + 2] = True
        fl[..., i_echo] = fl_
    t2sa = np.squeeze(utils.unmask(t2ss[fl], masksum > 1))
    s0va = np.squeeze(utils.unmask(s0vs[fl], masksum > 1))

    # Full T2* maps with S0 estimation errors
    t2saf = t2sa.copy()
    s0vaf = s0va.copy()
    t2saf[masksum == 1] = t2ss[masksum == 1, 0]
    s0vaf[masksum == 1] = s0vs[masksum == 1, 0]

    return t2sa, s0va, t2ss, s0vs, t2saf, s0vaf


def fit_decay_ts(data, mask, tes, masksum, start_echo):
    """
    Fit voxel- and timepoint-wise monoexponential decay models to estimate
    T2* and S0 timeseries.

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
    t2sa : (S x E x T) :obj:`numpy.ndarray`
        Limited T2* map
    s0va : (S x E x T) :obj:`numpy.ndarray`
        Limited S0 map
    t2ss : (S x E x T) :obj:`numpy.ndarray`
        ???
    s0vs : (S x E x T) :obj:`numpy.ndarray`
        ???
    t2saf : (S x E x T) :obj:`numpy.ndarray`
        Full T2* map
    s0vaf : (S x E x T) :obj:`numpy.ndarray`
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
