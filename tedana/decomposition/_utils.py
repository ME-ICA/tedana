"""
Utility functions for tedana decomposition
"""
import logging

import pywt
import numpy as np
from scipy import stats

LGR = logging.getLogger(__name__)


def eimask(dd, ees=None):
    """
    Returns mask for data between [0.001, 5] * 98th percentile of dd

    Parameters
    ----------
    dd : (S x E x T) array_like
        Input data, where `S` is samples, `E` is echos, and `T` is time
    ees : (N,) :obj:`list`
        Indices of echos to assess from `dd` in calculating output

    Returns
    -------
    imask : (S x N) :obj:`numpy.ndarray`
        Boolean array denoting
    """

    if ees is None:
        ees = range(dd.shape[1])
    imask = np.zeros((dd.shape[0], len(ees)), dtype=bool)
    for ee in ees:
        LGR.debug('Creating eimask for echo {}'.format(ee))
        perc98 = stats.scoreatpercentile(dd[:, ee, :].flatten(), 98,
                                         interpolation_method='lower')
        lthr, hthr = 0.001 * perc98, 5 * perc98
        LGR.debug('Eimask threshold boundaries: '
                  '{:.03f} {:.03f}'.format(lthr, hthr))
        m = dd[:, ee, :].mean(axis=1)
        imask[np.logical_and(m > lthr, m < hthr), ee] = True

    return imask


def dwtmat(mmix):
    """
    Wavelet transform data using order 2 Daubechies wavelet.

    Parameters
    ----------
    mmix : {S, T} :obj:`numpy.ndarray`
        Data to wavelet transform.

    Returns
    -------
    mmix_wt : {S, 2C} :obj:`numpy.ndarray`
        Wavelet-transformed data. Approximation and detail coefficients are
        horizontally concatenated for each row in mmix.
    n_coefs_approx : :obj:`int`
        The number of approximation coefficients from the wavelet
        transformation. Used to split the wavelet-transformed data into
        approximation and detail coefficients.
    """
    n_samp = mmix.shape[0]
    coefs_tup = pywt.dwt(mmix[0, :], 'db2')
    n_coefs_approx = len(coefs_tup[0])
    n_coefs_total = len(np.hstack(coefs_tup))
    mmix_wt = np.zeros((n_samp, n_coefs_total))
    for i_samp in range(n_samp):
        coefs_tup = pywt.dwt(mmix[i_samp], 'db2')
        mmix_wt[i_samp] = np.hstack(coefs_tup)

    return mmix_wt, n_coefs_approx


def idwtmat(mmix_wt, n_coefs_approx):
    """
    Invert wavelet transform data with order 2 Daubechies wavelet.

    Parameters
    ----------
    mmix_wt : {S, 2C} :obj:`numpy.ndarray`
        Wavelet-transformed data. Approximation and detail coefficients are
        horizontally concatenated for each row in mmix.
    n_coefs_approx : :obj:`int`
        The number of approximation coefficients from the wavelet
        transformation. Used to split the wavelet-transformed data into
        approximation and detail coefficients.

    Returns
    -------
    mmix_iwt : {S, T} :obj:`numpy.ndarray`
        Inverse wavelet-transformed data.
    """
    n_samp = mmix_wt.shape[0]
    coefs_approx = mmix_wt[:, :n_coefs_approx]
    coefs_detail = mmix_wt[:, n_coefs_approx:]
    n_trs = len(pywt.idwt(coefs_approx[0, :], coefs_detail[0, :], 'db2'))
    mmix_iwt = np.zeros((n_samp, n_trs))
    for i_samp in range(n_samp):
        mmix_iwt[i_samp] = pywt.idwt(coefs_approx[i_samp, :],
                                     coefs_detail[i_samp, :], 'db2')

    return mmix_iwt
