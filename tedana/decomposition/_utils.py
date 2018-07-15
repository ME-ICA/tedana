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
    ees : (N,) list
        Indices of echos to assess from `dd` in calculating output

    Returns
    -------
    imask : (S x N) :obj:`numpy.ndarray`
        Boolean array denoting
    """

    if ees is None:
        ees = range(dd.shape[1])
    imask = np.zeros([dd.shape[0], len(ees)], dtype=bool)
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
    Wavelet transform data.

    Parameters
    ----------
    mmix : :obj:`numpy.ndarray`
        Data to wavelet transform.

    Returns
    -------
    mmix_wt : :obj:`numpy.ndarray`
        Wavelet-transformed data.
    cAlen : :obj:`int`
        Index of some kind?
    """
    llt = len(np.hstack(pywt.dwt(mmix[0], 'db2')))
    mmix_wt = np.zeros([mmix.shape[0], llt])
    for ii in range(mmix_wt.shape[0]):
        wtx = pywt.dwt(mmix[ii], 'db2')
        mmix_wt[ii] = np.hstack(wtx)
    cAlen = len(wtx[0])
    return mmix_wt, cAlen


def idwtmat(mmix_wt, cAl):
    """
    Invert wavelet transform data.

    Parameters
    ----------
    mmix_wt : :obj:`numpy.ndarray`
        Wavelet-transformed data.
    cAl : :obj:`int`
        Index of some kind?

    Returns
    -------
    mmix_iwt : :obj:`numpy.ndarray`
        Inverse wavelet-transformed data.
    """
    lt = len(pywt.idwt(mmix_wt[0, :cAl], mmix_wt[0, cAl:], 'db2'))
    mmix_iwt = np.zeros([mmix_wt.shape[0], lt])
    for ii in range(mmix_iwt.shape[0]):
        mmix_iwt[ii] = pywt.idwt(mmix_wt[ii, :cAl], mmix_wt[ii, cAl:], 'db2')
    return mmix_iwt
