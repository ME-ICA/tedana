"""
Utility functions for tedana decomposition
"""
import logging

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
