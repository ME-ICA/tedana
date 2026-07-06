"""Temporal noise metrics."""
import numpy as np
from scipy.signal import detrend
from scipy.stats import kurtosis


def compute_spike(*, mixing: np.ndarray) -> np.ndarray:
    """Temporal kurtosis of each component time series (transient-spike index).

    Each column is linearly detrended (removing drift without inflating the
    statistic for slow/task structure), then its Fisher kurtosis is taken. A
    component dominated by an isolated transient has high kurtosis; smooth or
    oscillatory signal stays low. Sign-invariant.

    Parameters
    ----------
    mixing : (T x C) array_like
        Component mixing matrix (time series per component).

    Returns
    -------
    (C,) :obj:`numpy.ndarray`
        Fisher temporal kurtosis per component.
    """
    detrended = detrend(np.asarray(mixing, dtype=np.float64), axis=0, type="linear")
    return kurtosis(detrended, axis=0, fisher=True)
