"""
Misc. utils for metric calculation.
"""
import numpy as np
from scipy import stats

from tedana.stats import computefeats2


def determine_signs(weights, axis=0):
    """
    Determine component-wise optimal signs using voxel-wise parameter estimates.

    Parameters
    ----------
    weights : (S x C) array_like
        Parameter estimates for optimally combined data against the mixing
        matrix.

    Returns
    -------
    signs : (C) array_like
        Array of 1 and -1 values corresponding to the appropriate flips for the
        mixing matrix's component time series.
    """
    # compute skews to determine signs based on unnormalized weights,
    signs = stats.skew(weights, axis=axis)
    signs /= np.abs(signs)
    return signs


def flip_components(*args, signs):
    # correct mixing & weights signs based on spatial distribution tails
    return [arg * signs for arg in args]


def sort_df(df, by='kappa', ascending=False):
    """
    Sort DataFrame and get index.
    """
    # Order of kwargs is preserved at 3.6+
    argsort = df[by].argsort()
    if not ascending:
        argsort = argsort[::-1]
    df = df.loc[argsort].reset_index(drop=True)
    return df, argsort


def apply_sort(*args, sort_idx, axis=0):
    """
    Apply a sorting index.
    """
    for arg in args:
        assert arg.shape[axis] == len(sort_idx)
    return [np.take(arg, sort_idx, axis=axis) for arg in args]
