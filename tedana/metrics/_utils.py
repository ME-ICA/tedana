"""
Misc. utils for metric calculation.
"""
import logging

import numpy as np
from scipy import stats

LGR = logging.getLogger(__name__)


def dependency_resolver(dict_, requested_metrics, base_inputs):
    """
    Identify all necessary metrics based on a list of requested metrics and
    the metrics each one requires to be calculated, as defined in a dictionary.

    Parameters
    ----------
    dict_ : :obj:`dict`
        Dictionary containing lists, where each key is a metric name and its
        associated value is the list of metrics or inputs required to calculate
        it.
    requested_metrics : :obj:`list`
        Child metrics for which the function will determine parents.
    base_inputs : :obj:`list`
        A list of inputs to the metric collection function, to differentiate
        them from metrics to be calculated.

    Returns
    -------
    required_metrics :obj:`list`
        A comprehensive list of all metrics and inputs required to generate all
        of the requested inputs.
    """
    not_found = [k for k in requested_metrics if k not in dict_.keys()]
    if not_found:
        raise ValueError('Unknown metric(s): {}'.format(', '.join(not_found)))

    required_metrics = requested_metrics
    escape_counter = 0
    while True:
        required_metrics_new = required_metrics[:]
        for k in required_metrics:
            if k in dict_.keys():
                new_metrics = dict_[k]
            elif k not in base_inputs:
                print("Warning: {} not found".format(k))
            required_metrics_new += new_metrics
        if set(required_metrics) == set(required_metrics_new):
            # There are no more parent metrics to calculate
            break
        else:
            required_metrics = required_metrics_new
        escape_counter += 1
        if escape_counter >= 10:
            LGR.warning('dependency_resolver in infinite loop. Escaping early.')
            break
    return required_metrics


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
    """
    Flip an arbitrary set of input arrays based on a set of signs.

    Parameters
    ----------
    *args : array_like
        Any number of arrays with one dimension the same length as signs.
        If multiple dimensions share the same size as signs, behavior of this
        function will be unpredictable.
    signs : array_like of :obj:`int`
        Array of +/- 1 by which to flip the values in each argument.

    Returns
    -------
    *args : array_like
        Input arrays after sign flipping.
    """
    assert signs.ndim == 1, 'Argument "signs" must be one-dimensional.'
    for arg in args:
        assert len(signs) in arg.shape, \
            ('Size of argument "signs" must match size of one dimension in '
             'each of the input arguments.')
        assert sum(x == len(signs) for x in arg.shape) == 1, \
            ('Only one dimension of each input argument can match the length '
             'of argument "signs".')
    # correct mixing & weights signs based on spatial distribution tails
    return [arg * signs for arg in args]


def sort_df(df, by='kappa', ascending=False):
    """
    Sort DataFrame and get index.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame to sort.
    by : :obj:`str` or None, optional
        Column by which to sort the DataFrame. Default is 'kappa'.
    ascending : :obj:`bool`, optional
        Whether to sort the DataFrame in ascending (True) or descending (False)
        order. Default is False.

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        DataFrame after sorting, with index resetted.
    argsort : array_like
        Sorting index.
    """
    if by is None:
        return df, df.index.values

    # Order of kwargs is preserved at 3.6+
    argsort = df[by].argsort()
    if not ascending:
        argsort = argsort[::-1]
    df = df.loc[argsort].reset_index(drop=True)
    return df, argsort


def apply_sort(*args, sort_idx, axis=0):
    """
    Apply a sorting index to an arbitrary set of arrays.
    """
    for arg in args:
        assert arg.shape[axis] == len(sort_idx)
    return [np.take(arg, sort_idx, axis=axis) for arg in args]


def check_mask(data, mask):
    """
    Check that no zero-variance voxels remain in masked data.

    Parameters
    ----------
    data : (S [x E] x T) array_like
        Data to be masked and evaluated.
    mask : (S) array_like
        Boolean mask.

    Raises
    ------
    ValueError
    """
    assert data.ndim <= 3
    assert mask.shape[0] == data.shape[0]
    masked_data = data[mask, ...]
    dims_to_check = list(range(1, data.ndim))
    for dim in dims_to_check:
        # ignore singleton dimensions
        if masked_data.shape[dim] == 1:
            continue

        masked_data_std = masked_data.std(axis=dim)
        zero_idx = np.where(masked_data_std == 0)
        n_bad_voxels = len(zero_idx[0])
        if n_bad_voxels > 0:
            raise ValueError('{0} voxels in masked data have zero variance. '
                             'Mask is too liberal.'.format(n_bad_voxels))
