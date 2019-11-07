"""
Utility functions for tedana.selection
"""
import logging

import numpy as np

LGR = logging.getLogger(__name__)

# Functions that are used for interacting with comptable


def selectcomps2use(comptable, decide_comps):
    """
    Give a list of components that fit a classification types
    """

    if decide_comps == 'all':
        # This classification always has a value, this is a way to get all
        # values in the same structure as the subset lists below
        # There's probably a simpler way to do this
        comps2use = comptable['classification'] is not None
    elif (type(decide_comps) == str):
        comps2use = comptable['classification'] == decide_comps
    else:
        comps2use = comptable['classification'] == decide_comps[0]
        for didx in range(len(decide_comps - 1)):
            comps2use = (comps2use | (comptable['classification'] == decide_comps[didx + 1]))

    # If no components are selected, then return None.
    # The function that called this can check for None and exit before
    # attempting any computations on no data
    if comps2use.sum() == 0:
        comps2use = None

    return comps2use


def change_comptable_classifications(comptable, iftrue, iffalse,
                                     decision_boolean, nodeidxstr, decision_tree_steps):
    """
    Given information on whether a decision critereon is true or false for each component
    change or don't change the compnent classification
    """
    if iftrue != 'no_change':
        changeidx = decision_boolean.index[decision_boolean.values is True]
        comptable['classification'][changeidx] = iftrue
        comptable['rationale'][changeidx] += (nodeidxstr + ': ' + iftrue + '; ')
    if iffalse != 'no_change':
        changeidx = decision_boolean.index[decision_boolean.values is False]
        comptable['classification'][changeidx] = iffalse
        comptable['rationale'][changeidx] += (nodeidxstr + ': ' + iffalse + '; ')

    decision_tree_steps[-1]['numtrue'] = (decision_boolean is True).sum()
    decision_tree_steps[-1]['numfalse'] = (decision_boolean is False).sum()

    return comptable, decision_tree_steps


def clean_dataframe(comptable):
    """
    Reorder columns in component table so "rationale" and "classification" are
    last and remove trailing semicolons from rationale column.
    """
    cols_at_end = ['classification', 'rationale']
    comptable = (comptable[[c for c in comptable if c not in cols_at_end] +
                 [c for c in cols_at_end if c in comptable]])
    comptable['rationale'] = comptable['rationale'].str.rstrip(';')
    return comptable

# Functions that validate inputted parameters or other processing steps


def confirm_metrics_exist(comptable, necessary_metrics, functionname=None):
    """
    Confirm that all metrics declared in necessary_metrics are
    already included in comptable.

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
            Component metric table. One row for each component, with a column for
            each metric. The index should be the component number.
    necessary_metrics :obj:`list` a 1D list of strings of metrics

    Returns
    -------
    metrics_exist : :obj:`bool`
            True if all metrics in necessary_metrics are on the comptable
            False if one or more metrics are in necessary_metrics, but not in the comptable
    missing_metrics : :obj:`list`
            if metrics_exist then this is a list of strings containing the metric
            names in necessary_metrics that aren't in the comptable
            if not metrics_exist then an empty list

    If metrics_exist is false then print an error and end the program

    Notes
    -----
    This doesn't check if there are data in each metric's column, just that
    the columns exist. Also, this requires identical strings for the names
    of the metrics in necessary_metrics and the column labels in comptable
    """

    missing_metrics = np.setdiff1d(necessary_metrics, list(set(comptable.columns)))
    metrics_exist = not missing_metrics

    if metrics_exist is False:
        if functionname:
            error_msg = ("Necessary metrics for " + functionname + " are not in comptable. "
                         "Need to calculate the following metrics: " + str(missing_metrics))
        else:
            error_msg = ("Necessary metrics are not in comptable (calling function unknown). "
                         "Need to calculate the following metrics: " + str(missing_metrics))
        raise ValueError(error_msg)

    return metrics_exist


def are_only_necessary_metrics_used(used_metrics, necessary_metrics, functionname):
    """
    Checks if all metrics that are declared as necessary are actually used and
    if any used_metrics weren't explicitly declared. If either of these happen,
    a warning is added to the logger that notes which metrics weren't declared
    or used.

    Parameters
    ----------
    used_metrics: :obj:`list`
            A list of strings of the metric names that were used in the
            decision tree
    necessary_metrics: :obj:`list`
            A list of strings of the metric names that were declared
            to be used in the decision tree
    functionname: :obj:`str`
            The function name for the decision tree that was run

    Returns
    -------
    A warning that includes a list of metrics that were used, but
    not declared or declared, but not used. If only declared metrics
    where used, then this function has no output
    """
    onlyin_used_metrics = np.setdiff1d(used_metrics, necessary_metrics)
    if not onlyin_used_metrics:
        LGR.warning(functionname + " uses the following metrics that are not " +
                    "declared as necessary metrices: " + str(onlyin_used_metrics))

    onlyin_necessary_metrics = np.setdiff1d(necessary_metrics, used_metrics)
    if not onlyin_necessary_metrics:
        LGR.warning(functionname + " declared the following metrics as necessary " +
                    "but does not use them: " + str(onlyin_necessary_metrics))

# Functions that edit decision_tree_steps


def new_decision_node_info(decision_tree_steps, functionname,
                           metrics_used, iftrue, iffalse,
                           additionalparameters=None):
    """
    create a new node that logs steps in the decision tree
    """

    # ADD ERROR TESTING SO THAT A CRASH FROM A MISSING VARIABLE IS INTELLIGENTLY LOGGED
    tmp_decision_tree = {'nodeidx': None,
                         'functionname': functionname,
                         'metrics_used': metrics_used,
                         'iftrue': iftrue,
                         'iffalse': iffalse,
                         'additionalparameters': additionalparameters,
                         'report_extra_log': [],  # optionally defined by user
                         'numfalse': [],  # will be filled in at runtime
                         'numtrue': [],  # will be filled in at runtime
                         }

    if decision_tree_steps is None:
        decision_tree_steps = [tmp_decision_tree]
        decision_tree_steps['nodeidx'] = 1
    else:
        tmp_decision_tree['nodeidx'] = len(decision_tree_steps) + 1
        decision_tree_steps.append(tmp_decision_tree)

    return decision_tree_steps


def log_decision_tree_step(nodeidxstr, functionname, comps2use,
                           decide_comps=None, decision_tree_steps=None):
    """
        Logging text to add for every decision tree calculation
    """

    if comps2use is None:
        LGR.info("Step " + nodeidxstr + " " + functionname + " not applied because "
                 "no remaining components were classified as " + str(decide_comps))
    else:
        LGR.info("Step " + nodeidxstr + " " + functionname + " applied to "
                 + str((comps2use is True).sum()) + " components. "
                 + str(decision_tree_steps[-1]['numtrue']) + " were True and "
                 + str(decision_tree_steps[-1]['numfalse']) + " were False")

# Calculations that are used in decision tree functions


def getelbow_cons(arr, return_val=False):
    """
    Elbow using mean/variance method - conservative

    Parameters
    ----------
    arr : (C,) array_like
        Metric (e.g., Kappa or Rho) values.
    return_val : :obj:`bool`, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    :obj:`int` or :obj:`float`
        Either the elbow index (if return_val is True) or the values at the
        elbow index (if return_val is False)
    """
    if arr.ndim != 1:
        raise ValueError('Parameter arr should be 1d, not {0}d'.format(arr.ndim))
    arr = np.sort(arr)[::-1]
    nk = len(arr)
    temp1 = [(arr[nk - 5 - ii - 1] > arr[nk - 5 - ii:nk].mean() + 2 * arr[nk - 5 - ii:nk].std())
             for ii in range(nk - 5)]
    ds = np.array(temp1[::-1], dtype=np.int)
    dsum = []
    c_ = 0
    for d_ in ds:
        c_ = (c_ + d_) * d_
        dsum.append(c_)
    e2 = np.argmax(np.array(dsum))
    elind = np.max([getelbow(arr), e2])

    if return_val:
        return arr[elind]
    else:
        return elind


def getelbow(arr, return_val=False):
    """
    Elbow using linear projection method - moderate

    Parameters
    ----------
    arr : (C,) array_like
        Metric (e.g., Kappa or Rho) values.
    return_val : :obj:`bool`, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    :obj:`int` or :obj:`float`
        Either the elbow index (if return_val is True) or the values at the
        elbow index (if return_val is False)
    """
    if arr.ndim != 1:
        raise ValueError('Parameter arr should be 1d, not {0}d'.format(arr.ndim))
    arr = np.sort(arr)[::-1]
    n_components = arr.shape[0]
    coords = np.array([np.arange(n_components), arr])
    p = coords - coords[:, 0].reshape(2, 1)
    b = p[:, -1]
    b_hat = np.reshape(b / np.sqrt((b ** 2).sum()), (2, 1))
    proj_p_b = p - np.dot(b_hat.T, p) * np.tile(b_hat, (1, n_components))
    d = np.sqrt((proj_p_b ** 2).sum(axis=0))
    k_min_ind = d.argmax()

    if return_val:
        return arr[k_min_ind]
    else:
        return k_min_ind
