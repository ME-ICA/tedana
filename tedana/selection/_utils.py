"""
Utility functions for tedana.selection
"""

import logging
import re
import numpy as np
from tedana.stats import getfbounds
from tedana.metrics.dependence import generate_decision_table_score

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

# Functions that are used for interacting with comptable


def selectcomps2use(comptable, decide_comps):
    """
    Give a list of components that fit a classification types
    Since 'all' converts string to boolean, it will miss components with
    no classification. This means, in the initialization of comptree, all
    components need to be labeled as unclassified, NOT empty
    WILL NEED TO ADD NUMBER INDEXING TO selectcomps2use
    """
    if type(decide_comps) == str:
        decide_comps = [decide_comps]
    if decide_comps[0] == 'all':
        # All components with any string in the classification field
        # are set to True
        comps2use = list(range(comptable.shape[0]))
    # elif (type(decide_comps) == str):
    #    comps2use = comptable.index[comptable['classification'] == decide_comps].tolist()
    elif (type(decide_comps) == list) and (type(decide_comps[0]) == str):
        comps2use = []
        for didx in range(len(decide_comps)):
            newcomps2use = comptable.index[
                comptable['classification'] == decide_comps[didx]].tolist()
            comps2use = list(set(comps2use + newcomps2use))
    else:
        # decide_comps is already a string of indices
        comps2use = decide_comps

    # If no components are selected, then return None.
    # The function that called this can check for None and exit before
    # attempting any computations on no data
    if not comps2use:
        comps2use = None

    return comps2use


def change_comptable_classifications(comptable, iftrue, iffalse,
                                     decision_boolean, decision_node_idx_str):
    """
    Given information on whether a decision critereon is true or false for each component
    change or don't change the component classification

    Note
    ----
    May want to add a check here so that, if a component classification is changed from
    accept, reject, or ignore, to something else, throw a warning. A user would have the power
    to change component labels in any order, but the ideal is that once something is assigned
    as accept, reject, or ignore, those are final classifications that should not be changed.
    If this is added, then there should be an option to override the warning. That override
    would be necessary when manual_classify is used to remove all classification info at the
    start of a decision tree. It also might be useful to have an override for ignore.
    The use case for this would be, if the total explained variance of all the ignored components
    is above a threshold (i.e >5% of accepted explained variance) then move the highest variance
    ignored components with rho/kappa>a threshold form ignore to reject
    """
    print(('iftrue={}, iffalse={}, decision_node_idx_str{}').format(
        iftrue, iffalse, decision_node_idx_str))
    if iftrue != 'nochange':
        changeidx = decision_boolean.index[np.asarray(decision_boolean)]
        comptable.loc[changeidx, 'classification'] = iftrue
        comptable.loc[changeidx, 'rationale'] += (decision_node_idx_str + ': ' + iftrue + '; ')
    if iffalse != 'nochange':
        changeidx = decision_boolean.index[~np.asarray(decision_boolean)]
        comptable.loc[changeidx, 'classification'] = iffalse
        comptable.loc[changeidx, 'rationale'] += (decision_node_idx_str + ': ' + iffalse + '; ')

    # decision_tree_steps[-1]['numtrue'] = (decision_boolean is True).sum()
    # decision_tree_steps[-1]['numfalse'] = (decision_boolean is False).sum()

    return comptable  # , decision_tree_steps


def clean_dataframe(comptable):
    """
    Reorder columns in component table so "rationale" and "classification" are
    last and remove trailing semicolons from rationale column.
    """
    cols_at_end = ['classification', 'rationale']
    comptable = (
        comptable[[c for c in comptable if c not in cols_at_end] + [c for c in cols_at_end
                                                                    if c in comptable]])
    comptable['rationale'] = comptable['rationale'].str.rstrip(';')
    return comptable

# Functions that validate inputted parameters or other processing steps


def confirm_metrics_exist(comptable, necessary_metrics, function_name=None):
    """
    Confirm that all metrics declared in necessary_metrics are
    already included in comptable.

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
            Component metric table. One row for each component, with a column for
            each metric. The index should be the component number.
    necessary_metrics : :obj:`list` a 1D list of strings of metrics
    function_name : :obj:`str`
        Text identifying the function name that called this function

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

    missing_metrics = set(necessary_metrics) - set(comptable.columns)
    metrics_exist = len(missing_metrics) == 0

    if metrics_exist is False:
        if function_name is not None:
            error_msg = ("Necessary metrics for " + function_name + " are not in comptable. "
                         "Need to calculate the following metrics: " + str(missing_metrics))
        else:
            error_msg = ("Necessary metrics are not in comptable (calling function unknown). "
                         "Need to calculate the following metrics: " + str(missing_metrics))

        raise ValueError(error_msg)

    return metrics_exist, missing_metrics


# def are_only_necessary_metrics_used(used_metrics, necessary_metrics, function_name):
#     """
#     Checks if all metrics that are declared as necessary are actually used and
#     if any used_metrics weren't explicitly declared. If either of these happen,
#     a warning is added to the logger that notes which metrics weren't declared
#     or used.

#     Parameters
#     ----------
#     used_metrics: :obj:`list`
#             A list of strings of the metric names that were used in the
#             decision tree
#     necessary_metrics: :obj:`list`
#             A list of strings of the metric names that were declared
#             to be used in the decision tree
#     function_name: :obj:`str`
#             The function name for the decision tree that was run

#     Returns
#     -------
#     A warning that includes a list of metrics that were used, but
#     not declared or declared, but not used. If only declared metrics
#     where used, then this function has no output
#     """
#     onlyin_used_metrics = np.setdiff1d(used_metrics, necessary_metrics)
#     if not onlyin_used_metrics:
#         LGR.warning(function_name + " uses the following metrics that are not "
#                     "declared as necessary metrices: " + str(onlyin_used_metrics))

#     onlyin_necessary_metrics = np.setdiff1d(necessary_metrics, used_metrics)
#     if not onlyin_necessary_metrics:
#         LGR.warning(function_name + " declared the following metrics as necessary "
#                     "but does not use them: " + str(onlyin_necessary_metrics))

# Functions that edit decision_tree_steps


def new_decision_node_info(decision_tree_steps, function_name,
                           metrics_used, iftrue, iffalse,
                           additionalparameters=None):
    """
    create a new node that logs steps in the decision tree
    """

    # ADD ERROR TESTING SO THAT A CRASH FROM A MISSING VARIABLE IS INTELLIGENTLY LOGGED
    tmp_decision_tree = {'nodeidx': None,
                         'function_name': function_name,
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


def log_decision_tree_step(function_name_idx, comps2use,
                           decide_comps=None,
                           numTrue=None, numFalse=None):
    """
        Logging text to add for every decision tree calculation
    """

    if comps2use is None:
        LGR.info(function_name_idx + " not applied because "
                 "no remaining components were classified as " + str(decide_comps))
    else:
        LGR.info((function_name_idx + "applied to " + str(np.array(comps2use).sum()) + " "
                  "components. " + str(numTrue) + " were True "
                  "and " + str(numFalse) + "were False"))
        # decision_tree_steps[-1]['numtrue']) + " "
        # "were True and " + str(decision_tree_steps[-1]['numfalse']) + " were False"))


def create_dnode_outputs(used_metrics, node_label, numTrue, numFalse,
                         n_echos=None, n_vols=None, kappa_elbow=None, rho_elbow=None,
                         num_prov_accept=None, max_good_meanmetricrank=None,
                         varex_threshold=None, low_perc=None, high_perc=None,
                         extend_factor=None, restrict_factor=None,
                         ignore_prex_X_steps=None, num_acc_guess=None
                         ):
    """
    Take several parameters that should be output from each decision node function
    and put them in a dictionary under the key 'outputs' When the decision is output
    as part of the decision tree class, this will be added to the dictionary with
    parameters that called the function with all the outputs under the 'outputs' key.

    Parameters
    ----------
    Required parameters
    used_metrics: :obj: `list[str]`
        A list of all metrics from the comptable header used within this function.
        Note, this must be a list even if only one metric is used
    node_label: :obj: `str`
        A brief label for what happens in this node that can be used in a decision
        tree summary table or flow chart.
    numTrue, numFalse: :obj: `int`
        The number of components that were classified as true or false respectively
    in this decision tree step.

    Optional parameters that are ignored, if None
    n_echos: :obj:`int`
        The number of echos in the multi-echo data
    n_vols: :obj:`int`
        The number of volumes (time points) in the fMRI data
    kappa_elbow: :obj:`float`
        The kappa threshold below which components should be rejected or ignored
    rho_elbow: :obj:`float`
        The rho threshold above which components should be rejected or ignored


    Returns
    -------
    dnode_outputs: :obj:`dict`
        A dict that contains the inputted parameters that are not 'None'
    """

    dnode_outputs = {'outputs': {
        'used_metrics': used_metrics,
        'node_label': node_label,
        'numTrue': numTrue,
        'numFalse': numFalse
    }}
    if n_echos:
        dnode_outputs['outputs'].update({'n_echos': n_echos})
    if n_vols:
        dnode_outputs['outputs'].update({'n_vols': n_vols})
    if kappa_elbow:
        dnode_outputs['outputs'].update({'kappa_elbow': kappa_elbow})
    if rho_elbow:
        dnode_outputs['outputs'].update({'rho_elbow': rho_elbow})
    if high_perc:
        dnode_outputs['outputs'].update({'high_perc': high_perc})
    if low_perc:
        dnode_outputs['outputs'].update({'low_perc': low_perc})
    if max_good_meanmetricrank:
        dnode_outputs['outputs'].update({'max_good_meanmetricrank': max_good_meanmetricrank})
    if varex_threshold:
        dnode_outputs['outputs'].update({'varex_threshold': varex_threshold})
    if num_prov_accept:
        dnode_outputs['outputs'].update({'num_prov_accept': num_prov_accept})
    if restrict_factor:
        dnode_outputs['outputs'].update({'restrict_factor': num_prov_accept})
    if ignore_prex_X_steps:
        dnode_outputs['outputs'].update({'ignore_prex_X_steps': num_prov_accept})
    if num_acc_guess:
        dnode_outputs['outputs'].update({'num_acc_guess': num_prov_accept})

    return dnode_outputs

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


def kappa_elbow_kundu(comptable, n_echos):
    """
    Calculate an elbow for kappa using the approach originally in
    Prantik Kundu's MEICA v2.7 code

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
        Only the 'kappa' column is used in this function
    n_echos: :obj:`int`
        The number of echos in the multi-echo data

    Returns
    -------
    kappa_elbow: :obj:`float`
        The 'elbow' value for kappa values, above which components are considered
        more likely to contain T2* weighted signals
    """
    # low kappa threshold
    f05, _, f01 = getfbounds(n_echos)
    # get kappa values for components below a significance threshold
    kappas_nonsig = comptable.loc[comptable['kappa'] < f01, 'kappa']

    # Would an elbow from all Kappa values *ever* be lower than one from
    # a subset of lower values?
    kappa_elbow = np.min((getelbow(kappas_nonsig, return_val=True),
                          getelbow(comptable['kappa'], return_val=True)))

    return kappa_elbow


def get_extend_factor(n_vols=None, extend_factor=None):
    """
    extend_factor is a scaler used to set a threshold for the mean metric rank
    It is either defined by the number of volumes in the time series or if directly
    defined by the user. If it is defined by the user, that takes precedence over
    using the number of volumes in a calculation

    Parameters
    ----------
    n_vols: :obj:`int`
        The number of volumes in an fMRI time series. default=None
        In the MEICA code, extend_factor was hard-coded to 2 for data with more
        than 100 volumes and 3 for data with less than 100 volumes.
        Now is linearly ramped from 2-3 for vols between 90 & 110

    extend_factor: :obj:`float`
        The scaler used to set a threshold for mean metric rank. default=None

    Returns
    -------
    extend_factor: :obj:`float`

    Note
    ----
    Either n_vols OR extend_factor is a required input
    """

    if extend_factor:
        LGR.info('extend_factor={}, as defined by user'.format(extend_factor))
    elif n_vols:
        if n_vols < 90:
            extend_factor = 3
        elif n_vols < 110:
            extend_factor = 2 + (n_vols - 90) / 20
        else:
            extend_factor = 2
        LGR.info('extend_factor={}, based on number of fMRI volumes'.format(extend_factor))
    else:
        error_msg = 'get_extend_factor need n_vols or extend_factor as an input'
        LGR.error(error_msg)
        ValueError(error_msg)
    return extend_factor


def get_new_meanmetricrank(comptable, comps2use, decision_node_idx,
                           calc_new_rank=False):
    """
    If a revised mean metric rank was already calculated, use that.
    If not, calculate a new mean metric rank based on the components
    identified in comps2use

    Parameters
    ----------
    comptable
    comps2use
    decision_node_idx: :obj:`int`
        The index for the current decision node
    calc_new_rank: :obj:`bool`
        calculate a new mean metric rank even if a revised mean
        metric rank was already calculated

    Return
    ------
    comptable
    meanmetricrank
    """

    if not calc_new_rank:
        # check to see if a revised mean metric rank was already calculated
        # and use that rank. A revised mean metric rank would be named
        # 'mean metric rank' followed by a number that is lower than the
        # current decision_node_idx
        for didx in range(decision_node_idx, -1, -1):
            if ('mean metric rank ' + str(didx)) in comptable.columns:
                return comptable['mean metric rank ' + str(didx)], comptable
        else:
            comptable['mean metric rank ' + str(decision_node_idx)] = (
                generate_decision_table_score(
                    comptable.loc[comps2use, 'kappa'],
                    comptable.loc[comps2use, 'dice_FT2'],
                    comptable.loc[comps2use, 'signal_minus_noise_t'],
                    comptable.loc[comps2use, 'countnoise'],
                    comptable.loc[comps2use, 'countsigFT2']))
            return comptable['mean metric rank ' + str(decision_node_idx)], comptable


def prev_classified_comps(comptable, decision_node_idx, classification_label, prev_X_steps=0):
    """
    Output a list of components with a specific label during the current or
    previous X steps of the decision tree. For example, if
    classification_label = ['provisionalaccept'] and prev_X_steps = 0
    then this outputs the indices of components that are currenlty
    classsified as provisionalaccept. If prev_X_steps=2, then this will
    output components that are classified as provisionalaccept or were
    classified as such any time before the previous two decision tree steps

    Parameters
    ----------
    comptable
    n_echos: :obj:`int`
        The number of echos in the multi-echo data set
    decision_node_idx: :obj:`int`
        The index of the node in the decision tree that called this function
    classification_label: :obj:`list[str]`
        A list of strings containing classification labels to identify in components
        For example: ['provisionalaccept']
    prev_X_steps: :obj:`int`
        If 0, then just count the number of provisionally accepted or rejected
        or unclassified components in the current node. If this is a positive
        integer, then also check if a component was a in one of those three
        categories in ignore_prev_X_steps previous nodes. default=0

    Returns
    -------
    full_comps2use: :obj:`list[int]`
        A list of indices of components that have or add classification_lable
    """

    full_comps2use = selectcomps2use(comptable, classification_label)
    rationales = comptable['rationale']

    if prev_X_steps > 0:  # if checking classifications in prevision nodes
        for compidx in range(len(comptable)):
            tmp_rationale = rationales.values[compidx]
            tmp_list = re.split(':|;| ', tmp_rationale)
            while("" in tmp_list):  # remove blank strings after splitting rationale
                tmp_list.remove("")
            # Check the previous nodes
            # This is inefficient, but it should work
            for didx in range(max(0, decision_node_idx - prev_X_steps), decision_node_idx):
                if str(didx) in tmp_list:
                    didx_loc = tmp_list.index(str(didx))
                    if(didx_loc > 1):
                        tmp_classifier = (tmp_list[didx_loc - 1])
                        if tmp_classifier in classification_label:
                            full_comps2use.append(compidx)

    full_comps2use = list(set(full_comps2use))

    return full_comps2use
