"""
Functions that will be used as steps in a decision tree
"""
import logging
import numpy as np
import pandas as pd
# from scipy import stats

from scipy.stats import scoreatpercentile
from tedana.stats import getfbounds
from tedana.selection._utils import (confirm_metrics_exist, selectcomps2use,
                                     log_decision_tree_step, change_comptable_classifications,
                                     getelbow)

# clean_dataframe, new_decision_node_info,
LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

decision_docs = {
    'comptable': """\
comptable : (C x M) :obj:`pandas.DataFrame`
    Component metric table. One row for each component, with a column for
    each metric. The index should be the component number.\
""",
    'decision_node_idx': """\
decison_node_idx : :obj: `int`
    The decision tree function are run as part of an ordered list.
    This is the positional index for when this function has been run
    as part of this list.\
""",
    'iftrue': """\
iftrue : :obj:`str`
    If the condition in this step is true, give the component
    the label in this string. Options are 'accept', 'reject',
    'provisionalaccept', 'provisionalreject', 'ignore', or 'nochange'
    If 'nochange' then don't change the current component classification\
""",
    'iffalse': """\
iffalse: :obj:`str`
    If the condition in this step is false, give the component the label
    in this string. Same options as iftrue\
""",
    'decide_comps': """\
decide_comps: :obj:`str` or :obj:`list[str]` or :obj:`int` or :obj:`list[int]`
    If this is string or a list of strings describing what classifications
    of components to operate on, using the same labels as in iftrue.
    For example: decide_comps='unclassified' means to operate only on
    unclassified components. The label 'all' will operate on all
    components regardess of classification.
    If this is an int or a list of int, operate on components with the
    listed integer indices. For example: [0 1 5] will operate on 3 components
    regardless of what their current classifications are.\
""",
    'log_extra': """\
log_extra_report, log_extra_info: :obj:`str`
    Text for each function call is automatically placed in the logger output
    In addition to that text, the text in these these strings will also be
    included in the logger with the report or info codes respectively.
    These might be useful to give a narrative explanation of why a step was
    parameterized a certain way. default="" (no extra logging)\
""",
    'only_used_metrics': """\
only_used_metrics: :obj:`bool`
    If true, this function will only return the names of the comptable metrics
    that will be used when this function is fully run. default=False\
""",
    'custom_node_label': """\
custom_node_label: :obj:`str`
    A brief label for what happens in this node that can be used in a decision
tree summary table or flow chart. If custom_node_label is not empty, then the
text in this parameter is used instead of the text would be automatically
assigned within the function call default=""\
""",
    'basicreturns': """\
comptable: (C x M) :obj:`pandas.DataFrame`
    Component metric table. One row for each component, with a column for
    each metric. The index should be the component number.
    Labels in the 'classifications' for components initially labeled in
    decide_comps may change depending on the the iftrue and iffalse instructions.
    When a classification changes, the 'rationale' column is appended to include
    and additional decision node index and change. For example, if this function
    is the 5th decision node run and a component is reclassified as 'ignore',
    the string in 'rationale' is appended to include '5: ignore;'
    comptable is only only returned if only_used_metrics=False
dnode_outputs: :obj:`dict`
    Several parameters that should be output from each decision node function
    in a dictionary under the key 'outputs' When a function is run as output
    as part of the decision tree class, this output will be added to the dictionary
    with parameters that called the function with all the outputs under the 'outputs'
    key. dnode_outputs includes the following fields\
    used_metrics: :obj: `list[str]`
        A list of all metrics from the comptable header used within this function.
        Note, this must be a list even if only one metric is used
    node_label: :obj: `str`
        A brief label for what happens in this node that can be used in a decision
    tree summary table or flow chart. This is defined in the function unless
    custom_node_label is not empty. In that case, node_label=custom_node_label
    numTrue, numFalse: :obj: `int`
        The number of components that were classified as true or false respectively
    in this decision tree step.\
""",
    'n_echos': """\
n_echos: :obj:`int`
    The number of echos in the multi-echo data
        \
""",
    'n_vols': """\
n_vols: :obj:`int`
    The number of volumes (time points) in the fMRI data
        \
""",
    'kappa_elbow': """\
kappa_elbow: :obj:`float`
    The kappa threshold below which components should be rejected or ignored
        \
""",
    'rho_elbow': """\
rho_elbow: :obj:`float`
    The rho threshold above which components should be rejected or ignored
        \
"""
}


def create_dnode_outputs(used_metrics, node_label, numTrue, numFalse,
                         n_echos=None, kappa_elbow=None, rho_elbow=None,
                         num_prov_accept=None, max_good_meanmetricrank=None,
                         varex_threshold=None
                         ):
    """
    Take several parameters that should be output from each decision node function
    and put them in a dictionary under the key 'outputs' When the decision is output
    as part of the decision tree class, this will be added to the dictionary with
    parameters that called the function with all the outputs under the 'outputs' key

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
    if kappa_elbow:
        dnode_outputs['outputs'].update({'kappa_elbow': kappa_elbow})
    if rho_elbow:
        dnode_outputs['outputs'].update({'rho_elbow': rho_elbow})

    return dnode_outputs


def manual_classify(comptable, decision_node_idx,
                    decide_comps, new_classification,
                    clear_rationale=False,
                    log_extra_report="", log_extra_info="",
                    custom_node_label="", only_used_metrics=False):
    """
    Explicitly assign a classifictation, defined in iffrue,
    to all the components in decide_comps. This was designed
    with three use cases in mind:
    1. Set the classifictions of all components to unclassified
    for the first node of a decision tree.
    clear_rationale=True is recommended for this use case
    2. Shift all components between classifications, such as
    provisionalaccept to accept for the penultimate node in the
    decision tree.
    3. Manually re-classify components by number based on user
    observations.

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {decide_comps}
    new_classification: :obj: `str`
        Assign all components identified in decide_comps the classification
        in new_classification. Options are 'accept', 'reject',
        'provisionalaccept', 'provisionalreject', or 'ignore'
    clear_rationale: :obj: `bool`
        If True, reset all values in the 'rationale' column to empty strings
        If False, do nothing
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}

    Note
    ----
    Unlike other decision node functions, iftrue and iffalse are not inputs
    since the same classification is assigned to all components listed in
    decide_comps
    """
    used_metrics = []
    if only_used_metrics:
        return used_metrics

    iftrue = new_classification
    iffalse = 'nochange'

    function_name_idx = ("manual_classify, step " + str(decision_node_idx))
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = ("Set " + str(decide_comps) + " to " + new_classification)

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use = selectcomps2use(comptable, decide_comps)

    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        numTrue = 0
        numFalse = 0
    else:
        decision_boolean = pd.Series(True, index=comps2use)
        comptable = change_comptable_classifications(
                        comptable, iftrue, iffalse,
                        decision_boolean, str(decision_node_idx))
        numTrue = decision_boolean.sum()
        numFalse = np.logical_not(decision_boolean).sum()
        print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
            numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(function_name_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

    if clear_rationale:
        comptable['rationale'] = ""
        LGR.info(function_name_idx + " all 'rationale' values are set to empty strings")

    dnode_outputs = create_dnode_outputs(used_metrics, node_label, numTrue, numFalse)

    return comptable, dnode_outputs


manual_classify.__doc__ = manual_classify.__doc__.format(**decision_docs)


def metric1_greaterthan_metric2(comptable, decision_node_idx, iftrue, iffalse,
                                decide_comps, metric1, metric2, metric2_scale=1,
                                log_extra_report="", log_extra_info="",
                                custom_node_label="", only_used_metrics=False):
    """
    Is metric1 > (metric2_scale*metric2)?
    This can be used to directly compare any 2 metrics and use that info
    to change component classification. If either metric is an number,
    this can also compare a metric again a fixed threshold.

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {iftrue}
    {iffalse}
    {decide_comps}
    metric1, metric2: :obj:`str` or :obj:`float`
        The labels for the two metrics to be used for comparision.
        for example, metric1='rho' and metric2='kappa' means this
        function will test rho>kappa. One of the two can also be a number.
        In that case a metric would be compared against a fixed threshold.
        For example metric1=0 and metric2='T2fitdiff_invsout_ICAmap_Tstat'
        means this function will test 0>T2fitdiff_invsout_ICAmap_Tstat
    metric2_scale: :obj:`float`, optional
            Multiple a metric2's value by metric2_scale before testing if it
            is greater than the metric1 value. default=1
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    """

    used_metrics = []
    if isinstance(metric1, str):
        used_metrics.append(metric1)
    if isinstance(metric2, str):
        used_metrics.append(metric2)
    if only_used_metrics:
        return used_metrics

    function_name_idx = ("metric1_greaterthan_metric2, step " + str(decision_node_idx))
    if custom_node_label:
        node_label = custom_node_label
    elif metric2_scale == 1:
        node_label = (str(metric1) + ">" + str(metric2))
    else:
        node_label = (str(metric1) + ">(" + str(metric2_scale) + "*" + str(metric2) + ")")

    # Might want to add additional default logging to functions here
    # The function input will be logged before the function call
    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    confirm_metrics_exist(comptable, used_metrics, function_name=function_name_idx)

    # decision_tree_steps = new_decision_node_info(decision_tree_steps, function_name,
    #                                             necessary_metrics, iftrue, iffalse,
    #                                             additionalparameters=None)
    # nodeidxstr = str(decision_tree_steps[-1]['nodeidx'])

    comps2use = selectcomps2use(comptable, decide_comps)

    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        numTrue = 0
        numFalse = 0
    else:
        if isinstance(metric1, str):
            val1 = comptable.loc[comps2use, metric1]
        else:
            val1 = metric1  # should be a fixed number
        if isinstance(metric2, str):
            val2 = comptable.loc[comps2use, metric2]
        else:
            val2 = metric2  # should be a fixed number
        decision_boolean = val1 > (metric2_scale * val2)

        comptable = change_comptable_classifications(
                        comptable, iftrue, iffalse,
                        decision_boolean, str(decision_node_idx))
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
            numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(function_name_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

    dnode_outputs = create_dnode_outputs(used_metrics, node_label, numTrue, numFalse)

    return comptable, dnode_outputs


metric1_greaterthan_metric2.__doc__ = metric1_greaterthan_metric2.__doc__.format(**decision_docs)


def classification_exists(comptable, decision_node_idx, iftrue, iffalse,
                          decide_comps, class_comp_exists,
                          log_extra_report="", log_extra_info="",
                          custom_node_label="", only_used_metrics=False):
    """
    If there are not compontents with a classification specified in class_comp_exists,
    change the classification of all components in decide_comps
    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {iftrue}
    {iffalse}
    {decide_comps}
    class_comp_exists: :obj:`str` or :obj:`list[str]` or :obj:`int` or :obj:`list[int]`
        This has the same structure options as decide_comps. This function tests
        whether any components have the classifications defined in this variable.
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}

    """

    used_metrics = []
    if only_used_metrics:
        return used_metrics

    function_name_idx = ("classification_exists, step " + str(decision_node_idx))
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = "Change {} if {} doesn't exist".format(decide_comps, classification_exists)

    # Might want to add additional default logging to functions here
    # The function input will be logged before the function call
    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    comps2use = selectcomps2use(comptable, decide_comps)
    do_comps_exist = selectcomps2use(comptable, class_comp_exists)

    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        numTrue = 0
        numFalse = 0
    elif do_comps_exist is None:
        # should be false for all components
        decision_boolean = comptable.loc[comps2use, 'component'] < -100
        comptable = change_comptable_classifications(
                        comptable, iftrue, iffalse,
                        decision_boolean, str(decision_node_idx))
        numTrue = np.asarray(decision_boolean).sum()
        # numtrue should always be 0 in this situation
        numFalse = np.logical_not(decision_boolean).sum()
        print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
            numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(function_name_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)
    else:
        numTrue = len(comps2use)
        numFalse = 0
        log_decision_tree_step(function_name_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

    dnode_outputs = create_dnode_outputs(used_metrics, node_label, numTrue, numFalse)

    return comptable, dnode_outputs


def meanmetricrank_and_variance_greaterthan_thresh(comptable, decision_node_idx, iftrue, iffalse,
                                                   decide_comps, n_vols,
                                                   high_perc=90,
                                                   log_extra_report="", log_extra_info="",
                                                   custom_node_label="", only_used_metrics=False):
    """
    The 'mean metric rank' (formerly d_table) is the mean of rankings of 5 metrics:
        'kappa', 'DICE_FT2', 'T2fitdiff invsout ICAmap Tstat',
        and 'countnoise', 'countsig in T2clusters'
    For these 5 metrics, a lower rank (smaller number) is less likely to be
    T2* weighted.
    This function tests of meanmetricrank is above a threshold based on the number
    of provisionally accepted components & variance based on a threshold related
    to the variance of provisionally accepted components. This is indented to
    reject components that are greater than both of these thresholds

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {iftrue}
    {iffalse}
    {decide_comps}
    HIGH_PERC: :obj:`int`
        A percentile threshold to apply to components to set the variance
        threshold. default=90
    n_vols: :obj:`int`
        The number of volumes in the fMRI time series.
        Used to calculate the threshold for meanmetricrank
        In the MEICA code, this was hard-coded to 2 for data with more
        than 100 volumes and 3 for data with less than 100 volumes.
        default=2. Now is linearly ramped from 2-3 for vols between 90 & 110
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    dnode_ouputs also contains:
    num_prov_accept: :obj:`int`
        Number of provisionally accepted components
    max_good_meanmetricrank: :obj:`float`
        The threshold used meanmetricrank
    varex_threshold: :obj:`float`
        The threshold used for variance
    """

    used_metrics = ['mean metric rank', 'variance explained']
    if only_used_metrics:
        return used_metrics

    function_name_idx = (
        "meanmetricrank_and_variance_greaterthan_thresh, step " + str(decision_node_idx))
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = ('MeanRank & Variance Thresholding')

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    metrics_exist, missing_metrics = confirm_metrics_exist(
        comptable, used_metrics, function_name=function_name_idx)

    comps2use = selectcomps2use(comptable, decide_comps)
    provaccept_comps2use = selectcomps2use(comptable, ['provisionalaccept'])
    if (comps2use is None) or (provaccept_comps2use is None):
        if comps2use is None:
            log_decision_tree_step(function_name_idx, comps2use,
                                   decide_comps=decide_comps)
        if provaccept_comps2use is None:
            log_decision_tree_step(function_name_idx, comps2use,
                                   decide_comps='provisionalaccept')
        dnode_outputs = create_dnode_outputs(used_metrics, node_label, 0, 0)
    else:
        num_prov_accept = len(provaccept_comps2use)
        varex_threshold = scoreatpercentile(
            comptable.loc[provaccept_comps2use, 'variance explained'], high_perc)

        if n_vols < 90:
            extend_factor = 3
        elif n_vols < 110:
            extend_factor = 2 + (n_vols - 90) / 20
        else:
            extend_factor = 2
        max_good_meanmetricrank = extend_factor * num_prov_accept

        decision_boolean1 = comptable.loc[comps2use, 'mean metric rank'] > max_good_meanmetricrank
        decision_boolean2 = comptable.loc[comps2use, 'variance explained'] > varex_threshold
        decision_boolean = decision_boolean1 & decision_boolean2

        comptable = change_comptable_classifications(
                        comptable, iftrue, iffalse,
                        decision_boolean, str(decision_node_idx))
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
            numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(function_name_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

        dnode_outputs = create_dnode_outputs(used_metrics, node_label, numTrue, numFalse,
                                             num_prov_accept=num_prov_accept,
                                             varex_threshold=varex_threshold,
                                             max_good_meanmetricrank=max_good_meanmetricrank)

    return comptable, dnode_outputs


meanmetricrank_and_variance_greaterthan_thresh.__doc__ = (
    meanmetricrank_and_variance_greaterthan_thresh.__doc__.format(**decision_docs))


def variance_lessthan_thresholds(comptable, decision_node_idx, iftrue, iffalse,
                                 decide_comps, var_metric='varexp',
                                 single_comp_threshold=0.1,
                                 all_comp_threshold=1.0,
                                 log_extra_report="", log_extra_info="",
                                 custom_node_label="", only_used_metrics=False):
    """
    Finds componentes with variance<single_comp_threshold.
    If the sum of the variance for all components that meet this criteria
    is greater than all_comp_threshold then keep the lowest variance
    components so that the sum of their variances is less than all_comp_threshold

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {iftrue}
    {iffalse}
    {decide_comps}
    varmetric: :obj:`str`
        The name of the metric in comptable for variance. default=varexp
        This is an option so that it is possible to set this to normvarexp
        or some other variance measure
    single_comp_threshold: :obj:`float`
        The threshold for which all components need to have lower variance
    all_comp_threshold: :obj: `float`
        The threshold for which the sum of all components<single_comp_threshold
        needs to be under
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    """

    used_metrics = [var_metric]
    if only_used_metrics:
        return used_metrics

    function_name_idx = ("variance_lt_thresholds, step " + str(decision_node_idx))
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = ('{}<{}. All variance<{}').format(
            used_metrics, single_comp_threshold, all_comp_threshold)

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)
    metrics_exist, missing_metrics = confirm_metrics_exist(
        comptable, used_metrics, function_name=function_name_idx)

    comps2use = selectcomps2use(comptable, decide_comps)
    if comps2use is None:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        numTrue = 0
        numFalse = 0
    else:
        variance = comptable.loc[comps2use, var_metric]
        decision_boolean = variance < single_comp_threshold
        # if all the low variance components sum above all_comp_threshold
        # keep removing the highest remaining variance component until
        # the sum is below all_comp_threshold. This is an inefficient
        # way to do this, but it works & should never cause an infinite loop
        if variance[decision_boolean].sum() > all_comp_threshold:
            while variance[decision_boolean].sum() > all_comp_threshold:
                cutcomp = variance[decision_boolean].idxmax
                decision_boolean[cutcomp] = False
        comptable = change_comptable_classifications(
                        comptable, iftrue, iffalse,
                        decision_boolean, str(decision_node_idx))
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
            numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(function_name_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

    dnode_outputs = create_dnode_outputs(used_metrics, node_label, numTrue, numFalse)

    return comptable, dnode_outputs


variance_lessthan_thresholds.__doc__ = variance_lessthan_thresholds.__doc__.format(**decision_docs)


def kappa_rho_elbow_cutoffs_kundu(comptable, decision_node_idx, iftrue, iffalse,
                                  decide_comps, n_echos,
                                  log_extra_report="", log_extra_info="",
                                  custom_node_label="", only_used_metrics=False):
    """
    Calculates 'elbows' for kappa and rho values across compnents and thresholds
    on kappa>kappa_elbow & rho<rho_elbow

    Parameters
    ----------
    {comptable}
    {decision_node_idx}
    {iftrue}
    {iffalse}
    {decide_comps}
    {n_echos}
    {log_extra}
    {custom_node_label}
    {only_used_metrics}

    Returns
    -------
    {basicreturns}
    dnode_ouputs also contains:
    {n_echos}
    {kappa_elbow}
    {rho_elbow}

    Note
    ----
    This script is currently hard coded for a specific way to calculate kappa and rho elbows
    based on the method by Kundu in the MEICA v2.7 code. Another elbow calculation would
    require a distinct function. Ideally, there can be one elbow function can allows for
    some more flexible options
    """

    used_metrics = ['kappa', 'rho']
    if only_used_metrics:
        return used_metrics

    function_name_idx = ("kappa_rho_elbow_cutoffs_kundu, step " + str(decision_node_idx))
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = ('Kappa&Rho Elbow Thresholding')

    LGR.info("Note: This matches the elbow selecton criteria in Kundu's MEICA v2.7"
             " except there is a variance threshold that is used for the rho criteria that "
             "really didn't make sense and is being excluded.")

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        RepLGR.info(log_extra_report)

    metrics_exist, missing_metrics = confirm_metrics_exist(
        comptable, used_metrics, function_name=function_name_idx)

    comps2use = selectcomps2use(comptable, decide_comps)
    unclassified_comps2use = selectcomps2use(comptable, 'unclassified')

    if (comps2use is None) or (unclassified_comps2use is None):
        if comps2use is None:
            log_decision_tree_step(function_name_idx, comps2use,
                                   decide_comps=decide_comps)
        if unclassified_comps2use is None:
            log_decision_tree_step(function_name_idx, comps2use,
                                   decide_comps='unclassified')
        numTrue = 0
        numFalse = 0
    else:

        f05, _, f01 = getfbounds(n_echos)
        # get kappa values for components below a significance threshold
        kappas_nonsig = comptable.loc[comptable['kappa'] < f01, 'kappa']

        # Would an elbow from all Kappa values *ever* be lower than one from
        # a subset of lower values?
        kappa_elbow = np.min((getelbow(kappas_nonsig, return_val=True),
                              getelbow(comptable['kappa'], return_val=True)))

        # The first elbow used to be for rho values of the unclassified components
        # excluding a few based on differences of variance. Now it's all unclassified
        # components
        # Upper limit for variance explained is median across components with high
        # Kappa values. High Kappa is defined as Kappa above Kappa elbow.
        varex_upper_p = np.median(
            comptable.loc[comptable['kappa'] > getelbow(comptable['kappa'], return_val=True),
                          'variance explained'])
        temp_comptable = comptable.loc[unclassified_comps2use].sort_values(
            by=['variance explained'], ascending=False)
        diff_vals = temp_comptable['variance explained'].diff(-1)
        top_five = diff_vals[:5]
        bad_from_top_five = top_five.loc[top_five > varex_upper_p]
        idx = bad_from_top_five.index[-1]
        comps_to_drop = top_five.loc[:idx].index.values
        comps_to_keep = list(set(unclassified_comps2use) - set(comps_to_drop))

        rho_elbow = np.mean((getelbow(comptable.loc[comps_to_keep, 'rho'], return_val=True),
                             getelbow(comptable['rho'], return_val=True),
                             f05))

        decision_boolean = (
                comptable.loc[comps2use, 'kappa'] >= kappa_elbow) & (
                comptable.loc[comps2use, 'rho'] < rho_elbow)

        comptable = change_comptable_classifications(
                            comptable, iftrue, iffalse,
                            decision_boolean, str(decision_node_idx))
        numTrue = np.asarray(decision_boolean).sum()
        numFalse = np.logical_not(decision_boolean).sum()
        print(('numTrue={}, numFalse={}, numcomps2use={}'.format(
                numTrue, numFalse, len(comps2use))))
        log_decision_tree_step(function_name_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

    dnode_outputs = create_dnode_outputs(used_metrics, node_label, numTrue, numFalse,
                                         n_echos=n_echos,
                                         kappa_elbow=kappa_elbow, rho_elbow=rho_elbow)

    return comptable, dnode_outputs


kappa_rho_elbow_cutoffs_kundu.__doc__ = kappa_rho_elbow_cutoffs_kundu.__doc__.format(
    **decision_docs)
