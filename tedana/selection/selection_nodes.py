"""
Functions that will be used as steps in a decision tree
"""
import logging
# import numpy as np
# from scipy import stats

# from tedana.stats import getfbounds
from tedana.selection._utils import (confirm_metrics_calculated,
                                     selectcomps2use,
                                     log_decision_tree_step, change_comptable_classifications)
# getelbow, clean_dataframe, new_decision_node_info,
LGR = logging.getLogger(__name__)

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
used_metrics: :obj: `str` or :obj: `list[str]`
    A single metric name or a list of all metrics used within this function
node_label: :obj: `str`
    A brief label for what happens in this node that can be used in a decision
tree summary table or flow chart. This is defined in the function unless
custom_node_label is not empty. In that case, node_label=custom_node_label
numTrue, numFalse: :obj: `int`
    The number of components that were classified as true or false respectively
in this decision tree step.\
"""
}


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

    functionname_idx = ("manual_classify, step " + str(decision_node_idx))
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = ("Set " + str(decide_comps) + " to " + new_classification)

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        LGR.report(log_extra_report)

    comps2use = selectcomps2use(comptable, decide_comps)

    if comps2use is None:
        log_decision_tree_step(functionname_idx, comps2use)
    else:
        comptable = change_comptable_classifications(
                        comptable, iftrue, iffalse,
                        comps2use, str(decision_node_idx))
        numTrue = (comps2use is True).sum()
        numFalse = (comps2use is False).sum()
        log_decision_tree_step(functionname_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

    if clear_rationale:
        comptable['rationale'] = ""
        LGR.info(functionname_idx + " all 'rationale' values are set to empty strings")
    return comptable, used_metrics, node_label, numTrue, numFalse


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

    functionname_idx = ("metric1_greaterthan_metric2, step " + str(decision_node_idx))
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
        LGR.report(log_extra_report)

    metrics_exist, missing_metrics = confirm_metrics_calculated(comptable, used_metrics)
    if metrics_exist is False:
        error_msg = ("Necessary metrics for " + functionname_idx + "are not in comptable. "
                     "Need to calculate the following metrics: " + missing_metrics)
        raise ValueError(error_msg)

    # decision_tree_steps = new_decision_node_info(decision_tree_steps, functionname,
    #                                             necessary_metrics, iftrue, iffalse,
    #                                             additionalparameters=None)
    # nodeidxstr = str(decision_tree_steps[-1]['nodeidx'])

    comps2use = selectcomps2use(comptable, decide_comps)

    if comps2use is None:
        log_decision_tree_step(functionname_idx, comps2use)
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
        numTrue = (comps2use is True).sum()
        numFalse = (comps2use is False).sum()
        log_decision_tree_step(functionname_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

    return comptable, used_metrics, node_label, numTrue, numFalse


metric1_greaterthan_metric2.__doc__ = metric1_greaterthan_metric2.__doc__.format(**decision_docs)


def variance_lt_thresholds(comptable, decision_node_idx, iftrue, iffalse,
                           decide_comps, varmetric='varexp',
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

    used_metrics = [varmetric]

    if only_used_metrics:
        return used_metrics

    functionname_idx = ("variance_lt_thresholds, step " + str(decision_node_idx))
    if custom_node_label:
        node_label = custom_node_label
    else:
        node_label = (used_metrics + "<" + str(single_comp_threshold) + "."
                      " All variance<" + str(all_comp_threshold))

    if log_extra_info:
        LGR.info(log_extra_info)
    if log_extra_report:
        LGR.report(log_extra_report)

    metrics_exist, missing_metrics = confirm_metrics_calculated(comptable, used_metrics)
    if metrics_exist is False:
        error_msg = ("Necessary metrics for " + functionname_idx + "are not in comptable. "
                     "Need to calculate the following metrics: " + missing_metrics)
        raise ValueError(error_msg)

    comps2use = selectcomps2use(comptable, decide_comps)
    if comps2use is None:
        log_decision_tree_step(functionname_idx, comps2use)
    else:
        variance = comptable.loc[comps2use, varmetric]
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
        numTrue = (comps2use is True).sum()
        numFalse = (comps2use is False).sum()
        log_decision_tree_step(functionname_idx, comps2use,
                               numTrue=numTrue,
                               numFalse=numFalse)

    return comptable, used_metrics, node_label, numTrue, numFalse
