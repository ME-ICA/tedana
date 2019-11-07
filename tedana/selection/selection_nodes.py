"""
Functions that will be used as steps in a decision tree
"""
import logging
# import numpy as np
# from scipy import stats

# from tedana.stats import getfbounds
from tedana.selection._utils import (confirm_metrics_calculated,
                                     new_decision_node_info, selectcomps2use,
                                     log_decision_tree_step, change_comptable_classifications)
# getelbow, clean_dataframe
LGR = logging.getLogger(__name__)


def RhoGtKappa(comptable, decision_tree_steps, iftrue='reject', iffalse='no_change',
               decide_comps='all', kappa_scale=1):
    """
    Is Rho>(kappa_scale*Kappa)

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
            Component metric table. One row for each component, with a column for
            each metric. The index should be the component number.
    decision_tree_steps : :obj:`list[dict]`
            A list of dictionaries that contains information about what happens
            at each decision tree node.
    iftrue : :obj:`str`
            If the condition in this step is true, give the component
            the label in this string. Options are 'accept', 'reject',
            'provisionalaccept', 'provisionalreject', 'ignore', or 'no_change'
            If 'no_change' then don't change the current component classification
    iffalse: :obj:`str`
            If the condition in this step is false, give the component
            the label in this string. Same options as iftrue
    decide_comps: :obj:`union(str, list[str])`
            A string or a list of strings listing what classifications of components
            to operate on. For example: If 'all' then run on all components.
            If 'unclassified' run on only components labeled 'unclassified'.
            If ['unclassified', 'provionalreject'] run on components with either of
            those classifications
    kappa_scale: :obj:`float`
            Multiple a component's kappa value by kappa_scale before testing if it
            is greater than a component's rho value. default=1

    Returns
    -------
    comptable: (C x M) :obj:`pandas.DataFrame`
            Component metric table. One row for each component, with a column for
            each metric. The index should be the component number.
            Component classifications may change and, for components where the
            classification has changed, add a string to rationale

    decision_tree_steps, necessary_metrics
    """

    functionname = 'RhoGtKappa'
    necessary_metrics = ('kappa', 'rho')
    metrics_exist, missing_metrics = confirm_metrics_calculated(comptable, necessary_metrics)
    if metrics_exist is False:
        error_msg = ("Necessary metrics for " + functionname + " are not in comptable. "
                     "Need to calculate the following metrics: " + missing_metrics)
        raise ValueError(error_msg)

    decision_tree_steps = new_decision_node_info(decision_tree_steps, functionname,
                                                 necessary_metrics, iftrue, iffalse,
                                                 additionalparameters=None)
    nodeidxstr = str(decision_tree_steps[-1]['nodeidx'])

    comps2use = selectcomps2use(comptable, decide_comps)

    if comps2use is None:
        log_decision_tree_step(nodeidxstr, functionname, comps2use,
                               decide_comps=decide_comps)
    else:
        decision_boolean = (
            comptable.loc[comps2use, 'rho'] > (kappa_scale * comptable.loc[comps2use, 'kappa']))

        comptable, decision_tree_steps = change_comptable_classifications(
                        comptable, iftrue, iffalse,
                        decision_boolean, nodeidxstr, decision_tree_steps)

        log_decision_tree_step(nodeidxstr, functionname, comps2use,
                               decision_tree_steps=decision_tree_steps)

    return comptable, decision_tree_steps, necessary_metrics
