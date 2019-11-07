"""
Functions that will be used as steps in a decision tree
"""
import logging
import numpy as np
from scipy import stats

from tedana.stats import getfbounds
from tedana.selection._utils import ( 
        getelbow, clean_dataframe, confirm_metrics_calculated, 
        new_decision_node_info, selectcomps2use, log_decision_tree_step,
        change_comptable_classifications)

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
    decision_tree_steps : :obj:`list`
            A list of dictionaries that contains information about what happens
            at each decision tree node.
    iftrue : :obj:`str`
            If the condition in this step is true, either give the component
            the label in this string. Options are 'accept', 'reject', 
            'provisionalaccept', 'provisionalreject', 'ignore', or 'no_change'
            If
            
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
        decision_boolean = comptable.loc[comps2use, 'rho'] > (kappa_scale*comptable.loc[comps2use, 'kappa'])
        
        comptable, decision_tree_steps = change_comptable_classifications(
                        comptable, iftrue, iffalse, 
                        decision_boolean, nodeidxstr, decision_tree_steps)

        log_decision_tree_step(nodeidxstr, functionname, comps2use, 
            decision_tree_steps=decision_tree_steps)

    return comptable, decision_tree_steps, necessary_metrics

    
CompTable, DecisionTreeSteps, MetricsUsed 
    = CountsigS0GtT2(CompTable, IfTrue=’Reject’, IfFalse=’NoChange’,
        DecideComps=’all’, CoutsigScale=1, DecisionTreeSteps=DecisionTreeSteps)
# Is CountsigS0>CountSigT2 (currently: I003)
    MetricsUsed = CountsigS0, CountSigT2
    for the components in DecideComps
    if CompTable.CountsigS0>(CountsigScale*CompTable.CountSigT2):
        Change the component classification to Reject
        Update the tedana codes (i.e. I00X) to say this function
        changed the classification of this component
    else
        NoChange to the component classification
    Add a new element to DecisionTreeSteps
    return CompTable and MetricsUsed
