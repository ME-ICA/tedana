"""
Functions that will be used as steps in a decision tree
"""
import logging
import numpy as np
from scipy import stats

from tedana.selection import decision_tree_ancillary
from tedana.stats import getfbounds
from tedana.selection._utils import getelbow, clean_dataframe

LGR = logging.getLogger(__name__)


def RhoGtKappa(comptable, decision_tree_steps, iftrue='reject', iffalse='no_change',
               decide_comps='all', kappa_scale=1):
    """
     Is Rho Greater than Kappa? (currently: I002)
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

    