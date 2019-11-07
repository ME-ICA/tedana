"""
Functions that include workflows to identify 
TE-dependent and TE-independent components.
"""
import logging
import numpy as np
from scipy import stats

from tedana.stats import getfbounds
from tedana.selection._utils import getelbow, clean_dataframe

LGR = logging.getLogger(__name__)




def confirm_metrics_calculated(comptable, necessary_metrics):
    # NEED TO ACTUALLY WRITE THIS FUNCTION
    metrics_exist = True
    missing_metrics = None
    return metrics_exist, missing_metrics

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
                         'numfalse': [],
                         'numtrue': []}

    if decision_tree_steps is None:
        decision_tree_steps = [tmp_decision_tree]
        decision_tree_steps['nodeidx'] = 1
    else:
        tmp_decision_tree['nodeidx'] = len(decision_tree_steps) + 1
        decision_tree_steps.append(tmp_decision_tree)

    return decision_tree_steps

def selectcomps2use(comptable, decide_comps):
    """
    Give a list of components that fit a classification types
    """
    if decide_comps == 'all':
        # This classification always has a value, this is a way to get all
        # values in the same structure as the subset lists below
        # There's probably a simpler way to do this
        comps2use = comptable['classification'] != None
    elif (type(decide_comps)==str):
        comps2use = comptable['classification'] == decide_comps
    else:
        comps2use = comptable['classification'] == decide_comps[0]
        for didx in range(len(decide_comps-1)):
            comps2use = (comps2use | 
                (comptable['classification'] == decide_compsl[didx+1]))

    # If no components are selected, then return None.
    # The function that called this can check for None and exit before
    # attempting any computations on no data
    if comps2use.sum() == 0:
        comps2use = None

    return comps2use


def log_decision_tree_step(nodeidxstr, functionname, comps2use, 
    decide_comps=None, decision_tree_steps=None):
    """
        Logging text to add for every decision tree calculation
    """

    if comps2use is None:
        LGR.info("Step " + nodeidxstr + " " + functionname + " not applied because " +
            "no remaining components were classified as " + str(decide_comps))
    else:
        LGR.info("Step " + nodeidxstr + " " + functionname + " applied to " +
                str((comps2use==True).sum()) + " components. " 
                + str(decision_tree_steps[-1]['numtrue']) + " were True and "
                + str(decision_tree_steps[-1]['numfalse']) + " were False")


def change_comptable_classifications(comptable, iftrue, iffalse, 
        decision_boolean, nodeidxstr, decision_tree_steps):
    """
    Given information on whether a decision critereon is true or false for each component
    change or don't change the compnent classification
    """
    if iftrue != 'no_change':
        changeidx=decision_boolean.index[decision_boolean.values==True]
        comptable['classification'][changeidx] = iftrue
        comptable['rationale'][changeidx]+=(nodeidxstr + ': ' + iftrue + '; ')
    if iffalse != 'no_change':
        changeidx=decision_boolean.index[decision_boolean.values==False]
        comptable['classification'][changeidx] = iffalse
        comptable['rationale'][changeidx]+=(nodeidxstr + ': ' + iffalse + '; ')

    decision_tree_steps[-1]['numtrue'] = (decision_boolean==True).sum()
    decision_tree_steps[-1]['numfalse'] = (decision_boolean==False).sum()
    
    return comptable, decision_tree_steps