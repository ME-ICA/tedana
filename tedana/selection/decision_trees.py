"""
Functions that include workflows to identify
TE-dependent and TE-independent components.
"""

import inspect
import json
import logging
from pkg_resources import resource_filename

from tedana.selection._utils import (
    clean_dataframe, confirm_metrics_exist,
    are_only_necessary_metrics_used)
from tedana.selection import selection_nodes

LGR = logging.getLogger(__name__)
VALID_TREES = [
    'mdt', 'minimal_decision_tree1'
]


class TreeError(Exception):
    pass


def load_config(tree):
    fname = resource_filename('tedana', 'selection/data/{}.json'.format(tree))
    try:
        with open(fname, 'r') as src:
            dectree = json.loads(src.read())
    except FileNotFoundError as err:
        raise ValueError('Invalid decision tree name: {}. Must be one of '
                         '{}'.format(tree, VALID_TREES))
    return validate_tree(dectree)


def validate_tree(tree):
    """
    Confirms that provided `tree` is a valid decision tree

    Parameters
    ----------
    tree : dict
        Ostensible decision tree dictionary

    Returns
    -------
    tree : dict
        Validated decision tree dictionary

    Raises
    ------
    TreeError
    """

    err_msg = ''
    tree_info = ['tree_id', 'info', 'report', 'refs', 'necessary_metrics', 'nodes']
    defaults = {'comptable', 'decision_node_idx'}

    for k in tree_info:
        try:
            assert tree.get(k) is not None
        except AssertionError as err:
            err_msg += 'Decision tree missing required info: {}\n'.format(k)

    for i, node in enumerate(tree['nodes']):
        try:
            fcn = getattr(selection_nodes, node.get('functionname'))
            sig = inspect.signature(fcn)
        except (AttributeError, TypeError) as err:
            err_msg += ('Node {} has invalid functionname parameter: {}\n'
                        .format(i, node.get('functionname')))
            continue

        sig = inspect.signature(fcn)
        pos = set([p for p, i in sig.parameters.items() if i.default is inspect.Parameter.empty])
        kwargs = set(sig.parameters.keys()) - pos

        missing_pos = pos - set(node.get('parameters').keys()) - defaults
        if len(missing_pos) > 0:
            err_msg += ('Node {} is missing required parameter(s): {}\n'
                        .format(i, missing_pos))
        invalid_kwargs = set(node.get('kwargs').keys()) - kwargs
        if len(invalid_kwargs) > 0:
            err_msg += ('Node {} has additional, undefined kwarg(s): {}\n'
                        .format(i, invalid_kwargs))

    if err_msg:
        raise TreeError('\n' + err_msg)

    return tree


class DecisionTree:
    """
    Classifies components based on specified `tree`

    The selection process uses previously calculated parameters listed in
    comptable for each ICA component such as Kappa (a T2* weighting metric),
    Rho (an S0 weighting metric), and variance explained.
    See `Notes` for additional calculated metrics used to classify each
    component into one of the listed groups.

    Parameters
    ----------
    tree : str
        Decision tree to use
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric; the index should be the component number!
    n_echos : int
        Number of echos in original data; this is only used to get threshold
        for F statistic related to the "elbow" calculation.
    user_notes : str, optional
        Additional user notes about decision tree

    Returns
    -------
    comptable : :obj:`pandas.DataFrame`
        Updated component table with additional metrics and with classification
        (i.e., accepted, rejected, or ignored)
    nodes : list of dict
        Nodes used in decision tree, including function names, parameters
        provided, and relevant modifications made to comptable

    Notes
    -----
    The selection algorithm used in this function is a minimalist version based
    on ME-ICA by Prantik Kundu, and his original implementation is available
    at: https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model.py

    This component selection process uses multiple, previously calculated
    metrics that include kappa, rho, variance explained, noise and spatial
    frequency metrics, and measures of spatial overlap across metrics.

    For this decision tree:
        4 extreme rejection metrics are applied
        A kappa and rho elbow are calculated and used to reject components
        Potentially rejected components with very low variance explained
        are moved to ignored
    """

    def __init__(self, tree, comptable, n_echos, user_notes=None):
        # initialize stuff based on the info in specified `tree`
        self.tree = tree
        self.config = load_config(self.tree)
        self.comptable = comptable.copy()

        LGR.info('Performing component selection with ' + self.config['tree_id'])
        report = self.config['report']
        if user_notes is not None:
            report += user_notes
        LGR.report(report)
        LGR.refs(self.config['refs'])

        self.nodes = self.config['nodes']
        self.metrics = set(config['necessary_metrics'])
        self.used_metrics = []

        # this will crash the program with an error message if not all
        # necessary_metrics are in the comptable
        confirm_metrics_exist(self.comptable, self.metrics, self.tree)

    # for each node that is run:
    # 1. Create decision_node_idx as a variable to pass to the function & in the decision tree dict
    # 2. Log the function call in LGR.info.
    # 3. Run function.
    #     This will return: comptable (updated), used_metrics, numTrue, numFalse
    # 4. Add information to the decision tree dict: numTrue, numFalse, used_metrics

    # The comptable values for classification & rationale may be updated
    # A new element is added to decision_tree_steps with an index incremented by 1
    # necessary_metrics is a list of metrics used in this specific function

    def run(self):
        used_metrics = set()
        for ii, node in enumerate(self.nodes):
            fcn = getattr(selection_nodes, node['functionname'])
            params, kwargs = node['parameters'], node['kwargs']
            LGR.info('Running function {} with parameters: {}'
                     .format(node['functionname'], {**params, **kwargs}))
            self.comptable, metrics, numTrue, numFalse = \
                fcn(self.comptable, decision_node_idx=ii, **params, **kwargs)
            used_metrics.update(metrics)
            self.nodes[ii].update(
                decision_node_idx=ii,
                numTrue=numTrue,
                numFalse=numFalse,
                used_metrics=metrics
            )

        # Move decision columns to end
        self.comptable = clean_dataframe(self.comptable)
        self.are_only_necessary_metrics_used(used_metrics)

        return self.comptable, self.nodes

    def are_only_necessary_metrics_used(self, used_metrics):
        # This function checks if all metrics that are declared as necessary
        # are actually used and if any used_metrics weren't explicitly declared
        # If either of these happen, a warning is added to the logger
        not_declared = set(used_metrics) - set(self.metrics)
        not_used = set(self.metrics) - set(used_metrics)
        if len(not_declared) > 0:
            LGR.warn('Decision tree {} used additional metrics not declared '
                     'as necessary: {}'.format(self.tree, not_declared))
        if len(not_used) > 0:
            LGR.warn('Decision tree {} failed to use metrics that were '
                     'declared as necessary: {}'.format(self.tree, not_used))


# def manual_selection(comptable, acc=None, rej=None):
#     """
#     Perform manual selection of components.

#     Parameters
#     ----------
#     comptable : (C x M) :obj:`pandas.DataFrame`
#         Component metric table, where `C` is components and `M` is metrics
#     acc : :obj:`list`, optional
#         List of accepted components. Default is None.
#     rej : :obj:`list`, optional
#         List of rejected components. Default is None.

#     Returns
#     -------
#     comptable : (C x M) :obj:`pandas.DataFrame`
#         Component metric table with classification.
#     """
#     LGR.info('Performing manual ICA component selection')
#     test1 = 'classification' in comptable.columns
#     test2 = 'original_classification' not in comptable.columns
#     if test1 and test2:
#         comptable['original_classification'] = comptable['classification']
#         comptable['original_rationale'] = comptable['rationale']

#     comptable['classification'] = 'accepted'
#     comptable['rationale'] = ''

#     all_comps = comptable.index.values
#     if acc is not None:
#         acc = [int(comp) for comp in acc]

#     if rej is not None:
#         rej = [int(comp) for comp in rej]

#     if acc is not None and rej is None:
#         rej = sorted(np.setdiff1d(all_comps, acc))
#     elif acc is None and rej is not None:
#         acc = sorted(np.setdiff1d(all_comps, rej))
#     elif acc is None and rej is None:
#         LGR.info('No manually accepted or rejected components supplied. '
#                  'Accepting all components.')
#         # Accept all components if no manual selection provided
#         acc = all_comps[:]
#         rej = []

#     ign = np.setdiff1d(all_comps, np.union1d(acc, rej))
#     comptable.loc[acc, 'classification'] = 'accepted'
#     comptable.loc[rej, 'classification'] = 'rejected'
#     comptable.loc[rej, 'rationale'] += 'I001;'
#     comptable.loc[ign, 'classification'] = 'ignored'
#     comptable.loc[ign, 'rationale'] += 'I001;'

#     # Move decision columns to end
#     comptable = clean_dataframe(comptable)
#     return comptable
