"""
Functions that include workflows to identify
TE-dependent and TE-independent components.
"""
import logging
import yaml
from jsonschema import validate
from pathlib import Path

# import numpy as np
# from scipy import stats

# from tedana.stats import getfbounds
from tedana.selection._utils import (
    clean_dataframe, confirm_metrics_calculated,
    are_only_necessary_metrics_used)  # getelbow
from tedana.selection.selection_nodes import RhoGtKappa

LGR = logging.getLogger(__name__)


# class Metrics:
#     def compute_metric(self, metric):
#         print("Important stuff goes here")

#     def get_metric(self, metric):
#         if metric not in self.comptable.columns:
#             self.compute_metric(metric)

#     def kappa(self):
#         return self.comptable['kappa']


# metrics = Metrics(...)
# metrics.kappa
# hasattr(metrics, kappa)

# my_tree = DecisionTree(config, metrics, n_echos)
# my_tree.run()


def read_config(config_file):
    # raise NotImplemented

    config_file = Path(config_file)
    is_valid = False
    test_dict = yaml.load(config_file.read_text())

    # dtree_schema = {
    #                 "full_tree_info": {
    #                     'tree_id': {"type": "str"},
    #                     'info': {"type": "str"},
    #                     'report': {"type": "str"},
    #                     'refs': {"type": "str"},
    #                     'necessary_metrics': {"type": "list"}
    #                                   },
    #                 "tree_node": {
    #                     'functionname': {"type": "str"},
    #                     'iftrue': {"type": "str"},
    #                     'iffalse': {"type": "str"},
    #                     'additionalparams': {"type": "str"},  # use arg?
    #                     'report_extra_details': {"type": "str"}
    #                 }
    #         }

    # validate(test_dict, dtree_schema)

    # 'decision_tree_full': ['see below...']

    #         "$schema": "defined_inplace",
    #         "type": "object",
    #         "properties": {
    #             "env_vars": {"type": "array"},
    #             "env": {"type": "array"},
    #             "inputs": {"type": "array"},
    #             "execute": {"type": "object"},
    #             "keywords": {"type": "array"},
    #             "label": {"type": "string"},
    #         },
    #         "required": ["inputs", "execute"],
    #     }
    #     return self._schema


class decision_tree:
    """
    Classify components as "accepted," "rejected," or "ignored" based on
    relevant metrics.

    The selection process uses previously calculated parameters listed in
    comptable for each ICA component such as Kappa (a T2* weighting metric),
    Rho (an S0 weighting metric), and variance explained.
    See `Notes` for additional calculated metrics used to classify each
    component into one of the listed groups.

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    n_echos : :obj:`int`
        Number of echos in original data
        Only used to get threshold for F statistic related to the "elbow" calculation

    Returns
    -------
    comptable : :obj:`pandas.DataFrame`
        Updated component table with additional metrics and with
        classification (accepted, rejected, or ignored)

    Notes
    -----
    The selection algorithm used in this function is a minimalist version based on
    ME-ICA by Prantik Kundu, and his original implementation is available at:
    https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model.py

    This component selection process uses multiple, previously calculated
    metrics that include kappa, rho, variance explained, noise and spatial
    frequency metrics, and measures of spatial overlap across metrics.

    For this decision tree:
        4 extreme rejection metrics are applied
        A kappa and rho elbow are calculated and used to reject components
        Potentially rejected components with very low variance explained
        are moved to ignored
    """

    def __init__(self, config_file, metrics, n_echos, user_notes=None):
        # initialize stuff based on the info in full_tree_info
        config = read_config(config_file)

        LGR.info('Performing component selection with ' + config['tree_id'])
        report = config['report']
        if user_notes:
            report += user_notes
        LGR.report(report)
        LGR.refs(config['refs'])
        self.metrics = metrics
        self.nodes = config['nodes']
        # this will probably be grabbing functionname from each element
        self.node_list = [x['functionname'] for x in config['nodes']]

        self.necessary_metrics = config['necessary_metrics']
        self.used_metrics = []

        # This will crash the program with an error message if not all necessary_metrics
        # are in the comptable
        # This should not crash so that uncomputed metrics are allowed
        confirm_metrics_calculated(comptable, necessary_metrics, functionname=functionname)

    # for each node that is run:
    # 1. Create decision_node_idx as a variable to pass to the function & in the decision tree dict
    # 2. Log the function call in LGR.info.
    # 3. Run function.
    #     This will return: comptable (updated), used_metrics, numTrue, numFalse
    # 4. Add information to the decision tree dict: numTrue, numFalse, used_metrics

    # The comptable values for classification & rationale may be updated
    # A new element is added to decision_tree_steps with an index incremented by 1
    # necessary_metrics is a list of metrics used in this specific function

    def run_node(self, node):
        'nodeidx':  # will be filled in at runtime
            'functionname': tedana.metric.KappaGtElbow,
            'metrics_used':  # will be filled in at runtime
            'decide_comps': 'unclassified',
            'iftrue': 'provisionallyaccept',
            'iffalse': 'provisionallyreject',
            'additionalparameters': [],
            'report_extra_log': [],  # optionally defined by user
            'numfalse': [],  # will be filled in at runtime
            'numtrue': [],  # will be filled in at runtime

    def run(self):
        for ii, node in enumerate(self.node_list):
            self.run_node(self.nodes[ii])

        # This function checks if all metrics that are declared as necessary
        # are actually used and if any used_metrics weren't explicitly declared
        # If either of these happen, a warning is added to the logger
    are_only_necessary_metrics_used()

    # Move decision columns to end
    comptable = clean_dataframe(comptable)

    return comptable, decision_tree_steps


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
