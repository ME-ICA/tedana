"""
Functions that include workflows to identify
TE-dependent and TE-independent components.
"""
import os.path as op
import inspect
import json
import logging
from pkg_resources import resource_filename

from tedana.selection._utils import (
    clean_dataframe, confirm_metrics_exist)
from tedana.selection import selection_nodes

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

# These are the names of the json files containing decision
# trees that are stored in the ./selection/data/ directory
# A user can run the desision tree either using one of these
# names or by giving the full path to a tree in a different
# location
DEFAULT_TREES = [
    'minimal_decision_tree1',
    'kundu_MEICA27_decision_tree'
]


class TreeError(Exception):
    pass


def load_config(tree, path=None):
    """
    Loads the json file with the decision tree and validates that the
    fields in the decision tree are appropriate.

    Parameters
    ----------
    tree : :obj:`str`
        A json file name without the '.json' extension
    path : :obj:`str`
        The directory path where `tree` is located.
        If None, then look for `tree` within ./selection/data
        in the tedana code directory. default=None

    Returns
    -------
    tree : :obj:`dict`
        A validated decision tree for the component selection process.::|br|
        The `dict` has several required fields to describe the entire tree ::|br|
        `tree_id`: The name of the tree|br|
        `info`: A brief description of the tree to be used in info logging|br|
        `report`: A narrative description of the tree that could be used in report logging|br|
        `refs`: Publications that should be referenced, when this tree is used|br|
        `necessary_metrics`: The metrics in `comptable` that will be used by this tree|br|
        `nodes`: A list of dictionaries where each dictionary includes the information<br>
        to run one node in the decision tree. This includes:<br>
        `functionname`: The name of the function to be called<br>
        `parameters`: Required parameters for the function<br>
        The only parameter that is used in all functions is `decidecomps`.
        This is a list of component classifications, that this function should
        operate on. Most functions also include `iftrue` and `iffalse` which
        define how to to change the classification of a component if the
        criteria in the function is true or false.<br>
        `kwargs`: Optional parameters for the function
    """
    if path:
        fname = op.join(path, (tree + '.json'))
    else:
        fname = resource_filename('tedana', 'selection/data/{}.json'.format(tree))
    try:
        with open(fname, 'r') as src:
            dectree = json.loads(src.read())
    except FileNotFoundError:
        if path:
            raise ValueError('Invalid decision tree: {}. Default tree options are '
                             '{}'.format(fname, DEFAULT_TREES))
        else:
            raise ValueError('Invalid decision tree name: {}. Default tree options are '
                             '{}'.format(tree, DEFAULT_TREES))
    return validate_tree(dectree)


def validate_tree(tree):
    """
    Confirms that provided `tree` is a valid decision tree

    Parameters
    ----------
    tree : :obj:`dict`
        Ostensible decision tree for the component selection process

    Returns
    -------
    tree : :obj:`dict`
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
        except AssertionError:
            err_msg += 'Decision tree missing required info: {}\n'.format(k)

    for i, node in enumerate(tree['nodes']):
        try:
            fcn = getattr(selection_nodes, node.get('functionname'))
            sig = inspect.signature(fcn)
        except (AttributeError, TypeError):
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
    Classifies components based on specified `tree` when the class is initialized
    and then the `run` function is called.
    The expected output of running a decision tree is that every component
    will be classified as 'accept', 'reject', or 'ignore'.

    The selection process uses previously calculated parameters listed in
    `comptable` for each ICA component such as Kappa (a T2* weighting metric),
    Rho (an S0 weighting metric), and variance explained. See tedana.metrics
    for more detail on the calculated metrics

    Parameters
    ----------
    tree : :obj:`str`
        A json file name without the '.json' extension that contains the decision tree to use
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric; the index should be the component number!
    user_notes : str, optional
        Additional user notes about decision tree
    path : :obj:`str, optional`
        The directory path where `tree` is located.
        If None, then look for `tree` within ./selection/data
        in the tedana code directory. default=None

    Additional Parameters
    ---------------------
    Any parameter that is used by a decision tree node function can be passed
    as a parameter of DecisionTree class initialization function or can be
    included in the json file that defines the decision tree. If a parameter
    is set in the json file, that will take precedence. As a style rule, a
    parameter that is the same regardless of the inputted data should be
    defined in the decision tree json file. A parameter that is dataset specific
    should be passed through the initialization function. Parameters that may need
    to be passed through the class include:

    n_echos : :obj:`int, optional`
        Number of echos in multi-echo fMRI data
    n_vols: :obj:`int`
        Number of volumes (time points) in the fMRI data


    Returns
    -------
    comptable : :obj:`pandas.DataFrame`
        Updated component table with additional metrics and with classifications
        (i.e., accepted, rejected, or ignored) for each component
    nodes : :obj:`dict`
        Nodes used in decision tree. This includes the decision tree dict
        from the json file in the `tree` input. For every dict in the list of
        functions called in the decision tree, there is an added key `outputs`
        which includes four values:
        decison_node_idx : :obj:`int`
            The decision tree function are run as part of an ordered list.
            This is the positional index for when this function has been run
            as part of this list.
        used_metrics : :obj:`list[str]`
            A list of the metrics used in a node of the decision tree
        node_label : :obj:`str`
            A brief label for what happens in this node that can be used in a decision
            tree summary table or flow chart.
        numTrue, numFalse : :obj:`int`
            The number of components that were classified as true or false respectively
            in this decision tree step.

    Notes
    -----
    """

    def __init__(self, tree, comptable, **kwargs):
        # initialize stuff based on the info in specified `tree`
        self.tree = tree
        self.comptable = comptable.copy()

        self.__dict__.update(kwargs)
        if hasattr(self, 'path'):
            self.config = load_config(self.tree, self.path)
        else:
            self.config = load_config(self.tree)

        LGR.info('Performing component selection with ' + self.config['tree_id'])
        LGR.info(self.config.get('info', ''))
        RepLGR.info(self.config.get('report', ''))
        RefLGR.info(self.config.get('refs', ''))

        self.nodes = self.config['nodes']
        self.metrics = self.config['necessary_metrics']
        self.used_metrics = []

    def run(self):
        """
        Parse the parameters used to call each function in the component
        selection decision tree and run the functions to classify components

        Parameters all defined in class initialization

        Returns
        -------
        comptable : :obj:`pandas.DataFrame`
            Updated component table with additional metrics and with classifications
            (i.e., accepted, rejected, or ignored) for each component
        nodes : :obj:`dict`
            Nodes used in decision tree with updated information from run-time



        """

        # this will crash the program with an error message if not all
        # necessary_metrics are in the comptable
        confirm_metrics_exist(self.comptable, self.metrics, self.tree)

        used_metrics = set()
        for ii, node in enumerate(self.nodes):
            fcn = getattr(selection_nodes, node['functionname'])

            params, kwargs = node['parameters'], node['kwargs']
            params = self.check_null(params, node['functionname'])
            kwargs = self.check_null(kwargs, node['functionname'])

            LGR.info('Running function {} with parameters: {}'
                     .format(node['functionname'], {**params, **kwargs}))
            self.comptable, dnode_outputs = fcn(
                self.comptable, decision_node_idx=ii, **params, **kwargs)
            used_metrics.update(dnode_outputs['outputs']['used_metrics'])
            print(node['functionname'])
            # print(list(self.comptable['rationale']))
            # dnode_outputs is a dict that should always include fields for
            #   decision_node_idx, numTrue, numFalse, used_metrics, and node_label
            #   any other fields will also be logged in this output
            self.nodes[ii].update(dnode_outputs)

        # Move decision columns to end
        self.comptable = clean_dataframe(self.comptable)
        self.are_only_necessary_metrics_used(used_metrics)
        print(self.nodes)
        return self.comptable, self.nodes

    def check_necessary_metrics(self):
        used_metrics = set()
        for ii, node in enumerate(self.nodes):
            fcn = getattr(selection_nodes, node['functionname'])

            params, kwargs = node['parameters'], node['kwargs']
            params = self.check_null(params, node['functionname'])
            kwargs = self.check_null(kwargs, node['functionname'])

            LGR.info('Checking necessary metrics for function {} with parameters: {}'
                     .format(node['functionname'], {**params, **kwargs}))
            func_used_metrics = fcn(
                self.comptable, decision_node_idx=ii, **params, **kwargs,
                only_used_metrics=True)
            used_metrics.update(func_used_metrics)
        LGR.info('Used metrics: {}'.format(used_metrics))
        return used_metrics

    def check_null(self, params, fcn):
        for key, val in params.items():
            if val is None:
                try:
                    params[key] = getattr(self, key)
                except AttributeError:
                    raise ValueError('Parameter {} is required in node {}, but not defined. '
                                     .format(key, fcn) + 'If {} is dataset specific, it should be '
                                     'defined in the '.format(key) + ' initialization of '
                                     'DecisionTree. If it is fixed regardless of dataset, it '
                                     'should be defined in the json file that defines the '
                                     'decision tree.')

        return params

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
