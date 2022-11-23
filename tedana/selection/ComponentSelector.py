"""
Functions that include workflows to identify and label
TE-dependent and TE-independent components.
"""
import inspect
import logging
import os.path as op

from numpy import asarray

from tedana.io import load_json
from tedana.selection import selection_nodes
from tedana.selection.selection_utils import (
    clean_dataframe,
    confirm_metrics_exist,
    log_classification_counts,
)
from tedana.utils import get_resource_path

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")

# These are the names of the json files containing decision
# trees that are stored in the ./resouces/decision_trees/ directory
# A user can run the desision tree either using one of these
# names or by giving the full path to a tree in a different
# location
DEFAULT_TREES = ["minimal", "kundu"]


class TreeError(Exception):
    pass


def load_config(tree):
    """
    Loads the json file with the decision tree and validates that the
    fields in the decision tree are appropriate.

    Parameters
    ----------
    tree : :obj:`str`
        The named tree or path to a JSON file that defines one

    Returns
    -------
    tree : :obj:`dict`
        A validated decision tree for the component selection process.
    """

    #      Formerly used text
    #      The `dict` has several required fields to describe the entire tree
    #      - `tree_id`: :obj:`str` The name of the tree
    #      - `info`: :obj:`str` A brief description of the tree for info logging
    #      - `report`: :obj:`str`
    #      - A narrative description of the tree that could be used in report logging
    #      - `refs`: :obj:`str` Publications that should be referenced, when this tree is used
    #      - `necessary_metrics`: :obj:`list[str]`
    #      - The metrics in `component_table` that will be used by this tree
    #      - `intermediate_classifications`: :obj:`list[str]`
    #      - User specified component classification labels. 'accepted', 'rejected', and
    #      - 'unclassified' are defaults that don't need to be included here
    #      - `classification_tags`: :obj:`list[str]`
    #      - Descriptive labels to be used to explain why a component was accepted or rejected.
    #      - For example, ["Likely BOLD","Low variance"]
    #      - `nodes`: :obj:`list[dict]` Each dictionary includes the information
    #
    #        to run one node in the decision tree. Each node should either be able
    #        to change component classifications (function names starting with ``dec_``)
    #        or calculate values using information from multiple components
    #        (function names starting with ``calc_``)
    #        nodes includes:
    #        - `functionname`: :obj:`str` The name of the function to be called
    #        - `parameters`: :obj:`dict` Required parameters for the function
    #          The only parameter that is used in all functions is `decidecomps`,
    #          which are the component classifications the function should run on.
    #          Most ``dec_`` functions also include `ifTrue` and `ifFalse` which
    #          define how to to change the classification of a component if the
    #          criteria in the function is true or false.
    #
    #        - `kwargs`: :obj:`dict` Optional parameters for the function

    if tree in DEFAULT_TREES:
        fname = op.join(get_resource_path(), "decision_trees", tree + ".json")
    else:
        fname = tree

    try:
        dectree = load_json(fname)
    except FileNotFoundError:
        raise ValueError(
            f"Cannot find tree {tree}. Please check your path or use a "
            f"default tree ({DEFAULT_TREES})."
        )
    except IsADirectoryError:
        raise ValueError(
            f"Tree {tree} is a directory. Please supply a JSON file or "
            f"default tree ({DEFAULT_TREES})."
        )

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

    # Set the fields that should always be present
    err_msg = ""
    tree_expected_keys = [
        "tree_id",
        "info",
        "report",
        "refs",
        "necessary_metrics",
        "intermediate_classifications",
        "classification_tags",
        "nodes",
    ]
    defaults = {"selector", "decision_node_idx"}
    default_classifications = {"nochange", "accepted", "rejected", "unclassified"}
    default_decide_comps = {"all", "accepted", "rejected", "unclassified"}

    # Confirm that the required fields exist
    missing_keys = set(tree_expected_keys) - set(tree.keys())
    if missing_keys:
        # If there are missing keys, this function may crash before the end.
        # End function here with a clear error message rather than adding
        # `if assert tree.get()` statements before every section
        raise TreeError("\n" + f"Decision tree missing required fields: {missing_keys}")

    # Warn if unused fields exist
    unused_keys = set(tree.keys()) - set(tree_expected_keys) - set(["used_metrics"])
    # Make sure reconstruct_from doesn't trigger a warning; hacky, sorry
    if "reconstruct_from" in unused_keys:
        unused_keys.remove("reconstruct_from")

    if unused_keys:
        LGR.warning(f"Decision tree includes fields that are not used or logged {unused_keys}")

    # Combine the default classifications with the user inputted classifications
    all_classifications = set(tree.get("intermediate_classifications")) | set(
        default_classifications
    )
    all_decide_comps = set(tree.get("intermediate_classifications")) | set(default_decide_comps)
    for i, node in enumerate(tree["nodes"]):
        # Make sure each function defined in a node exists
        try:
            fcn = getattr(selection_nodes, node.get("functionname"))
            sig = inspect.signature(fcn)
        except (AttributeError, TypeError):
            err_msg += "Node {} has invalid functionname parameter: {}\n".format(
                i, node.get("functionname")
            )
            continue

        # Get a functions parameters and compare to parameters defined in the tree
        pos = set([p for p, i in sig.parameters.items() if i.default is inspect.Parameter.empty])
        kwargs = set(sig.parameters.keys()) - pos

        missing_pos = pos - set(node.get("parameters").keys()) - defaults
        if len(missing_pos) > 0:
            err_msg += "Node {} is missing required parameter(s): {}\n".format(i, missing_pos)

        invalid_params = set(node.get("parameters").keys()) - pos
        if len(invalid_params) > 0:
            err_msg += "Node {} has additional, undefined required parameters: {}\n".format(
                i, invalid_params
            )

        # Only if kwargs are inputted, make sure they are all valid
        if node.get("kwargs") is not None:
            invalid_kwargs = set(node.get("kwargs").keys()) - kwargs
            if len(invalid_kwargs) > 0:
                err_msg += (
                    "Node {} has additional, undefined optional parameters (kwargs): {}\n".format(
                        i, invalid_kwargs
                    )
                )

        # Gather all the classification labels used in each tree both for
        # changing classifications and for decide_comps which defines which
        # component classifications to use in each node then make sure these
        # classifications are in the predefined list.
        # It's important to require a predefined list of classifications
        # beccuse spelling inconsistencies cause problems and are hard to
        # catch. For example if a node is applied to "provisionalaccept"
        # nodes, but a previous node classified components as
        # "provisionalaccepted" they won't be included and there might not
        # be any other warnings
        compclass = set()
        if "ifTrue" in node.get("parameters").keys():
            tmp_comp = node["parameters"]["ifTrue"]
            if isinstance(tmp_comp, str):
                tmp_comp = [tmp_comp]
            compclass = compclass | set(tmp_comp)
        if "ifFalse" in node.get("parameters").keys():
            tmp_comp = node["parameters"]["ifFalse"]
            if isinstance(tmp_comp, str):
                tmp_comp = [tmp_comp]
            compclass = compclass | set(tmp_comp)
        nonstandard_labels = compclass.difference(all_classifications)
        if nonstandard_labels:
            LGR.warning(f"{compclass} in node {i} of the decision tree includes a classification")
        if "decide_comps" in node.get("parameters").keys():
            tmp_comp = node["parameters"]["decide_comps"]
            if isinstance(tmp_comp, str):
                tmp_comp = [tmp_comp]
            compclass = set(tmp_comp)
        nonstandard_labels = compclass.difference(all_decide_comps)
        if nonstandard_labels:
            LGR.warning(
                f"{compclass} in node {i} of the decision tree includes a classification "
                "label that was not predefined"
            )

        if node.get("kwargs") is not None:
            tagset = set()
            if "tag_ifTrue" in node.get("kwargs").keys():
                tagset.update(set([node["kwargs"]["tag_ifTrue"]]))
            if "tag_ifFalse" in node.get("kwargs").keys():
                tagset.update(set([node["kwargs"]["tag_ifFalse"]]))
            if "tag" in node.get("kwargs").keys():
                tagset.update(set([node["kwargs"]["tag"]]))
            undefined_classification_tags = tagset.difference(set(tree.get("classification_tags")))
            if undefined_classification_tags:
                LGR.warning(
                    f"{tagset} in node {i} of the decision tree includes a classification "
                    "tag that was not predefined"
                )

    if err_msg:
        raise TreeError("\n" + err_msg)

    return tree


class ComponentSelector:
    """
    Classifies components based on specified `tree` when the class is initialized
    and then the `select` function is called.
    The expected output of running a decision tree is that every component
    will be classified as 'accepted', or 'rejected'.

    The selection process uses previously calculated parameters listed in
    `component_table` for each ICA component such as Kappa (a T2* weighting metric),
    Rho (an S0 weighting metric), and variance explained. See tedana.metrics
    for more detail on the calculated metrics

    Parameters
    ----------
    tree : :obj:`str`
        A json file name without the '.json' extension that contains the decision tree to use
    component_table : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric; the index should be the component number!
    user_notes : :obj:`str, optional`
        Additional user notes about decision tree
    path : :obj:`str, optional`
        The directory path where `tree` is located.
        If None, then look for `tree` within ./selection/data
        in the tedana code directory. default=None


    Returns
    -------
    component_table : :obj:`pandas.DataFrame`
        Updated component table with two extra columns.
    cross_component_metrics : :obj:`Dict`
        Metrics that are each a single value calculated across components.
    component_status_table : :obj:`pandas.DataFrame`
        A table tracking the status of each component at each step.
    nodes : :obj:`list[dict]`
        Nodes used in decision tree.
    current_node_idx : :obj:`int`
        The index for the current node, which should be the last node in the decision tree.

    Notes
    -----
    Any parameter that is used by a decision tree node function can be passed
    as a parameter of ComponentSelector class initialization function or can be
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
    """

    def __init__(self, tree, component_table, cross_component_metrics={}, status_table=None):
        """
        Initialize the class using the info specified in the json file `tree`

        Any optional variables defined in the function call will be added to
        the class structure. Several trees expect n_echos to be defined.
        The full kundu tree also require n_vols (number of volumes) to be
        defined. An example initialization with these options would look like
        selector = ComponentSelector(tree, comptable, n_echos=n_echos,
        n_vols=n_vols)

        Notes
        -----
        The structure has the following fields loaded from tree:

        - nodes
        - necessary_metrics
        - intermediate_classifications
        - classification_tags

        Adds to the class structure:

        - component_status_table: empty dataframe
        - cross_component_metrics: empty dict
        - used_metrics: empty set
        """
        self.tree_name = tree

        self.__dict__.update(cross_component_metrics)
        self.cross_component_metrics = cross_component_metrics

        # Construct an un-executed selector
        self.component_table = component_table.copy()

        # To run a decision tree, each component needs to have an initial classification
        # If the classification column doesn't exist, create it and label all components
        # as unclassified
        if "classification" not in self.component_table:
            self.component_table["classification"] = "unclassified"

        self.tree = load_config(self.tree_name)
        tree_config = self.tree

        LGR.info("Performing component selection with " + tree_config["tree_id"])
        LGR.info(tree_config.get("info", ""))
        RepLGR.info(tree_config.get("report", ""))
        RefLGR.info(tree_config.get("refs", ""))

        self.tree["nodes"] = tree_config["nodes"]
        self.necessary_metrics = set(tree_config["necessary_metrics"])
        self.intermediate_classifications = tree_config["intermediate_classifications"]
        self.classification_tags = set(tree_config["classification_tags"])
        if "used_metrics" not in self.tree.keys():
            self.tree["used_metrics"] = set()
        else:
            self.tree["used_metrics"] = set(self.tree["used_metrics"])

        if status_table is None:
            self.component_status_table = self.component_table[
                ["Component", "classification"]
            ].copy()
            self.component_status_table = self.component_status_table.rename(
                columns={"classification": "initialized classification"}
            )
            self.start_idx = 0
        else:
            # Since a status table exists, we need to skip nodes up to the
            # point where the last tree finished
            self.start_idx = len(tree_config["nodes"])
            self.component_status_table = status_table

    def select(self):
        """
        Parse the parameters used to call each function in the component
        selection decision tree and run the functions to classify components

        Parameters all defined in class initialization

        Returns
        -------
        The following attributes are altered in this function are descibed in
        the ComponentSelector class description:
            component_table, cross_component_metrics, component_status_table,
            cross_component_metrics, used_metrics, nodes (outputs field),
            current_node_idx
        """
        # TODO: force-add classification tags
        if "classification_tags" not in self.component_table.columns:
            self.component_table["classification_tags"] = ""
        # this will crash the program with an error message if not all
        # necessary_metrics are in the comptable
        confirm_metrics_exist(
            self.component_table, self.necessary_metrics, function_name=self.tree_name
        )

        # for each node in the decision tree
        for self.current_node_idx, node in enumerate(self.tree["nodes"][self.start_idx :]):
            # parse the variables to use with the function
            fcn = getattr(selection_nodes, node["functionname"])

            params = node["parameters"]

            params = self.check_null(params, node["functionname"])

            if "kwargs" in node:
                kwargs = node["kwargs"]
                kwargs = self.check_null(kwargs, node["functionname"])
                all_params = {**params, **kwargs}
            else:
                kwargs = None
                all_params = {**params}
            # log the function name and parameters used
            LGR.info(
                "Step {}: Running function {} with parameters: {}".format(
                    self.current_node_idx, node["functionname"], all_params
                )
            )
            # run the decision node function
            if kwargs is not None:
                self = fcn(self, **params, **kwargs)
            else:
                self = fcn(self, **params)
            self.tree["used_metrics"].update(
                self.tree["nodes"][self.current_node_idx]["outputs"]["used_metrics"]
            )

            # log the current counts for all classification labels
            log_classification_counts(self.current_node_idx, self.component_table)

        # move decision columns to end
        self.component_table = clean_dataframe(self.component_table)
        # warning anything called a necessary metric wasn't used and if
        # anything not called a necessary metric was used
        self.are_only_necessary_metrics_used()

        self.are_all_components_accepted_or_rejected()

    def add_manual(self, indices, classification):
        """Add nodes that will manually classify components

        Parameters
        ----------
        indices: list[int]
            The indices to manually classify
        classification: str
            The classification to set the nodes to
        """
        self.tree["nodes"].append(
            {
                "functionname": "manual_classify",
                "parameters": {
                    "new_classification": classification,
                    "decide_comps": indices,
                },
                "kwargs": {
                    "dont_warn_reclassify": "true",
                },
            }
        )

    def check_null(self, params, fcn):
        """
        Checks that all required parameters for selection node functions are
        attributes in the class. Error if any are undefined

        Returns
        -------
        params
            The values for the inputted parameters
        """

        for key, val in params.items():
            if val is None:
                try:
                    params[key] = getattr(self, key)
                except AttributeError:
                    raise ValueError(
                        "Parameter {} is required in node {}, but not defined. ".format(key, fcn)
                        + "If {} is dataset specific, it should be "
                        "defined in the ".format(key) + " initialization of "
                        "ComponentSelector. If it is fixed regardless of dataset, it "
                        "should be defined in the json file that defines the "
                        "decision tree."
                    )

        return params

    def are_only_necessary_metrics_used(self):
        """
        Check if all metrics that are declared as necessary are actually
        used and if any used_metrics weren't explicitly declared necessary
        If either of these happen, a warning is added to the logger
        """
        not_declared = self.tree["used_metrics"] - self.necessary_metrics
        not_used = self.necessary_metrics - self.tree["used_metrics"]
        if len(not_declared) > 0:
            LGR.warning(
                f"Decision tree {self.tree_name} used the following metrics that were "
                "not declared as necessary: {not_declared}"
            )
        if len(not_used) > 0:
            LGR.warning(
                f"Decision tree {self.tree_name} did not use the following metrics "
                "that were declared as necessary: {not_used}"
            )

    def are_all_components_accepted_or_rejected(self):
        """
        After the tree has finished executing, check if all component
        classifications are either "accepted" or "rejected"
        If any other component classifications remain, log a warning
        """
        component_classifications = set(self.component_table["classification"].to_list())
        nonfinal_classifications = component_classifications.difference({"accepted", "rejected"})
        if nonfinal_classifications:
            for nonfinal_class in nonfinal_classifications:
                numcomp = asarray(self.component_table["classification"] == nonfinal_class).sum()
                LGR.warning(
                    f"{numcomp} components have a final classification of {nonfinal_class}. "
                    "At the end of the selection process, all components are expected "
                    "to be 'accepted' or 'rejected'"
                )

    @property
    def n_comps(self):
        """The number of components in the component table."""
        return len(self.component_table)

    @property
    def n_bold_comps(self):
        """The number of components that are considered bold-weighted."""
        ct = self.component_table
        return len(ct[ct.classification == "accepted"])

    @property
    def accepted_comps(self):
        """The number of components that are accepted."""
        return self.component_table["classification"] == "accepted"

    @property
    def rejected_comps(self):
        """The number of components that are rejected."""
        return self.component_table["classification"] == "rejected"

    @property
    def is_final(self):
        """Whether the classifications are all acccepted/rejected"""
        return (self.accepted_comps.sum() + self.rejected_comps.sum()) > self.n_comps

    @property
    def mixing(self):
        """The mixing matrix used to generate the components being decided upon."""
        return self.mixing_matrix

    @property
    def oc_data(self):
        """The optimally combined data being used for this tree."""
        return self.oc_data

    def to_files(self, io_generator):
        """Convert this selector into component files

        Parameters
        ----------
        io_generator: tedana.io.OutputGenerator
            The output generator to use for filename generation and saving.
        """
        io_generator.save_file(self.component_table, "ICA metrics tsv")
        io_generator.save_file(
            self.cross_component_metrics,
            "ICA cross component metrics json",
        )
        io_generator.save_file(self.component_status_table, "ICA status table tsv")
        io_generator.save_file(self.tree, "ICA decision tree json")
