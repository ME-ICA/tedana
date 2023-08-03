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
    """
    Passes errors that are raised when `validate_tree` fails
    """

    pass


def load_config(tree):
    """Load the json file with the decision tree and validate the fields in the decision tree.

    Parameters
    ----------
    tree : :obj:`str`
        The named tree or path to a JSON file that defines one

    Returns
    -------
    tree : :obj:`dict`
        A validated decision tree for the component selection process.
    """

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
    """Confirm that provided `tree` is a valid decision tree.

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
    # Make sure some fields don't trigger a warning; hacky, sorry
    ok_to_not_use = (
        "reconstruct_from",
        "generated_metrics",
    )
    for k in ok_to_not_use:
        if k in unused_keys:
            unused_keys.remove(k)
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
            err_msg += f"Node {i} has invalid functionname parameter: {node.get('functionname')}\n"
            continue

        # Get a functions parameters and compare to parameters defined in the tree
        pos = set([p for p, i in sig.parameters.items() if i.default is inspect.Parameter.empty])
        kwargs = set(sig.parameters.keys()) - pos

        missing_pos = pos - set(node.get("parameters").keys()) - defaults
        if len(missing_pos) > 0:
            err_msg += f"Node {i} is missing required parameter(s): {missing_pos}\n"

        invalid_params = set(node.get("parameters").keys()) - pos
        if len(invalid_params) > 0:
            err_msg += (
                f"Node {i} has additional, undefined required parameters: {invalid_params}\n"
            )

        # Only if kwargs are inputted, make sure they are all valid
        if node.get("kwargs") is not None:
            invalid_kwargs = set(node.get("kwargs").keys()) - kwargs
            if len(invalid_kwargs) > 0:
                err_msg += (
                    f"Node {i} has additional, undefined optional parameters (kwargs): "
                    f"{invalid_kwargs}\n"
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
        if "if_true" in node.get("parameters").keys():
            tmp_comp = node["parameters"]["if_true"]
            if isinstance(tmp_comp, str):
                tmp_comp = [tmp_comp]
            compclass = compclass | set(tmp_comp)
        if "if_false" in node.get("parameters").keys():
            tmp_comp = node["parameters"]["if_false"]
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
            if "tag_if_true" in node.get("kwargs").keys():
                tagset.update(set([node["kwargs"]["tag_if_true"]]))
            if "tag_if_false" in node.get("kwargs").keys():
                tagset.update(set([node["kwargs"]["tag_if_false"]]))
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
    """Load and classify components based on a specified ``tree``."""

    def __init__(self, tree, component_table, cross_component_metrics={}, status_table=None):
        """Initialize the class using the info specified in the json file ``tree``.

        Parameters
        ----------
        tree : :obj:`str`
            The named tree or path to a JSON file that defines one.
        component_table : (C x M) :obj:`pandas.DataFrame`
            Component metric table. One row for each component, with a column for
            each metric; the index should be the component number.
        cross_component_metrics : :obj:`dict`
            Metrics that are each a single value calculated across components.
            Default is empty dictionary.
        status_table : :obj:`pandas.DataFrame`
            A table tracking the status of each component at each step.
            Pass a status table if running additional steps on a decision
            tree that was already executed. Default=None.

        Notes
        -----
        Initializing the  ``ComponentSelector`` confirms tree is valid and
        loads all information in the tree json file into ``ComponentSelector``.

        Adds to the ``ComponentSelector``:

        - component_status_table: empty dataframe or contents of inputted status_table
        - cross_component_metrics: empty dict or contents of inputed values
        - used_metrics: empty set

        Any parameter that is used by a decision tree node function can be passed
        as a parameter in the ``ComponentSelector`` initialization or can be
        included in the json file that defines the decision tree.
        If a parameter is set in the json file, that will take precedence.
        As a style rule, a parameter that is the same regardless of the inputted data should be
        defined in the decision tree json file.
        A parameter that is dataset-specific should be passed through the initialization function.
        Dataset-specific parameters that may need to be passed during initialization include:

        n_echos : :obj:`int`
            Number of echos in multi-echo fMRI data.
            Required for kundu and minimal trees
        n_vols : :obj:`int`
            Number of volumes (time points) in the fMRI data
            Required for kundu tree

        An example initialization with these options would look like
        ``selector = ComponentSelector(tree, comptable, n_echos=n_echos, n_vols=n_vols)``
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
            LGR.info(f"Start is {self.start_idx}")
            self.component_status_table = status_table

    def select(self):
        """Apply the decision tree to data.

        Using the validated tree in ``ComponentSelector`` to run the decision
        tree functions to calculate cross_component metrics and classify
        each component as accepted or rejected.

        Notes
        -------
        The selection process uses previously calculated parameters stored in
        `component_table` for each ICA component such as Kappa (a T2* weighting metric),
        Rho (an S0 weighting metric), and variance explained. If a necessary metric
        is not calculated, this will not run. See `tedana.metrics` for more detail on
        the calculated metrics

        This can be used on a component_table with no component classifications or to alter
        classifications on a component_table that was already run (i.e. for manual
        classificaiton changes after visual inspection)

        When this is run, multiple elements in `ComponentSelector` will change including:

        - component_table: ``classification`` column with ``accepted`` or ``rejected`` labels
          and ``classification_tags`` column with can hold multiple comma-separated labels
          explaining why a classification happened
        - cross_component_metrics: Any values that were calculated based on the metric
          values across components or by direct user input
        - component_status_table: Contains the classification statuses at each node in
          the decision tree
        - used_metrics: A list of metrics used in the selection process
        - nodes: The original tree definition with an added ``outputs`` key listing
          everything that changed in each node
        - current_node_idx: The total number of nodes run in ``ComponentSelector``
        """

        if "classification_tags" not in self.component_table.columns:
            self.component_table["classification_tags"] = ""

        # this will crash the program with an error message if not all
        # necessary_metrics are in the comptable
        confirm_metrics_exist(
            self.component_table, self.necessary_metrics, function_name=self.tree_name
        )

        # for each node in the decision tree
        for self.current_node_idx, node in enumerate(
            self.tree["nodes"][self.start_idx :], start=self.start_idx
        ):
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

            LGR.debug(
                f"Step {self.current_node_idx}: Running function {node['functionname']} "
                f"with parameters: {all_params}"
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
            LGR.debug(
                f"Step {self.current_node_idx} Full outputs: "
                f"{self.tree['nodes'][self.current_node_idx]['outputs']}"
            )

        # move decision columns to end
        self.component_table = clean_dataframe(self.component_table)
        # warning anything called a necessary metric wasn't used and if
        # anything not called a necessary metric was used
        self.are_only_necessary_metrics_used()

        self.are_all_components_accepted_or_rejected()

    def add_manual(self, indices, classification):
        """Add nodes that will manually classify components.

        Parameters
        ----------
        indices : :obj:`list[int]`
            The indices to manually classify
        classification : :obj:`str`
            The classification to set the nodes to (i.e. accepted or rejected)
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
                    "tag": "manual reclassify",
                },
            }
        )

    def check_null(self, params, fcn):
        """
        Checks that all required parameters for selection node functions are
        attributes in the class. Error if any are undefined

        Returns
        -------
        params: :obj:`dict`
            The keys and values for the inputted parameters
        """

        for key, val in params.items():
            if val is None:
                try:
                    params[key] = getattr(self, key)
                except AttributeError:
                    raise ValueError(
                        f"Parameter {key} is required in node {fcn}, but not defined. "
                        f"If {key} is dataset specific, it should be "
                        "defined in the initialization of "
                        "ComponentSelector. If it is fixed regardless of dataset, it "
                        "should be defined in the json file that defines the "
                        "decision tree."
                    )

        return params

    def are_only_necessary_metrics_used(self):
        """
        Check if all metrics that are declared as necessary are actually
        used and if any used_metrics weren't explicitly declared necessary.
        If either of these happen, a warning is added to the logger
        """
        necessary_metrics = self.necessary_metrics
        not_declared = self.tree["used_metrics"] - necessary_metrics
        not_used = necessary_metrics - self.tree["used_metrics"]
        if len(not_declared) > 0:
            LGR.warning(
                f"Decision tree {self.tree_name} used the following metrics that were "
                f"not declared as necessary: {not_declared}"
            )
        if len(not_used) > 0:
            LGR.warning(
                f"Decision tree {self.tree_name} did not use the following metrics "
                f"that were declared as necessary: {not_used}"
            )

    def are_all_components_accepted_or_rejected(self):
        """
        After the tree has finished executing, check if all component
        classifications are either "accepted" or "rejected".
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
    def likely_bold_comps(self):
        """A boolean :obj:`pandas.Series` of components that are tagged "Likely BOLD"."""
        likely_bold_comps = self.component_table["classification_tags"].copy()
        for idx in range(len(likely_bold_comps)):
            if "Likely BOLD" in likely_bold_comps.loc[idx]:
                likely_bold_comps.loc[idx] = True
            else:
                likely_bold_comps.loc[idx] = False
        return likely_bold_comps

    @property
    def n_likely_bold_comps(self):
        """The number of components that are tagged "Likely BOLD"."""
        return self.likely_bold_comps.sum()

    @property
    def accepted_comps(self):
        """A boolean :obj:`pandas.Series` of components that are accepted."""
        return self.component_table["classification"] == "accepted"

    @property
    def n_accepted_comps(self):
        """The number of components that are accepted."""
        return self.accepted_comps.sum()

    @property
    def rejected_comps(self):
        """A boolean :obj:`pandas.Series` of components that are rejected."""
        return self.component_table["classification"] == "rejected"

    def to_files(self, io_generator):
        """Convert this selector into component files.

        Parameters
        ----------
        io_generator : :obj:`tedana.io.OutputGenerator`
            The output generator to use for filename generation and saving.
        """
        io_generator.save_file(self.component_table, "ICA metrics tsv")
        io_generator.save_file(
            self.cross_component_metrics,
            "ICA cross component metrics json",
        )
        io_generator.save_file(self.component_status_table, "ICA status table tsv")
        io_generator.save_file(self.tree, "ICA decision tree json")
