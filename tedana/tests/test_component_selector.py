"""Tests for the decision tree modularization."""

import glob
import json
import logging
import os
import os.path as op

import pandas as pd
import pytest

from tedana.selection import component_selector
from tedana.utils import get_resource_path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LGR = logging.getLogger("GENERAL")


# ----------------------------------------------------------------------
# Functions Used For Tests
# ----------------------------------------------------------------------


def sample_comptable():
    """Retrieves a sample component table."""
    sample_fname = op.join(THIS_DIR, "data", "sample_comptable.tsv")

    return pd.read_csv(sample_fname, delimiter="\t")


def dicts_to_test(treechoice):
    """
    Outputs decision tree dictionaries to use to test tree validation.

    Parameters
    ----------
    treechoice : :obj:`str` One of several labels to select which dict to output
        Options are:
        "valid": A tree that would trigger all warnings, but pass validation
        "extra_req_param": A tree with an undefined required parameter for a decision node function
        "extra_opt_param": A tree with an undefined optional parameter for a decision node function
        "missing_req_param": A missing required param in a decision node function
        "missing_function": An undefined decision node function
        "missing_key": A dict missing one of the required keys (report)
        "null_value": A parameter in one node improperly has a null value
        "external_missing_key": external_regressors_config missing a required key
        "external_invalid_statistic": external_regressors_config statistic is not "F"

    Returns
    -------
    tree : :ojb:`dict` A dict that can be input into component_selector.validate_tree
    """
    # valid_dict is a simple valid dictionary to test
    # It includes a few things that should trigger warnings, but not errors.
    valid_dict = {
        "tree_id": "valid_simple_tree",
        "info": "This is a short valid tree",
        "report": "",
        # Warning for an unused key
        "unused_key": "There can be added keys that are valid, but aren't used",
        "necessary_metrics": ["kappa", "rho"],
        "intermediate_classifications": ["random1"],
        "classification_tags": ["Random1"],
        "external_regressor_config": [
            {
                "regress_ID": "nuisance",
                "info": (
                    "Fits all external nuisance regressors to "
                    "a single model using an F statistic"
                ),
                "report": (
                    "External nuisance regressors that fit to "
                    "components using a linear model were rejected."
                ),
                "detrend": True,
                "statistic": "F",
                "regressors": ["^(?!signal).*$"],
                "partial_models": {"Motion": ["^mot_.*$"], "CSF": ["^csf.*$"]},
            },
            {
                "regress_ID": "task",
                "info": "Fits all task regressors to a single model using an F statistic",
                "report": (
                    "Task regressors that fit to components using a linear model and "
                    "have some T2* weighting were accepted even if they would have "
                    "been rejected base on other criteriea."
                ),
                "detrend": True,
                "statistic": "F",
                "regressors": ["^signal.*$"],
                "extra field": 42,
            },
        ],
        "nodes": [
            {
                "functionname": "dec_left_op_right",
                "parameters": {
                    "if_true": "rejected",
                    "if_false": "nochange",
                    "decide_comps": "all",
                    "op": ">",
                    "left": "rho",
                    "right": "kappa",
                },
                "kwargs": {
                    "log_extra_info": "random1 if Kappa<Rho",
                    "tag_if_true": "random1",
                },
            },
            {
                "functionname": "dec_left_op_right",
                "parameters": {
                    "if_true": "random2",
                    "if_false": "nochange",
                    "decide_comps": "all",
                    "op": ">",
                    "left": "kappa",
                    "right": "rho",
                },
                "kwargs": {
                    "log_extra_info": "random2 if Kappa>Rho",
                    # Warning for an non-predefined classification assigned to a component
                    "tag_if_true": "random2notpredefined",
                },
            },
            {
                "functionname": "manual_classify",
                "parameters": {
                    "new_classification": "accepted",
                    # Warning for an non-predefined classification used to select
                    # components to operate on
                    "decide_comps": "random2notpredefined",
                },
                "kwargs": {
                    "log_extra_info": "",
                    # Warning for a tag that wasn't predefined
                    "tag": "Random2_NotPredefined",
                },
            },
            {
                "functionname": "manual_classify",
                "parameters": {
                    "new_classification": "rejected",
                    "decide_comps": "random1",
                },
                "kwargs": {
                    "tag": "Random1",
                    # log_extra_report was removed from the code.
                    # If someone runs a tree that uses this field, rather than crash
                    # it will log a warning
                    "log_extra_report": "This should not be logged",
                },
            },
        ],
    }

    tree = valid_dict
    if treechoice == "valid":
        return tree
    elif treechoice == "extra_req_param":
        tree["nodes"][0]["parameters"]["nonexistent_req_param"] = True
    elif treechoice == "extra_opt_param":
        tree["nodes"][0]["kwargs"]["nonexistent_opt_param"] = True
    elif treechoice == "missing_req_param":
        tree["nodes"][0]["parameters"].pop("op")
    elif treechoice == "missing_function":
        tree["nodes"][0]["functionname"] = "not_a_function"
    elif treechoice == "missing_key":
        tree.pop("report")
    elif treechoice == "null_value":
        tree["nodes"][0]["parameters"]["left"] = None
    elif treechoice == "external_missing_key":
        tree["external_regressor_config"][1].pop("statistic")
    elif treechoice == "external_invalid_statistic":
        # Will also trigger statistic isn't F and partial models exist
        tree["external_regressor_config"][0]["statistic"] = "corr"
    else:
        raise Exception(f"{treechoice} is an invalid option for treechoice")

    return tree


# ----------------------------------------------------------------------
# component_selector Tests
# ----------------------------------------------------------------------


# load_config
# -----------
def test_load_config_fails():
    """Tests for load_config failure modes."""

    # We recast to ValueError in the file not found and directory cases
    with pytest.raises(ValueError):
        component_selector.load_config("THIS FILE DOES NOT EXIST.txt")

    # Raises IsADirectoryError for a directory
    with pytest.raises(ValueError):
        component_selector.load_config(".")

    # Note: we defer validation errors for validate_tree even though
    # load_config may raise them


def test_load_config_succeeds():
    """Tests to make sure load_config succeeds."""

    # The minimal tree should have an id of "minimal_decision_tree"
    tree = component_selector.load_config("minimal")
    assert tree["tree_id"] == "minimal_decision_tree"

    # Load the meica tree as a json file rather than just the label
    fname = op.join(get_resource_path(), "decision_trees", "meica.json")
    tree = component_selector.load_config(fname)
    assert tree["tree_id"] == "MEICA_decision_tree"

    # If "kundu" is used as a tree, it should log a warning and output the tedana_orig tree
    tree = component_selector.load_config("kundu")
    assert tree["tree_id"] == "tedana_orig_decision_tree"


def test_minimal():
    """Smoke test for constructor for ComponentSelector using minimal tree."""
    xcomp = {
        "n_echos": 3,
    }
    selector = component_selector.ComponentSelector(tree="minimal")
    selector.select(component_table=sample_comptable(), cross_component_metrics=xcomp.copy())

    # rerun without classification_tags column initialized
    selector = component_selector.ComponentSelector(tree="minimal")
    temp_comptable = sample_comptable().drop(columns="classification_tags")
    selector.select(component_table=temp_comptable, cross_component_metrics=xcomp.copy())


def test_add_manual():
    """Test to confirm add_manual correctly adds a node to tree."""
    xcomp = {
        "n_echos": 3,
    }
    selector = component_selector.ComponentSelector(tree="minimal")
    selector.select(component_table=sample_comptable(), cross_component_metrics=xcomp.copy())
    # Add three manual_classify nodes to the end of the tree with different properties
    selector.add_manual([0, 1, 2], "accept", classification_tags="perfect")
    selector.add_manual([3, 4, 5], "accept", classification_tags="")
    selector.add_manual([6, 7, 8], "accept")
    assert selector.tree["nodes"][-3]["functionname"] == "manual_classify"
    assert selector.tree["nodes"][-3]["parameters"]["decide_comps"] == [0, 1, 2]
    assert selector.tree["nodes"][-3]["kwargs"]["tag"] == "perfect"
    assert selector.tree["nodes"][-2]["parameters"]["decide_comps"] == [3, 4, 5]
    assert selector.tree["nodes"][-2]["kwargs"]["tag"] == "manual reclassify"
    assert selector.tree["nodes"][-1]["parameters"]["decide_comps"] == [6, 7, 8]
    assert selector.tree["nodes"][-1]["kwargs"]["tag"] == "manual reclassify"


# validate_tree
# -------------


def test_validate_tree_succeeds():
    """
    Tests to make sure validate_tree suceeds for all default decision trees.

    Tested on all default trees in ./tedana/resources/decision_trees
    Note: If there is a tree in the default trees directory that
    is being developed and not yet valid, it's file name should
    begin with 'X'.
    """

    default_tree_names = glob.glob(
        os.path.join(f"{THIS_DIR}/../resources/decision_trees/", "[!X]*.json")
    )

    for tree_name in default_tree_names:
        f = open(tree_name)
        tree = json.load(f)
        assert component_selector.validate_tree(tree)

        # Test a few extra possabilities just using the minimal.json tree
        if "/minimal.json" in tree_name:
            # Need to test handling of the tag_if_false kwarg somewhere
            tree["nodes"][1]["kwargs"]["tag_if_false"] = "testing tag"
            assert component_selector.validate_tree(tree)


def test_validate_tree_warnings(caplog):
    """Test to make sure validate_tree triggers all warning conditions."""

    # A tree that raises all possible warnings in the validator should still be valid
    assert component_selector.validate_tree(dicts_to_test("valid"))

    assert (
        r"Decision tree includes fields that are not used or logged ['unused_key']" in caplog.text
    )
    assert (
        r"['random1'] in node 0 of the decision tree includes "
        "a classification tag that was not predefined"
    ) in caplog.text
    assert (
        r"['nochange', 'random2'] in node 1 of the decision tree includes a classification"
        in caplog.text
    )
    assert (
        r"['random2notpredefined'] in node 1 of the decision tree "
        "includes a classification tag that was not predefined"
    ) in caplog.text
    assert (
        r"['random2notpredefined'] in node 2 of the decision tree includes "
        "a classification label that was not predefined"
    ) in caplog.text
    assert (
        r"['Random2_NotPredefined'] in node 2 of the decision tree "
        "includes a classification tag that was not predefined"
    ) in caplog.text
    assert (r"Node 3 includes the 'log_extra_report' parameter.") in caplog.text
    assert (
        "External regressor dictionary 1 includes fields "
        r"that are not used or logged ['extra field']"
    ) in caplog.text


def test_validate_tree_fails():
    """Test to make sure validate_tree fails for invalid trees.

    Tests ../resources/decision_trees/invalid*.json and
    ./data/ComponentSelection/invalid*.json trees.
    """

    # An empty dict should not be valid
    with pytest.raises(
        component_selector.TreeError, match="Decision tree missing required fields"
    ):
        component_selector.validate_tree({})
    # A tree that is missing a required key should not be valid
    with pytest.raises(
        component_selector.TreeError, match=r"Decision tree missing required fields: {'report'}"
    ):
        component_selector.validate_tree(dicts_to_test("missing_key"))
    # Calling a selection node function that does not exist should not be valid
    with pytest.raises(
        component_selector.TreeError,
        match=r"Node 0 has invalid functionname parameter: not_a_function",
    ):
        component_selector.validate_tree(dicts_to_test("missing_function"))

    # Calling a function with an non-existent required parameter should not be valid
    with pytest.raises(
        component_selector.TreeError,
        match=r"Node 0 has additional, undefined required parameters: {'nonexistent_req_param'}",
    ):
        component_selector.validate_tree(dicts_to_test("extra_req_param"))

    # Calling a function with an non-existent optional parameter should not be valid
    with pytest.raises(
        component_selector.TreeError, match=r"Node 0 has additional, undefined optional parameters"
    ):
        component_selector.validate_tree(dicts_to_test("extra_opt_param"))

    # Calling a function missing a required parameter should not be valid
    with pytest.raises(
        component_selector.TreeError, match=r"Node 0 is missing required parameter"
    ):
        component_selector.validate_tree(dicts_to_test("missing_req_param"))

    with pytest.raises(
        component_selector.TreeError,
        match=r"External regressor dictionary 1 is missing required fields: {'statistic'}",
    ):
        component_selector.validate_tree(dicts_to_test("external_missing_key"))

    with pytest.raises(
        component_selector.TreeError,
        match=(
            "statistic in external_regressor_config 1 is corr. It must be one of the following: "
            "{'f'}\nExternal regressor dictionary cannot include partial_models "
            "if statistic is not F"
        ),
    ):
        component_selector.validate_tree(dicts_to_test("external_invalid_statistic"))


def test_check_null_fails():
    """Tests to trigger check_null missing parameter error."""

    selector = component_selector.ComponentSelector(tree="minimal")
    selector.tree = dicts_to_test("null_value")

    params = selector.tree["nodes"][0]["parameters"]
    functionname = selector.tree["nodes"][0]["functionname"]
    with pytest.raises(ValueError):
        selector.check_null(params, functionname)


def test_check_null_succeeds():
    """Tests check_null finds empty parameter in self."""
    selector = component_selector.ComponentSelector(tree="minimal")
    selector.tree = dicts_to_test("null_value")

    # "left" is missing from the function definition in node
    # but is found as an initialized cross component metric
    # so this should execute successfully
    selector.cross_component_metrics_ = {
        "left": 3,
    }

    params = selector.tree["nodes"][0]["parameters"]
    functionname = selector.tree["nodes"][0]["functionname"]
    selector.check_null(params, functionname)


def test_are_only_necessary_metrics_used_warning():
    """Tests a warning that wasn't triggered in other test workflows."""
    selector = component_selector.ComponentSelector(tree="minimal")
    # selector.select(component_table=sample_comptable())

    # warning when an element of necessary_metrics was not in used_metrics
    selector.tree["used_metrics"] = {"A", "B", "C"}
    selector.necessary_metrics = {"B", "C", "D"}
    selector.are_only_necessary_metrics_used()


def test_are_all_components_accepted_or_rejected():
    """Tests warnings are triggered in are_all_components_accepted_or_rejected."""
    selector = component_selector.ComponentSelector(tree="minimal")
    selector.select(component_table=sample_comptable(), cross_component_metrics={"n_echos": 3})
    selector.component_table_.loc[7, "classification"] = "intermediate1"
    selector.component_table_.loc[[1, 3, 5], "classification"] = "intermediate2"
    selector.are_all_components_accepted_or_rejected()


def test_selector_properties_smoke():
    """Tests to confirm properties match expected results."""

    # Runs on un-executed component table to smoke test three class
    # functions that are used to count various types of component
    # classifications in the component table
    selector = component_selector.ComponentSelector(tree="minimal")
    selector.component_table_ = sample_comptable()

    assert selector.n_comps_ == 21

    # Also runs selector.likely_bold_comps_ and should need to deal with sets in each field
    assert selector.n_likely_bold_comps_ == 17

    assert selector.n_accepted_comps_ == 17

    assert selector.rejected_comps_.sum() == 4
