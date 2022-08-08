"""Tests for the decision tree modularization"""
import glob
import json
import os
import os.path as op

import pandas as pd
import pytest

from tedana.selection import ComponentSelector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Functions Used For Tests
# ----------------------------------------------------------------------


def sample_comptable():
    """Retrieves a sample component table"""
    sample_fname = op.join(THIS_DIR, "data", "sample_comptable.tsv")

    return pd.read_csv(sample_fname, delimiter="\t")


def dicts_to_test(treechoice):
    """
    Outputs decision tree dictionaries to use to test tree validation

    Parameters
    ----------
    treechoice: :obj:`str` One of several labels to select which dict to output
        Options are:
        "valid": A tree that would trigger all warnings, but pass validation
        "extra_req_param": A tree with an undefined required parameter for a decision node function
        "extra_opt_param": A tree with an undefined optional parameter for a decision node function
        "missing_req_param": A missing required param in a decision node function
        "missing_function": An undefined decision node function
        "missing_key": A dict missing one of the required keys (refs)

    Returns
    -------
    tree: :ojb:`dict` A dict that can be input into ComponentSelector.validate_tree
    """

    # valid_dict is a simple valid dictionary to test
    # It includes a few things that should trigger warnings, but not errors.
    valid_dict = {
        "tree_id": "valid_simple_tree",
        "info": "This is a short valid tree",
        "report": "",
        "refs": "",
        # Warning for an unused key
        "unused_key": "There can be added keys that are valid, but aren't used",
        "necessary_metrics": ["kappa", "rho"],
        "intermediate_classifications": ["random1"],
        "classification_tags": ["Random1"],
        "nodes": [
            {
                "functionname": "dec_left_op_right",
                "parameters": {
                    "ifTrue": "rejected",
                    "ifFalse": "nochange",
                    "decide_comps": "all",
                    "op": ">",
                    "left": "rho",
                    "right": "kappa",
                },
                "kwargs": {
                    "log_extra_info": "random1 if Kappa<Rho",
                    "tag_ifTrue": "random1",
                },
            },
            {
                "functionname": "dec_left_op_right",
                "parameters": {
                    "ifTrue": "random2",
                    "ifFalse": "nochange",
                    "decide_comps": "all",
                    "op": ">",
                    "left": "kappa",
                    "right": "rho",
                },
                "kwargs": {
                    "log_extra_info": "random2 if Kappa>Rho",
                    "log_extra_report": "",
                    # Warning for an non-predefined classification assigned to a component
                    "tag_ifTrue": "random2notpredefined",
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
                    "log_extra_report": "",
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
        tree.pop("refs")
    else:
        raise Exception(f"{treechoice} is an invalid option for treechoice")

    return tree


# ----------------------------------------------------------------------
# ComponentSelector Tests
# ----------------------------------------------------------------------

# load_config
# -----------
def test_load_config_fails():
    """Tests for load_config failure modes"""

    # We recast to ValueError in the file not found and directory cases
    with pytest.raises(ValueError):
        ComponentSelector.load_config("THIS FILE DOES NOT EXIST.txt")

    # Raises IsADirectoryError for a directory
    with pytest.raises(ValueError):
        ComponentSelector.load_config(".")

    # Note: we defer validation errors for validate_tree even though
    # load_config may raise them


def test_load_config_succeeds():
    """Tests to make sure load_config succeeds"""

    # The minimal tree should have an id of "minimal_decision_tree_test1"
    tree = ComponentSelector.load_config("minimal")
    assert tree["tree_id"] == "minimal_decision_tree_test1"


def test_minimal():
    """Smoke test for constructor for ComponentSelector using minimal tree"""
    xcomp = {
        "n_echos": 3,
    }
    tree = ComponentSelector.ComponentSelector(
        "minimal",
        sample_comptable(),
        cross_component_metrics=xcomp,
    )
    tree.select()


# validate_tree
# -------------


def test_validate_tree_succeeds():
    """
    Tests to make sure validate_tree suceeds for all default
    decision trees in  decision trees
    Tested on all default trees in ./tedana/resources/decision_trees
    Note: If there is a tree in the default trees directory that
    is being developed and not yet valid, it's file name should
    include 'invalid' as a prefix
    """

    default_tree_names = glob.glob(
        os.path.join(THIS_DIR, "../resources/decision_trees/[!invalid]*.json")
    )

    for tree_name in default_tree_names:
        f = open(tree_name)
        tree = json.load(f)
        assert ComponentSelector.validate_tree(tree)

        # Test a few extra possabilities just using the minimal.json tree
        if "/minimal.json" in tree_name:
            # Should remove/ignore the "reconstruct_from" key during validation
            tree["reconstruct_from"] = "testinput"
            # Need to test handling of the tag_ifFalse kwarg somewhere
            tree["nodes"][1]["kwargs"]["tag_ifFalse"] = "testing tag"
            assert ComponentSelector.validate_tree(tree)


def test_validate_tree_warnings():
    """
    Tests to make sure validate_tree triggers all warning conditions
    but still succeeds
    """

    # A tree that raises all possible warnings in the validator should still be valid
    assert ComponentSelector.validate_tree(dicts_to_test("valid"))


def test_validate_tree_fails():
    """
    Tests to make sure validate_tree fails for invalid trees
    Tests ../resources/decision_trees/invalid*.json and
    ./data/ComponentSelection/invalid*.json trees
    """

    # An empty dict should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree({})

    # A tree that is missing a required key should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("missing_key"))

    # Calling a selection node function that does not exist should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("missing_function"))

    # Calling a function with an non-existent required parameter should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("extra_req_param"))

    # Calling a function with an non-existent optional parameter should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("extra_opt_param"))

    # Calling a function missing a required parameter should not be valid
    with pytest.raises(ComponentSelector.TreeError):
        ComponentSelector.validate_tree(dicts_to_test("missing_req_param"))
