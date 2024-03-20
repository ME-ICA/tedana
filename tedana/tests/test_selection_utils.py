"""Tests for the tedana.selection.selection_utils module."""

import os

import numpy as np
import pandas as pd
import pytest

from tedana.selection import selection_utils
from tedana.selection.component_selector import ComponentSelector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_component_table(options=None):
    """
    Retrieves a sample component table.

    Options: Different strings will also the contents of the component table
        'provclass': Change the classifications to "provisional accept" for 4 components
        'unclass': Change 4 classifications to "provisional accept", 2 to accepted,
        2 to rejected, and the rest to "unclassified"
    """

    sample_fname = os.path.join(THIS_DIR, "data", "sample_comptable.tsv")
    component_table = pd.read_csv(sample_fname, delimiter="\t")
    component_table["classification_tags"] = ""
    if options == "unclass":
        component_table["classification"] = "unclassified"
        component_table.loc[[16, 18], "classification"] = "accepted"
        component_table.loc[[11, 13], "classification"] = "rejected"

    if (options == "provclass") or (options == "unclass"):
        component_table.loc[[2, 4, 6, 8], "classification"] = "provisional accept"
    return component_table


def sample_selector(options=None):
    """Retrieve a sample component table and initializes a selector.

    The selector uses that component table and the minimal tree.

    options: Different strings will alter the selector
       'provclass': Change the classifications to "provisional accept" for 4 components
        'unclass': Change 4 classifications to "provisional accept" and the rest to "unclassified"
    """

    tree = "minimal"

    component_table = sample_component_table(options=options)

    xcomp = {
        "n_echos": 3,
        "n_vols": 201,
        "test_elbow": 21,
    }
    selector = ComponentSelector(tree=tree)

    # Add an un-executed component table,cross component metrics, and status table
    selector.component_table_ = component_table.copy()
    selector.cross_component_metrics_ = xcomp
    selector.component_status_table_ = selector.component_table_[
        ["Component", "classification"]
    ].copy()
    selector.component_status_table_ = selector.component_status_table_.rename(
        columns={"classification": "initialized classification"}
    )

    selector.current_node_idx_ = 0

    return selector


##############################################################
# Functions that are used for interacting with component_table
##############################################################


def test_selectcomps2use_succeeds():
    """
    Tests to make sure selectcomps2use runs with full range of inputs.

    Include tests to make sure the correct number of components are selected
    from the pre-defined sample_comptable.tsv component table.
    """
    selector = sample_selector()

    decide_comps_options = [
        "rejected",
        ["accepted"],
        "all",
        ["accepted", "rejected"],
        4,
        [2, 6, 4],
        "NotALabel",
    ]
    # Given the pre-defined comptable in sample_table_selector, these
    #   are the expected number of components that should be selected
    #   for each of the above decide_comps_options
    decide_comps_lengths = [4, 17, 21, 21, 1, 3, 0]

    for idx, decide_comps in enumerate(decide_comps_options):
        comps2use = selection_utils.selectcomps2use(selector.component_table_, decide_comps)
        assert len(comps2use) == decide_comps_lengths[idx], (
            f"selectcomps2use test should select {decide_comps_lengths[idx]} with "
            f"decide_comps={decide_comps}, but it selected {len(comps2use)}"
        )


def test_selectcomps2use_fails():
    """Tests for selectcomps2use failure modes."""
    selector = sample_selector()

    decide_comps_options = [
        18.2,  # no floats
        [11.2, 13.1],  # no list of floats
        ["accepted", 4],  # needs to be either int or string, not both
        [4, 3, -1, 9],  # no index should be < 0
        [2, 4, 6, 21],  # no index should be > number of 0 indexed components
        22,  # no index should be > number of 0 indexed components
    ]
    for decide_comps in decide_comps_options:
        with pytest.raises(ValueError):
            selection_utils.selectcomps2use(selector.component_table_, decide_comps)

    selector.component_table_ = selector.component_table_.drop(columns="classification")
    with pytest.raises(ValueError):
        selection_utils.selectcomps2use(selector.component_table_, "all")


def test_comptable_classification_changer_succeeds():
    """All conditions where comptable_classification_changer should run.

    Note: This confirms the function runs, but not that outputs are accurate.

    Also tests conditions where the warning logger is used, but doesn't
    check the logger.
    """

    def validate_changes(expected_classification):
        # check every element that was supposed to change, did change
        changeidx = decision_boolean.index[np.asarray(decision_boolean) == boolstate]
        new_vals = selector.component_table_.loc[changeidx, "classification"]
        for val in new_vals:
            assert val == expected_classification

    # Change if true
    selector = sample_selector(options="provclass")
    decision_boolean = selector.component_table_["classification"] == "provisional accept"
    boolstate = True
    selector = selection_utils.comptable_classification_changer(
        selector, boolstate, "accepted", decision_boolean, tag_if="testing_tag"
    )
    validate_changes("accepted")

    # Run nochange condition
    selector = sample_selector(options="provclass")
    decision_boolean = selector.component_table_["classification"] == "provisional accept"
    selector = selection_utils.comptable_classification_changer(
        selector, boolstate, "nochange", decision_boolean, tag_if="testing_tag"
    )
    validate_changes("provisional accept")

    # Change if false
    selector = sample_selector(options="provclass")
    decision_boolean = selector.component_table_["classification"] != "provisional accept"
    boolstate = False
    selector = selection_utils.comptable_classification_changer(
        selector, boolstate, "rejected", decision_boolean, tag_if="testing_tag1, testing_tag2"
    )
    validate_changes("rejected")

    # Change from accepted to rejected, which should output a warning
    # (test if the warning appears?)
    selector = sample_selector(options="provclass")
    decision_boolean = selector.component_table_["classification"] == "accepted"
    boolstate = True
    selector = selection_utils.comptable_classification_changer(
        selector, boolstate, "rejected", decision_boolean, tag_if="testing_tag"
    )
    validate_changes("rejected")

    # Change from rejected to accepted and suppress warning
    selector = sample_selector(options="provclass")
    decision_boolean = selector.component_table_["classification"] == "rejected"
    boolstate = True
    selector = selection_utils.comptable_classification_changer(
        selector,
        boolstate,
        "accepted",
        decision_boolean,
        tag_if="testing_tag",
        dont_warn_reclassify=True,
    )
    validate_changes("accepted")


def test_change_comptable_classifications_succeeds():
    """All conditions where change_comptable_classifications should run."""

    selector = sample_selector(options="provclass")

    # Given the rho values in the sample table, decision_boolean should have
    # 2 True and 2 False values
    comps2use = selection_utils.selectcomps2use(selector.component_table_, "provisional accept")
    rho = selector.component_table_.loc[comps2use, "rho"]
    decision_boolean = rho < 13.5

    selector, n_true, n_false = selection_utils.change_comptable_classifications(
        selector,
        "accepted",
        "nochange",
        decision_boolean,
        tag_if_true="testing_tag1",
        tag_if_false="testing_tag2",
    )

    assert n_true == 2
    assert n_false == 2
    # check every element that was supposed to change, did change
    changeidx = decision_boolean.index[np.asarray(decision_boolean) == True]  # noqa: E712
    new_vals = selector.component_table_.loc[changeidx, "classification"]
    for val in new_vals:
        assert val == "accepted"


def test_clean_dataframe_smoke():
    """A smoke test for the clean_dataframe function."""
    component_table = sample_component_table(options="comptable")
    selection_utils.clean_dataframe(component_table)


#################################################
# Functions to validate inputs or log information
#################################################


def test_confirm_metrics_exist_succeeds():
    """Tests confirm_metrics_exist run with correct inputs."""
    component_table = sample_component_table(options="comptable")

    # Testing for metrics that exist with 1 or 2 necessary metrics in a set
    # Returns True if an undefined metric exists so using "assert not"
    assert not selection_utils.confirm_metrics_exist(component_table, {"kappa"})
    assert not selection_utils.confirm_metrics_exist(component_table, {"kappa", "rho"})


def test_confirm_metrics_exist_fails():
    """Tests confirm_metrics_exist for failure conditions."""

    component_table = sample_component_table(options="comptable")

    # Should fail with and error would have default or pre-defined file name
    with pytest.raises(ValueError):
        selection_utils.confirm_metrics_exist(component_table, {"kappa", "quack"})
    with pytest.raises(ValueError):
        selection_utils.confirm_metrics_exist(
            component_table, {"kappa", "mooo"}, function_name="farm"
        )


def test_log_decision_tree_step_smoke():
    """A smoke test for log_decision_tree_step."""

    selector = sample_selector()

    # Standard run for logging classification changes
    comps2use = selection_utils.selectcomps2use(selector.component_table_, "reject")
    selection_utils.log_decision_tree_step(
        "Step 0: test_function_name",
        comps2use,
        decide_comps="reject",
        n_true=5,
        n_false=2,
        if_true="accept",
        if_false="reject",
    )

    # Standard use for logging cross_component_metric calculation
    outputs = {
        "calc_cross_comp_metrics": [
            "kappa_elbow_kundu",
            "rho_elbow_kundu",
        ],
        "kappa_elbow_kundu": 45,
        "rho_elbow_kundu": 12,
    }
    selection_utils.log_decision_tree_step(
        "Step 0: test_function_name", comps2use, calc_outputs=outputs
    )

    # Puts a warning in the logger if outputs doesn't have a cross_component_metrics field
    outputs = {
        "kappa_elbow_kundu": 45,
        "rho_elbow_kundu": 12,
    }
    selection_utils.log_decision_tree_step(
        "Step 0: test_function_name", comps2use, calc_outputs=outputs
    )

    # Logging no components found with a specified classification
    comps2use = selection_utils.selectcomps2use(selector.component_table_, "NotALabel")
    selection_utils.log_decision_tree_step(
        "Step 0: test_function_name",
        comps2use,
        decide_comps="NotALabel",
        n_true=5,
        n_false=2,
        if_true="accept",
        if_false="reject",
    )


def test_log_classification_counts_smoke():
    """A smoke test for log_classification_counts."""

    component_table = sample_component_table(options="comptable")

    selection_utils.log_classification_counts(5, component_table)


#######################################################
# Calculations that are used in decision tree functions
#######################################################


def test_getelbow_smoke():
    """A smoke test for the getelbow function."""
    arr = np.random.random(100)
    idx = selection_utils.getelbow(arr)
    assert isinstance(idx, np.int32) or isinstance(idx, np.int64)

    val = selection_utils.getelbow(arr, return_val=True)
    assert isinstance(val, float)

    # Running an empty array should raise a ValueError
    arr = np.array([])
    with pytest.raises(ValueError):
        selection_utils.getelbow(arr)

    # Running a 2D array should raise a ValueError
    arr = np.random.random((100, 100))
    with pytest.raises(ValueError):
        selection_utils.getelbow(arr)


def test_getelbow_cons_smoke():
    """A smoke test for the getelbow_cons function."""
    arr = np.random.random(100)
    idx = selection_utils.getelbow_cons(arr)
    assert isinstance(idx, np.int32) or isinstance(idx, np.int64)

    val = selection_utils.getelbow_cons(arr, return_val=True)
    assert isinstance(val, float)

    # Running an empty array should raise a ValueError
    arr = np.array([])
    with pytest.raises(ValueError):
        selection_utils.getelbow_cons(arr)

    # Running a 2D array should raise a ValueError
    arr = np.random.random((100, 100))
    with pytest.raises(ValueError):
        selection_utils.getelbow_cons(arr)


def test_kappa_elbow_kundu_smoke():
    """A smoke test for the kappa_elbow_kundu function."""

    component_table = sample_component_table()

    # Normal execution. With n_echoes==5 a few components will be excluded for the nonsig threshold
    (
        kappa_elbow_kundu,
        kappa_allcomps_elbow,
        kappa_nonsig_elbow,
        varex_upper_p,
    ) = selection_utils.kappa_elbow_kundu(component_table, n_echos=5)
    assert isinstance(kappa_elbow_kundu, float)
    assert isinstance(kappa_allcomps_elbow, float)
    assert isinstance(kappa_nonsig_elbow, float)
    assert isinstance(varex_upper_p, float)

    # For the sample component_table, when n_echos=6, there are fewer than 5 components
    #  that are greater than an f01 threshold and a different condition in kappa_elbow_kundu is run
    (
        kappa_elbow_kundu,
        kappa_allcomps_elbow,
        kappa_nonsig_elbow,
        varex_upper_p,
    ) = selection_utils.kappa_elbow_kundu(component_table, n_echos=6)
    assert isinstance(kappa_elbow_kundu, float)
    assert isinstance(kappa_allcomps_elbow, float)
    assert isinstance(kappa_nonsig_elbow, type(None))
    assert isinstance(varex_upper_p, float)

    # Run using only a subset of components
    (
        kappa_elbow_kundu,
        kappa_allcomps_elbow,
        kappa_nonsig_elbow,
        varex_upper_p,
    ) = selection_utils.kappa_elbow_kundu(
        component_table,
        n_echos=5,
        comps2use=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 20],
    )
    assert isinstance(kappa_elbow_kundu, float)
    assert isinstance(kappa_allcomps_elbow, float)
    assert isinstance(kappa_nonsig_elbow, float)
    assert isinstance(varex_upper_p, float)


def test_rho_elbow_kundu_liberal_smoke():
    """A smoke test for the rho_elbow_kundu_liberal function."""

    component_table = sample_component_table(options="unclass")
    # Normal execution with default kundu threshold
    (
        rho_elbow_kundu,
        rho_allcomps_elbow,
        rho_unclassified_elbow,
        elbow_f05,
    ) = selection_utils.rho_elbow_kundu_liberal(component_table, n_echos=3)
    assert isinstance(rho_elbow_kundu, float)
    assert isinstance(rho_allcomps_elbow, float)
    assert isinstance(rho_unclassified_elbow, float)
    assert isinstance(elbow_f05, float)

    # Normal execution with liberal threshold
    (
        rho_elbow_kundu,
        rho_allcomps_elbow,
        rho_unclassified_elbow,
        elbow_f05,
    ) = selection_utils.rho_elbow_kundu_liberal(
        component_table, n_echos=3, rho_elbow_type="liberal"
    )
    assert isinstance(rho_elbow_kundu, float)
    assert isinstance(rho_allcomps_elbow, float)
    assert isinstance(rho_unclassified_elbow, float)
    assert isinstance(elbow_f05, float)

    # Run using only a subset of components
    (
        rho_elbow_kundu,
        rho_allcomps_elbow,
        rho_unclassified_elbow,
        elbow_f05,
    ) = selection_utils.rho_elbow_kundu_liberal(
        component_table,
        n_echos=3,
        rho_elbow_type="kundu",
        comps2use=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 20],
        subset_comps2use=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 18, 20],
    )
    assert isinstance(rho_elbow_kundu, float)
    assert isinstance(rho_allcomps_elbow, float)
    assert isinstance(rho_unclassified_elbow, float)
    assert isinstance(elbow_f05, float)

    # Run with no unclassified components and thus subset_comps2use is empty
    component_table = sample_component_table()
    (
        rho_elbow_kundu,
        rho_allcomps_elbow,
        rho_unclassified_elbow,
        elbow_f05,
    ) = selection_utils.rho_elbow_kundu_liberal(component_table, n_echos=3)
    assert isinstance(rho_elbow_kundu, float)
    assert isinstance(rho_allcomps_elbow, float)
    assert isinstance(rho_unclassified_elbow, type(None))
    assert isinstance(elbow_f05, float)

    with pytest.raises(ValueError):
        selection_utils.rho_elbow_kundu_liberal(
            component_table, n_echos=3, rho_elbow_type="perfect"
        )


def test_get_extend_factor_smoke():
    """A smoke test for get_extend_factor."""

    val = selection_utils.get_extend_factor(extend_factor=int(10))
    assert isinstance(val, float)

    for n_vols in [80, 100, 120]:
        val = selection_utils.get_extend_factor(n_vols=n_vols)
        assert isinstance(val, float)

    with pytest.raises(ValueError):
        selection_utils.get_extend_factor()
