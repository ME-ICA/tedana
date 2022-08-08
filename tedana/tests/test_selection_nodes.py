"""Tests for the tedana.selection.selection_nodes module."""
import os
from re import S

import numpy as np
import pandas as pd
import pytest

from tedana.selection import selection_nodes, selection_utils
from tedana.selection.ComponentSelector import ComponentSelector
from tedana.tests.test_selection_utils import sample_component_table, sample_selector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_manual_classify_smoke():
    """Smoke tests for all options in manual_classify"""

    selector = sample_selector(options="provclass")

    decide_comps = "provisional accept"
    new_classification = "accepted"

    # Outputs just the metrics used in this function (nothing in this case)
    used_metrics = selection_nodes.manual_classify(
        selector, decide_comps, new_classification, only_used_metrics=True
    )
    assert used_metrics == set()

    # Standard execution where components are changed from "provisional accept" to "accepted"
    # And all extra logging code is run
    selector = selection_nodes.manual_classify(
        selector,
        decide_comps,
        new_classification,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag="test tag",
    )
    # There should be 4 selected components and component_status_table should have a new column "Node 0"
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 4
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 0
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    selector.current_node_idx = 1
    selector = selection_nodes.manual_classify(selector, "NotAClassification", new_classification)
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # Changing components from "rejected" to "accepted" and suppressing warning
    selector.current_node_idx = 2
    selector = selection_nodes.manual_classify(
        selector,
        "rejected",
        new_classification,
        clear_classification_tags=True,
        log_extra_report="report log",
        log_extra_info="info log",
        tag="test tag",
        dont_warn_reclassify=True,
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 4
    assert f"Node {selector.current_node_idx}" in selector.component_status_table


def test_dec_left_op_right_succeeds():
    """tests for successful calls to dec_left_op_right"""

    selector = sample_selector(options="provclass")

    decide_comps = "provisional accept"

    # Outputs just the metrics used in this function {"kappa", "rho"}
    used_metrics = selection_nodes.dec_left_op_right(
        selector, "accepted", "rejected", decide_comps, ">", "kappa", "rho", only_used_metrics=True
    )
    assert len(used_metrics - {"kappa", "rho"}) == 0

    # Standard execution where components with kappa>rho are changed from "provisional accept" to "accepted"
    # And all extra logging code and options are run
    # left and right are both component_table_metrics
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        "kappa",
        "rho",
        left_scale=0.9,
        right_scale=1.4,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag_ifTrue="test true tag",
        tag_ifFalse="test false tag",
    )
    # scales are set to make sure 3 components are true and 1 is false using the sample component table
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 3
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 1
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column is created in component_status_table
    selector.current_node_idx = 1
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        "NotAClassification",
        ">",
        "kappa",
        "rho",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # Re-initializing selector so that it has components classificated as "provisional accept" again
    selector = sample_selector(options="provclass")
    # Test when left is a component_table_metric, & right is a cross_component_metric
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        "kappa",
        "test_elbow",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 3
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 1
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # right is a component_table_metric, left is a cross_component_metric
    # left also has a left_scale that's a cross component metric
    selector = sample_selector(options="provclass")
    selector.cross_component_metrics["new_cc_metric"] = 1.02
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        "test_elbow",
        "kappa",
        left_scale="new_cc_metric",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 1
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 3
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # left component_table_metric, right is a constant integer value
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        "kappa",
        21,
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 3
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 1
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # right component_table_metric, left is a constant float value
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        ">",
        21.0,
        "kappa",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 1
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 3
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # Testing combination of two statements. kappa>21 AND rho<14
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        "<",
        21.0,
        "kappa",
        left2="rho",
        op2="<",
        right2=14,
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 2
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 2
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # Testing combination of three statements. kappa>21 AND rho<14 AND 'variance explained'<5
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        "<",
        21.0,
        "kappa",
        left2="rho",
        op2="<",
        right2=14,
        left3="variance explained",
        op3="<",
        right3=5,
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 1
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 3
    assert f"Node {selector.current_node_idx}" in selector.component_status_table


def test_dec_left_op_right_fails():
    """tests for calls to dec_left_op_right that raise errors"""

    selector = sample_selector(options="provclass")
    decide_comps = "provisional accept"

    # Raise error for left value that is not a metric
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            "NotAMetric",
            21,
        )

    # Raise error for right value that is not a metric
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            21,
            "NotAMetric",
        )

    # Raise error for invalid operator
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            "><",
            "kappa",
            21,
        )

    # Raise error for right_scale that is not a number
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selector = selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            21.0,
            "kappa",
            right_scale="NotANumber",
        )

    # Raise error for right_scale that a column in the component_table
    #  which isn't allowed since the scale value needs to resolve to a
    #  a fixed number and not a different number for each component
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selector = selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            21.0,
            "kappa",
            right_scale="rho",
        )

    # Raise error if some but not all parameters for the second conditional statement are defined
    #  In this case, op2 is not defined
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            "kappa",
            21,
            left2="rho",
            right2=14,
        )

    # Raise error for invalid operator for op2
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            "kappa",
            21,
            left2="rho",
            op2="<>",
            right2=14,
        )

    # Raise error if some but not all parameters for the third conditional statement are defined
    #  In this case, op3 is not defined
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            "kappa",
            21,
            left2="rho",
            right2=14,
            op2="<",
            left3="variance explained",
            right3=5,
        )

    # Raise error if there's a third conditional statement but not a second statement
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selection_nodes.dec_left_op_right(
            selector,
            "accepted",
            "rejected",
            decide_comps,
            ">",
            "kappa",
            21,
            left3="variance explained",
            right3=5,
            op3="<",
        )


def test_dec_variance_lessthan_thresholds_smoke():
    """Smoke tests for dec_variance_lessthan_thresholds"""

    selector = sample_selector(options="provclass")
    decide_comps = "provisional accept"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.dec_variance_lessthan_thresholds(
        selector, "accepted", "rejected", decide_comps, only_used_metrics=True
    )
    assert len(used_metrics - {"variance explained"}) == 0

    # Standard execution where with all extra logging code and options changed from defaults
    selector = selection_nodes.dec_variance_lessthan_thresholds(
        selector,
        "accepted",
        "rejected",
        decide_comps,
        var_metric="normalized variance explained",
        single_comp_threshold=0.05,
        all_comp_threshold=0.09,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag_ifTrue="test true tag",
        tag_ifFalse="test false tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 1
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 3
    assert f"Node {selector.current_node_idx}" in selector.component_status_table

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    selector.current_node_idx = 1
    selector = selection_nodes.dec_variance_lessthan_thresholds(
        selector, "accepted", "rejected", "NotAClassification"
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # Running without specifying logging text generates internal text
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_variance_lessthan_thresholds(
        selector, "accepted", "rejected", decide_comps
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 4
    assert f"Node {selector.current_node_idx}" in selector.component_status_table


def test_calc_kappa_rho_elbows_kundu():
    """Smoke tests for calc_kappa_rho_elbows_kundu"""

    # Standard use of this function requires some components to be "unclassified"
    selector = sample_selector(options="unclass")
    decide_comps = "all"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.calc_kappa_rho_elbows_kundu(
        selector, decide_comps, only_used_metrics=True
    )
    assert len(used_metrics - {"kappa", "rho"}) == 0

    # Standard call to this function.
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(
        selector,
        decide_comps,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"kappa_elbow_kundu", "rho_elbow_kundu", "varex_upper_p"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["rho_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] > 0

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector(options="unclass")
    selector.cross_component_metrics["kappa_elbow_kundu"] = 1
    selector.cross_component_metrics["rho_elbow_kundu"] = 1
    selector.cross_component_metrics["varex_upper_p"] = 1
    decide_comps = "all"
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(
        selector,
        decide_comps,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.cross_component_metrics["kappa_elbow_kundu"] > 2
    assert selector.cross_component_metrics["rho_elbow_kundu"] > 2
    assert selector.cross_component_metrics["varex_upper_p"] > 2

    # Run with kappa_only==True
    selector = sample_selector(options="unclass")
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(selector, decide_comps, kappa_only=True)
    calc_cross_comp_metrics = {"kappa_elbow_kundu", "varex_upper_p"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_elbow_kundu"] > 0
    assert "rho_elbow_kundu" not in selector.tree["nodes"][selector.current_node_idx]["outputs"]
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] > 0

    # Run with rho_only==True
    selector = sample_selector(options="unclass")
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(selector, decide_comps, rho_only=True)
    calc_cross_comp_metrics = {"rho_elbow_kundu", "varex_upper_p"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["rho_elbow_kundu"] > 0
    assert "kappa_elbow_kundu" not in selector.tree["nodes"][selector.current_node_idx]["outputs"]
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] > 0

    # Should run normally with both kappa_only and rho_only==True
    selector = sample_selector(options="unclass")
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(
        selector, decide_comps, kappa_only=True, rho_only=True
    )
    calc_cross_comp_metrics = {"kappa_elbow_kundu", "rho_elbow_kundu", "varex_upper_p"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["rho_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] > 0

    # Log without running if no components of class decide_comps or no components
    #  classified as "unclassified" are in the component table
    selector = sample_selector()
    selector = selection_nodes.calc_kappa_rho_elbows_kundu(selector, "NotAClassification")
    calc_cross_comp_metrics = {"kappa_elbow_kundu", "rho_elbow_kundu", "varex_upper_p"}
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_elbow_kundu"] == None
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["rho_elbow_kundu"] == None
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_p"] == None


def test_dec_classification_doesnt_exist_smoke():
    """Smoke tests for dec_classification_doesnt_exist"""

    selector = sample_selector(options="unclass")
    decide_comps = ["unclassified", "provisional accept"]

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.dec_classification_doesnt_exist(
        selector,
        "rejected",
        decide_comps,
        class_comp_exists="provisional accept",
        only_used_metrics=True,
    )
    assert len(used_metrics) == 0

    # Standard execution where with all extra logging code and options changed from defaults
    selector = selection_nodes.dec_classification_doesnt_exist(
        selector,
        "accepted",
        decide_comps,
        class_comp_exists="provisional accept",
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag_ifTrue="test true tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    # Lists the number of components in decide_comps in numFalse
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 17
    # During normal execution, it will find provionally accepted components
    #  and do nothing so another node isn't created
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    # Running without specifying logging text generates internal text
    selector.current_node_idx = 1
    selector = selection_nodes.dec_classification_doesnt_exist(
        selector,
        "accepted",
        "NotAClassification",
        class_comp_exists="provisional accept",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 0
    assert f"Node {selector.current_node_idx}" not in selector.component_status_table

    # Other normal state is to change classifications when there are
    # no components with class_comp_exists. Since the component_table
    # initialized with sample_selector as not "provisional reject"
    # components, using that for class_comp_exists
    selector = sample_selector()
    decide_comps = "accepted"
    selector = selection_nodes.dec_classification_doesnt_exist(
        selector,
        "changed accepted",
        decide_comps,
        class_comp_exists="provisional reject",
        tag_ifTrue="test true tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numTrue"] == 17
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["numFalse"] == 0
    assert f"Node {selector.current_node_idx}" in selector.component_status_table


def test_calc_varex_thresh_smoke():
    """Smoke tests for calc_varex_thresh"""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector(options="provclass")
    decide_comps = "provisional accept"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.calc_varex_thresh(
        selector, decide_comps, thresh_label="upper", percentile_thresh=90, only_used_metrics=True
    )
    assert len(used_metrics - set(["variance explained"])) == 0

    # Standard call to this function.
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="upper",
        percentile_thresh=90,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"varex_upper_thresh", "upper_perc"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_thresh"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["upper_perc"] == 90

    # Standard call , but thresh_label is ""
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="",
        percentile_thresh=90,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"varex_thresh", "perc"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_thresh"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["perc"] == 90

    # Run warning logging code to see if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector(options="provclass")
    selector.cross_component_metrics["varex_upper_thresh"] = 1
    selector.cross_component_metrics["upper_perc"] = 1
    decide_comps = "provisional accept"
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="upper",
        percentile_thresh=90,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_thresh"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["upper_perc"] == 90

    # Raise error if percentile_thresh isn't a number
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_varex_thresh(
            selector, decide_comps, thresh_label="upper", percentile_thresh="NotANumber"
        )

    # Raise error if percentile_thresh isn't a number between 0 & 100
    selector = sample_selector(options="provclass")
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_varex_thresh(
            selector, decide_comps, thresh_label="upper", percentile_thresh=101
        )

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector = selection_nodes.calc_varex_thresh(
        selector, decide_comps="NotAClassification", thresh_label="upper", percentile_thresh=90
    )
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["varex_upper_thresh"] == None
    )
    # percentile_thresh doesn't depend on components and is assigned
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["upper_perc"] == 90


def test_calc_extend_factor_smoke():
    """Smoke tests for calc_extend_factor"""

    selector = sample_selector()

    # Outputs just the metrics used in this function {""}
    used_metrics = selection_nodes.calc_extend_factor(selector, only_used_metrics=True)
    assert used_metrics == set()

    # Standard call to this function.
    selector = selection_nodes.calc_extend_factor(
        selector,
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"extend_factor"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["extend_factor"] > 0

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector()
    selector.cross_component_metrics["extend_factor"] = 1.0
    selector = selection_nodes.calc_extend_factor(selector)

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["extend_factor"] > 0

    # Run with extend_factor defined as an input
    selector = sample_selector()
    selector = selection_nodes.calc_extend_factor(selector, extend_factor=1.2)

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["extend_factor"] == 1.2


def test_max_good_meanmetricrank_smoke():
    """Smoke tests for calc_max_good_meanmetricrank"""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector("provclass")
    # This function requires "extend_factor" to already be defined
    selector.cross_component_metrics["extend_factor"] = 2.0

    # Outputs just the metrics used in this function {""}
    used_metrics = selection_nodes.calc_max_good_meanmetricrank(
        selector, "provisional accept", only_used_metrics=True
    )
    assert used_metrics == set()

    # Standard call to this function.
    selector = selection_nodes.calc_max_good_meanmetricrank(
        selector,
        "provisional accept",
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"max_good_meanmetricrank"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["max_good_meanmetricrank"] > 0
    )

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector("provclass")
    selector.cross_component_metrics["max_good_meanmetricrank"] = 10
    selector.cross_component_metrics["extend_factor"] = 2.0

    selector = selection_nodes.calc_max_good_meanmetricrank(selector, "provisional accept")

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["max_good_meanmetricrank"] > 0
    )

    # Raise an error if "extend_factor" isn't pre-defined
    selector = sample_selector("provclass")
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_max_good_meanmetricrank(selector, "provisional accept")

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector.cross_component_metrics["extend_factor"] = 2.0

    selector = selection_nodes.calc_max_good_meanmetricrank(selector, "NotAClassification")
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["max_good_meanmetricrank"]
        == None
    )


def test_calc_varex_kappa_ratio_smoke():
    """Smoke tests for calc_varex_kappa_ratio"""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector("provclass")

    # Outputs just the metrics used in this function {""}
    used_metrics = selection_nodes.calc_varex_kappa_ratio(
        selector, "provisional accept", only_used_metrics=True
    )
    assert used_metrics == {"kappa", "variance explained"}

    # Standard call to this function.
    selector = selection_nodes.calc_varex_kappa_ratio(
        selector,
        "provisional accept",
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"kappa_rate"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_rate"] > 0

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector("provclass")
    selector.cross_component_metrics["kappa_rate"] = 10
    selector = selection_nodes.calc_varex_kappa_ratio(selector, "provisional accept")

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_rate"] > 0

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector = selection_nodes.calc_varex_kappa_ratio(selector, "NotAClassification")
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["kappa_rate"] == None

    # Raise error if "varex kappa ratio" is already in component_table
    selector = sample_selector("provclass")
    selector.component_table["varex kappa ratio"] = 42
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_varex_kappa_ratio(selector, "provisional accept")


def test_calc_revised_meanmetricrank_guesses_smoke():
    """Smoke tests for calc_revised_meanmetricrank_guesses"""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector("provclass")
    selector.cross_component_metrics["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics["rho_elbow_kundu"] = 15.2

    # Outputs just the metrics used in this function {""}
    used_metrics = selection_nodes.calc_revised_meanmetricrank_guesses(
        selector,
        ["provisional accept", "provisional reject", "unclassified"],
        only_used_metrics=True,
    )
    assert used_metrics == {
        "kappa",
        "dice_FT2",
        "signal-noise_t",
        "countnoise",
        "countsigFT2",
        "rho",
    }

    # Standard call to this function.
    selector = selection_nodes.calc_revised_meanmetricrank_guesses(
        selector,
        ["provisional accept", "provisional reject", "unclassified"],
        log_extra_report="report log",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"num_acc_guess", "conservative_guess", "restrict_factor"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["num_acc_guess"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["conservative_guess"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["restrict_factor"] == 2

    # Run warning logging code for if any of the cross_component_metrics already existed and would be over-written
    selector = sample_selector("provclass")
    selector.cross_component_metrics["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics["rho_elbow_kundu"] = 15.2
    selector.cross_component_metrics["num_acc_guess"] = 10
    selector.cross_component_metrics["conservative_guess"] = 10
    selector.cross_component_metrics["restrict_factor"] = 5
    selector = selection_nodes.calc_revised_meanmetricrank_guesses(
        selector, ["provisional accept", "provisional reject", "unclassified"]
    )

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["num_acc_guess"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["conservative_guess"] > 0
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["restrict_factor"] == 2

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector.cross_component_metrics["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics["rho_elbow_kundu"] = 15.2
    selector = selection_nodes.calc_revised_meanmetricrank_guesses(selector, "NotAClassification")
    assert selector.tree["nodes"][selector.current_node_idx]["outputs"]["num_acc_guess"] == None
    assert (
        selector.tree["nodes"][selector.current_node_idx]["outputs"]["conservative_guess"] == None
    )

    # Raise error if "d_table_score_node0" is already in component_table
    selector = sample_selector("provclass")
    selector.cross_component_metrics["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics["rho_elbow_kundu"] = 15.2
    selector.component_table["d_table_score_node0"] = 42
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_revised_meanmetricrank_guesses(
            selector, ["provisional accept", "provisional reject", "unclassified"]
        )

    # Raise error if restrict_factor isn't a number
    selector = sample_selector("provclass")
    selector.cross_component_metrics["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics["rho_elbow_kundu"] = 15.2
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_revised_meanmetricrank_guesses(
            selector,
            ["provisional accept", "provisional reject", "unclassified"],
            restrict_factor="2",
        )

    # Raise error if kappa_elbow_kundu isn't in cross_component_metrics
    selector = sample_selector("provclass")
    selector.cross_component_metrics["rho_elbow_kundu"] = 15.2
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_revised_meanmetricrank_guesses(
            selector, ["provisional accept", "provisional reject", "unclassified"]
        )
