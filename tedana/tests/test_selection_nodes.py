"""Tests for the tedana.selection.selection_nodes module."""

import os

import pytest

from tedana.selection import selection_nodes
from tedana.tests.test_selection_utils import sample_selector

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_manual_classify_smoke():
    """Smoke tests for all options in manual_classify."""

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
        log_extra_info="info log",
        custom_node_label="custom label",
        tag="test tag",
    )
    # There should be 4 selected components and component_status_table should
    # have a new column "Node 0"
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 4
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 0
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    selector.current_node_idx_ = 1
    selector = selection_nodes.manual_classify(selector, "NotAClassification", new_classification)
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    assert f"Node {selector.current_node_idx_}" not in selector.component_status_table_

    # Changing components from "rejected" to "accepted" and suppressing warning
    selector.current_node_idx_ = 2
    selector = selection_nodes.manual_classify(
        selector,
        "rejected",
        new_classification,
        clear_classification_tags=True,
        log_extra_info="info log",
        tag="test tag",
        dont_warn_reclassify=True,
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 4
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_


def test_dec_left_op_right_succeeds():
    """Tests for successful calls to dec_left_op_right."""

    selector = sample_selector(options="provclass")

    decide_comps = "provisional accept"

    # Outputs just the metrics used in this function {"kappa", "rho"}
    used_metrics = selection_nodes.dec_left_op_right(
        selector, "accepted", "rejected", decide_comps, ">", "kappa", "rho", only_used_metrics=True
    )
    assert len(used_metrics - {"kappa", "rho"}) == 0

    # Standard execution where components with kappa>rho are changed from
    # "provisional accept" to "accepted"
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
        log_extra_info="info log",
        custom_node_label="custom label",
        tag_if_true="test true tag",
        tag_if_false="test false tag",
    )
    # scales are set to make sure 3 components are true and 1 is false using
    # the sample component table
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 3
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 1
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column is created in component_status_table
    selector.current_node_idx_ = 1
    selector = selection_nodes.dec_left_op_right(
        selector,
        "accepted",
        "rejected",
        "NotAClassification",
        ">",
        "kappa",
        "rho",
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    assert f"Node {selector.current_node_idx_}" not in selector.component_status_table_

    # Re-initializing selector so that it has components classificated as
    # "provisional accept" again
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
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 3
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 1
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

    # right is a component_table_metric, left is a cross_component_metric
    # left also has a left_scale that's a cross component metric
    selector = sample_selector(options="provclass")
    selector.cross_component_metrics_["new_cc_metric"] = 1.02
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
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 1
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 3
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

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
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 3
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 1
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

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
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 1
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 3
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

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
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 2
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 2
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

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
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 1
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 3
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_


def test_dec_left_op_right_fails():
    """Tests for calls to dec_left_op_right that raise errors."""

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
    """Smoke tests for dec_variance_lessthan_thresholds."""

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
        log_extra_info="info log",
        custom_node_label="custom label",
        tag_if_true="test true tag",
        tag_if_false="test false tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 1
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 3
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    selector.current_node_idx_ = 1
    selector = selection_nodes.dec_variance_lessthan_thresholds(
        selector, "accepted", "rejected", "NotAClassification"
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    assert f"Node {selector.current_node_idx_}" not in selector.component_status_table_

    # Running without specifying logging text generates internal text
    selector = sample_selector(options="provclass")
    selector = selection_nodes.dec_variance_lessthan_thresholds(
        selector, "accepted", "rejected", decide_comps
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 4
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_


def test_calc_kappa_elbow():
    """Smoke tests for calc_kappa_elbow."""

    selector = sample_selector()
    decide_comps = "all"

    # Outputs just the metrics used in this function
    used_metrics = selection_nodes.calc_kappa_elbow(selector, decide_comps, only_used_metrics=True)
    assert len(used_metrics - {"kappa"}) == 0

    # Standard call to this function.
    selector = selection_nodes.calc_kappa_elbow(
        selector,
        decide_comps,
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {
        "kappa_elbow_kundu",
        "kappa_allcomps_elbow",
        "kappa_nonsig_elbow",
        "varex_upper_p",
    }
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_elbow_kundu"] > 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_allcomps_elbow"] > 0
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_nonsig_elbow"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_upper_p"] > 0

    # Using a subset of components for decide_comps.
    selector = selection_nodes.calc_kappa_elbow(
        selector,
        decide_comps="accepted",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {
        "kappa_elbow_kundu",
        "kappa_allcomps_elbow",
        "kappa_nonsig_elbow",
        "varex_upper_p",
    }
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_elbow_kundu"] > 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_allcomps_elbow"] > 0
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_nonsig_elbow"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_upper_p"] > 0

    # No components with "NotALabel" classification so nothing selected
    selector = sample_selector()
    decide_comps = "NotALabel"

    # Outputs just the metrics used in this function
    selector = selection_nodes.calc_kappa_elbow(selector, decide_comps)
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_elbow_kundu"] is None
    )
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_allcomps_elbow"]
        is None
    )
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_nonsig_elbow"] is None
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_upper_p"] is None


def test_calc_rho_elbow():
    """Smoke tests for calc_rho_elbow."""

    selector = sample_selector(options="unclass")
    decide_comps = "all"

    # Outputs just the metrics used in this function
    used_metrics = selection_nodes.calc_rho_elbow(selector, decide_comps, only_used_metrics=True)
    assert len(used_metrics - {"kappa", "rho", "variance explained"}) == 0

    # Standard call to this function.
    selector = selection_nodes.calc_rho_elbow(
        selector,
        decide_comps,
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {
        "rho_elbow_kundu",
        "rho_allcomps_elbow",
        "rho_unclassified_elbow",
        "elbow_f05",
    }
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_allcomps_elbow"] > 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_unclassified_elbow"] > 0
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["elbow_f05"] > 0

    # Standard call to this function using rho_elbow_type="liberal"
    selector = selection_nodes.calc_rho_elbow(
        selector,
        decide_comps,
        rho_elbow_type="liberal",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {
        "rho_elbow_liberal",
        "rho_allcomps_elbow",
        "rho_unclassified_elbow",
        "elbow_f05",
    }
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_elbow_liberal"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_allcomps_elbow"] > 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_unclassified_elbow"] > 0
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["elbow_f05"] > 0

    # Using a subset of components for decide_comps.
    selector = selection_nodes.calc_rho_elbow(
        selector,
        decide_comps=["accepted", "unclassified"],
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {
        "rho_elbow_kundu",
        "rho_allcomps_elbow",
        "rho_unclassified_elbow",
        "elbow_f05",
    }
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_elbow_kundu"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_allcomps_elbow"] > 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_unclassified_elbow"] > 0
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["elbow_f05"] > 0

    with pytest.raises(ValueError):
        selection_nodes.calc_rho_elbow(selector, decide_comps, rho_elbow_type="perfect")

    # No components with "NotALabel" classification so nothing selected
    selector = sample_selector()
    decide_comps = "NotALabel"

    # Outputs just the metrics used in this function
    selector = selection_nodes.calc_rho_elbow(selector, decide_comps)
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_elbow_kundu"] is None
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_allcomps_elbow"] is None
    )
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["rho_unclassified_elbow"]
        is None
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["elbow_f05"] is None


def test_calc_median_smoke():
    """Smoke tests for calc_median."""

    selector = sample_selector()
    decide_comps = "all"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.calc_median(
        selector,
        decide_comps,
        metric_name="variance explained",
        median_label="varex",
        only_used_metrics=True,
    )
    assert len(used_metrics - {"variance explained"}) == 0

    # Standard call to this function.
    selector = selection_nodes.calc_median(
        selector,
        decide_comps,
        metric_name="variance explained",
        median_label="varex",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"median_varex"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["median_varex"] > 0

    # repeating standard call and should make a warning because metric_varex already exists
    selector = selection_nodes.calc_median(
        selector, decide_comps, metric_name="variance explained", median_label="varex"
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["median_varex"] > 0

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector = selection_nodes.calc_median(
        selector,
        decide_comps="NotAClassification",
        metric_name="variance explained",
        median_label="varex",
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["median_varex"] is None

    # Crashes because median_label is not a string
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_median(
            selector,
            decide_comps,
            metric_name="variance explained",
            median_label=5,
            log_extra_info="info log",
            custom_node_label="custom label",
        )

    # Crashes because median_name is not a string
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_median(
            selector,
            decide_comps,
            metric_name=5,
            median_label="varex",
            log_extra_info="info log",
            custom_node_label="custom label",
        )


def test_dec_classification_doesnt_exist_smoke():
    """Smoke tests for dec_classification_doesnt_exist."""

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
        at_least_num_exist=1,
        class_comp_exists="provisional accept",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag="test true tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    # Lists the number of components in decide_comps in n_false
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 17
    # During normal execution, it will find provionally accepted components
    #  and do nothing so another node isn't created
    assert f"Node {selector.current_node_idx_}" not in selector.component_status_table_

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column not created in component_status_table
    # Running without specifying logging text generates internal text
    selector.current_node_idx_ = 1
    selector = selection_nodes.dec_classification_doesnt_exist(
        selector,
        "accepted",
        "NotAClassification",
        class_comp_exists="provisional accept",
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    assert f"Node {selector.current_node_idx_}" not in selector.component_status_table_

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
        tag="test true tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 17
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 0
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

    # Standard execution with at_least_num_exist=5 which should trigger the
    #   components don't exist output
    selector = sample_selector(options="unclass")
    selector = selection_nodes.dec_classification_doesnt_exist(
        selector,
        "accepted",
        decide_comps=["unclassified", "provisional accept"],
        at_least_num_exist=5,
        class_comp_exists="provisional accept",
        log_extra_info="info log",
        custom_node_label="custom label",
        tag="test true tag",
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 17
    # Lists the number of components in decide_comps in n_false
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 0
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_


def test_dec_reclassify_high_var_comps():
    """Tests for dec_reclassify_high_var_comps."""

    selector = sample_selector(options="unclass")
    decide_comps = "unclassified"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.dec_reclassify_high_var_comps(
        selector,
        "unclass_highvar",
        decide_comps,
        only_used_metrics=True,
    )
    assert len(used_metrics - {"variance explained"}) == 0

    # Raises an error since varex_upper_p not in cross_component_metrics
    #   & there are components in decide_comps
    with pytest.raises(ValueError):
        selection_nodes.dec_reclassify_high_var_comps(
            selector,
            "unclass_highvar",
            decide_comps,
        )

    # varex_upper_p not in cross_component_metrics,
    #   but doesn't raise an error because no components in decide_comps
    selection_nodes.dec_reclassify_high_var_comps(
        selector,
        "unclass_highvar",
        "NotAClassification",
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    assert f"Node {selector.current_node_idx_}" not in selector.component_status_table_

    # Add varex_upper_p to cross component_metrics to run normal test
    selector = sample_selector(options="unclass")
    selector.cross_component_metrics_["varex_upper_p"] = 0.97

    # Standard execution where with all extra logging code and options changed from defaults
    selection_nodes.dec_reclassify_high_var_comps(
        selector,
        "unclass_highvar",
        decide_comps,
        log_extra_info="info log",
        custom_node_label="custom label",
        tag="test true tag",
    )
    # Lists the number of components in decide_comps in n_true or n_false
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 3
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_false"] == 10
    assert f"Node {selector.current_node_idx_}" in selector.component_status_table_

    # No components with "NotALabel" classification so nothing selected and no
    #   Node 1 column is created in component_status_table
    selector.current_node_idx_ = 1
    selector = selection_nodes.dec_reclassify_high_var_comps(
        selector, "unclass_highvar", "NotAClassification"
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["n_true"] == 0
    assert f"Node {selector.current_node_idx_}" not in selector.component_status_table_


def test_calc_varex_thresh_smoke():
    """Smoke tests for calc_varex_thresh."""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector()
    decide_comps = "all"

    # Outputs just the metrics used in this function {"variance explained"}
    used_metrics = selection_nodes.calc_varex_thresh(
        selector, decide_comps, thresh_label="upper", percentile_thresh=90, only_used_metrics=True
    )
    assert len(used_metrics - {"variance explained"}) == 0

    # Standard call to this function.
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="upper",
        percentile_thresh=90,
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"varex_upper_thresh", "upper_perc"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_upper_thresh"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["upper_perc"] == 90

    # Standard call , but thresh_label is ""
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="",
        percentile_thresh=90,
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"varex_thresh", "perc"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_thresh"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["perc"] == 90

    # Standard call using num_highest_var_comps as an integer
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="new_lower",
        percentile_thresh=25,
        num_highest_var_comps=8,
    )
    calc_cross_comp_metrics = {"varex_new_lower_thresh", "new_lower_perc"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_new_lower_thresh"] > 0
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["new_lower_perc"] == 25

    # Standard call using num_highest_var_comps as a value in cross_component_metrics
    selector.cross_component_metrics_["num_acc_guess"] = 10
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="new_lower",
        percentile_thresh=25,
        num_highest_var_comps="num_acc_guess",
    )
    calc_cross_comp_metrics = {"varex_new_lower_thresh", "new_lower_perc"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_new_lower_thresh"] > 0
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["new_lower_perc"] == 25

    # Raise error if num_highest_var_comps is a string, but not in cross_component_metrics
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_varex_thresh(
            selector,
            decide_comps,
            thresh_label="new_lower",
            percentile_thresh=25,
            num_highest_var_comps="NotACrossCompMetric",
        )

    # Do not raise error if num_highest_var_comps is a string & not in cross_component_metrics,
    # but decide_comps doesn't select any components
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps="NoComponents",
        thresh_label="new_lower",
        percentile_thresh=25,
        num_highest_var_comps="NotACrossCompMetric",
    )
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_new_lower_thresh"]
        is None
    )
    # percentile_thresh doesn't depend on components and is assigned
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["new_lower_perc"] == 25

    # Raise error if num_highest_var_comps is not an integer
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_varex_thresh(
            selector,
            decide_comps,
            thresh_label="new_lower",
            percentile_thresh=25,
            num_highest_var_comps=9.5,
        )

    # Still run num_highest_var_comps is larger than the number of selected components
    #  NOTE: To match original functionaly this will run but add an info message
    #   and set num_highest_var_comps to the number of selected components
    #
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="new_lower",
        percentile_thresh=25,
        num_highest_var_comps=55,
    )
    calc_cross_comp_metrics = {"varex_new_lower_thresh", "new_lower_perc"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_new_lower_thresh"] > 0
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["new_lower_perc"] == 25

    # Run warning logging code to see if any of the cross_component_metrics
    # already exists and would be over-written
    selector = sample_selector(options="provclass")
    selector.cross_component_metrics_["varex_upper_thresh"] = 1
    selector.cross_component_metrics_["upper_perc"] = 1
    decide_comps = "provisional accept"
    selector = selection_nodes.calc_varex_thresh(
        selector,
        decide_comps,
        thresh_label="upper",
        percentile_thresh=90,
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_upper_thresh"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["upper_perc"] == 90

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
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["varex_upper_thresh"] is None
    )
    # percentile_thresh doesn't depend on components and is assigned
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["upper_perc"] == 90


def test_calc_extend_factor_smoke():
    """Smoke tests for calc_extend_factor."""

    selector = sample_selector()

    # Outputs just the metrics used in this function {""}
    used_metrics = selection_nodes.calc_extend_factor(selector, only_used_metrics=True)
    assert used_metrics == set()

    # Standard call to this function.
    selector = selection_nodes.calc_extend_factor(
        selector,
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"extend_factor"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["extend_factor"] > 0

    # Run warning logging code for if any of the cross_component_metrics
    # already existed and would be over-written
    selector = sample_selector()
    selector.cross_component_metrics_["extend_factor"] = 1.0
    selector = selection_nodes.calc_extend_factor(selector)

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["extend_factor"] > 0

    # Run with extend_factor defined as an input
    selector = sample_selector()
    selector = selection_nodes.calc_extend_factor(selector, extend_factor=1.2)

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["extend_factor"] == 1.2


def test_calc_max_good_meanmetricrank_smoke():
    """Smoke tests for calc_max_good_meanmetricrank."""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector("provclass")
    # This function requires "extend_factor" to already be defined
    selector.cross_component_metrics_["extend_factor"] = 2.0

    # Outputs just the metrics used in this function {""}
    used_metrics = selection_nodes.calc_max_good_meanmetricrank(
        selector, "provisional accept", only_used_metrics=True
    )
    assert used_metrics == set()

    # Standard call to this function.
    selector = selection_nodes.calc_max_good_meanmetricrank(
        selector,
        "provisional accept",
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"max_good_meanmetricrank"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["max_good_meanmetricrank"]
        > 0
    )

    # Standard call to this function with a user defined metric_suffix
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["extend_factor"] = 2.0
    selector = selection_nodes.calc_max_good_meanmetricrank(
        selector, "provisional accept", metric_suffix="testsfx"
    )
    calc_cross_comp_metrics = {"max_good_meanmetricrank_testsfx"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"][
            "max_good_meanmetricrank_testsfx"
        ]
        > 0
    )

    # Run warning logging code for if any of the cross_component_metrics
    # already existed and would be over-written
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["max_good_meanmetricrank"] = 10
    selector.cross_component_metrics_["extend_factor"] = 2.0

    selector = selection_nodes.calc_max_good_meanmetricrank(selector, "provisional accept")
    calc_cross_comp_metrics = {"max_good_meanmetricrank"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["max_good_meanmetricrank"]
        > 0
    )

    # Raise an error if "extend_factor" isn't pre-defined
    selector = sample_selector("provclass")
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_max_good_meanmetricrank(selector, "provisional accept")

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector.cross_component_metrics_["extend_factor"] = 2.0

    selector = selection_nodes.calc_max_good_meanmetricrank(selector, "NotAClassification")
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["max_good_meanmetricrank"]
        is None
    )


def test_calc_varex_kappa_ratio_smoke():
    """Smoke tests for calc_varex_kappa_ratio."""

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
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"kappa_rate"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_rate"] > 0

    # Run warning logging code for if any of the cross_component_metrics
    # already existed and would be over-written
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["kappa_rate"] = 10
    selector = selection_nodes.calc_varex_kappa_ratio(selector, "provisional accept")

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_rate"] > 0

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector = selection_nodes.calc_varex_kappa_ratio(selector, "NotAClassification")
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["kappa_rate"] is None

    # Raise error if "varex kappa ratio" is already in component_table
    selector = sample_selector("provclass")
    selector.component_table_["varex kappa ratio"] = 42
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_varex_kappa_ratio(selector, "provisional accept")


def test_calc_revised_meanmetricrank_guesses_smoke():
    """Smoke tests for calc_revised_meanmetricrank_guesses."""

    # Standard use of this function requires some components to be "provisional accept"
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics_["rho_elbow_kundu"] = 15.2

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
        log_extra_info="info log",
        custom_node_label="custom label",
    )
    calc_cross_comp_metrics = {"num_acc_guess", "conservative_guess", "restrict_factor"}
    output_calc_cross_comp_metrics = set(
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["calc_cross_comp_metrics"]
    )
    # Confirming the intended metrics are added to outputs and they have non-zero values
    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["num_acc_guess"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["conservative_guess"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["restrict_factor"] == 2

    # Run warning logging code for if any of the cross_component_metrics
    # already existed and would be over-written
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics_["rho_elbow_kundu"] = 15.2
    selector.cross_component_metrics_["num_acc_guess"] = 10
    selector.cross_component_metrics_["conservative_guess"] = 10
    selector.cross_component_metrics_["restrict_factor"] = 5
    selector = selection_nodes.calc_revised_meanmetricrank_guesses(
        selector, ["provisional accept", "provisional reject", "unclassified"]
    )

    assert len(output_calc_cross_comp_metrics - calc_cross_comp_metrics) == 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["num_acc_guess"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["conservative_guess"] > 0
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["restrict_factor"] == 2

    # Log without running if no components of decide_comps are in the component table
    selector = sample_selector()
    selector.cross_component_metrics_["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics_["rho_elbow_kundu"] = 15.2
    selector = selection_nodes.calc_revised_meanmetricrank_guesses(selector, "NotAClassification")
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["num_acc_guess"] is None
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["conservative_guess"] is None
    )

    # Raise error if "d_table_score_node0" is already in component_table
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics_["rho_elbow_kundu"] = 15.2
    selector.component_table_["d_table_score_node0"] = 42
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_revised_meanmetricrank_guesses(
            selector, ["provisional accept", "provisional reject", "unclassified"]
        )

    # Raise error if restrict_factor isn't a number
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["kappa_elbow_kundu"] = 19.1
    selector.cross_component_metrics_["rho_elbow_kundu"] = 15.2
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_revised_meanmetricrank_guesses(
            selector,
            ["provisional accept", "provisional reject", "unclassified"],
            restrict_factor="2",
        )

    # Raise error if kappa_elbow_kundu isn't in cross_component_metrics
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["rho_elbow_kundu"] = 15.2
    with pytest.raises(ValueError):
        selector = selection_nodes.calc_revised_meanmetricrank_guesses(
            selector, ["provisional accept", "provisional reject", "unclassified"]
        )

    # Do not raise error if kappa_elbow_kundu isn't in cross_component_metrics
    # and there are no components in decide_comps
    selector = sample_selector("provclass")
    selector.cross_component_metrics_["rho_elbow_kundu"] = 15.2

    selector = selection_nodes.calc_revised_meanmetricrank_guesses(
        selector, decide_comps="NoComponents"
    )
    assert selector.tree["nodes"][selector.current_node_idx_]["outputs"]["num_acc_guess"] is None
    assert (
        selector.tree["nodes"][selector.current_node_idx_]["outputs"]["conservative_guess"] is None
    )
