"""Functions that will be used as steps in a decision tree."""

import logging

import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile

from tedana.docs import fill_doc
from tedana.metrics.dependence import generate_decision_table_score
from tedana.selection.selection_utils import (
    change_comptable_classifications,
    confirm_metrics_exist,
    get_extend_factor,
    kappa_elbow_kundu,
    log_decision_tree_step,
    rho_elbow_kundu_liberal,
    selectcomps2use,
)

LGR = logging.getLogger("GENERAL")


@fill_doc
def manual_classify(
    selector,
    decide_comps,
    new_classification,
    clear_classification_tags=False,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag=None,
    dont_warn_reclassify=False,
):
    """Assign a classification defined in new_classification to the components in decide_comps.

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    new_classification : :obj:`str`
        Assign all components identified in decide_comps the classification
        in new_classification. Options are 'unclassified', 'accepted',
        'rejected', or intermediate_classification labels predefined in the
        decision tree
    clear_classification_tags : :obj:`bool`
        If True, reset all values in the 'classification_tags' column to empty
        strings. This also can create the classification_tags column if it
        does not already exist. If False, do nothing.
    tag : :obj:`str`
        A classification tag to assign to all components being reclassified.
        This should be one of the tags defined by classification_tags in
        the decision tree specification
    dont_warn_reclassify : :obj:`bool`
        By default, if this function changes a component classification from accepted or
        rejected to something else, it gives a warning, since those should be terminal
        classifications. If this is True, that warning is suppressed.
        (Useful if manual_classify is used to reset all labels to unclassified).
        Default=False
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    This was designed with three use cases in mind:
    (1) Set the classifications of all components to unclassified
    for the first node of a decision tree. clear_classification_tags=True is
    recommended for this use case.
    (2) Shift all components between classifications, such as provisionalaccept to accepted for the
    penultimate node in the decision tree.
    (3) Manually re-classify components by number based on user observations.

    Unlike other decision node functions, ``if_true`` and ``if_false`` are not inputs
    since the same classification is assigned to all components listed in ``decide_comps``.
    """
    # predefine all outputs that should be logged
    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "used_metrics": set(),
        "node_label": None,
        "n_true": None,
        "n_false": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    if_true = new_classification
    if_false = "nochange"

    function_name_idx = f"Step {selector.current_node_idx_}: manual_classify"
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Set " + str(decide_comps) + " to " + new_classification

    LGR.info(f"{function_name_idx}: {outputs['node_label']} ")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    comps2use = selectcomps2use(selector.component_table_, decide_comps)

    if not comps2use:
        log_decision_tree_step(function_name_idx, comps2use, decide_comps=decide_comps)
        outputs["n_true"] = 0
        outputs["n_false"] = 0
    else:
        decision_boolean = pd.Series(True, index=comps2use)
        selector, outputs["n_true"], outputs["n_false"] = change_comptable_classifications(
            selector,
            if_true,
            if_false,
            decision_boolean,
            tag_if_true=tag,
            dont_warn_reclassify=dont_warn_reclassify,
        )

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            n_true=outputs["n_true"],
            n_false=outputs["n_false"],
            if_true=if_true,
            if_false=if_false,
        )

    if clear_classification_tags:
        selector.component_table_["classification_tags"] = ""
        LGR.info(function_name_idx + " component classification tags are cleared")

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def dec_left_op_right(
    selector,
    if_true,
    if_false,
    decide_comps,
    op,
    left,
    right,
    left_scale=1,
    right_scale=1,
    op2=None,
    left2=None,
    right2=None,
    left2_scale=1,
    right2_scale=1,
    op3=None,
    left3=None,
    right3=None,
    left3_scale=1,
    right3_scale=1,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag_if_true=None,
    tag_if_false=None,
):
    """Perform a relational comparison.

    Parameters
    ----------
    %(selector)s
    %(tag_if_true)s
    %(tag_if_false)s
    %(decide_comps)s
    op : :obj:`str`
        Must be one of: ">", ">=", "==", "<=", "<"
        Applied the user defined operator to left op right
    left, right : :obj:`str` or :obj:`float`
        The labels for the two metrics to be used for comparision.
        For example: left='kappa', right='rho' and op='>' means this
        function will test kappa>rho. One of the two can also be a number.
        In that case, a metric would be compared against a fixed threshold.
        For example left='T2fitdiff_invsout_ICAmap_Tstat', right=0, and op='>'
        means this function will test T2fitdiff_invsout_ICAmap_Tstat>0
    left_scale, right_scale : :obj:`float` or :obj:`str`
        Multiply the left or right metrics value by a constant. For example
        if left='kappa', right='rho', right_scale=2, and op='>' this tests
        kappa>(2*rho). These can also be a string that is a value in
        cross_component_metrics, since those will resolve to a single value.
        This cannot be a label for a component_table column since that would
        output a different value for each component. Default=1
    op2 : :obj:`str`, Default=None
    left2, right2, left3, right3 : :obj:`str` or :obj:`float`, Default=None
    left2_scale, right2_scale, left3_scale, right3_scale : :obj:`float` or :obj:`str`, Default=1
        This function can also be used to calculate the intersection of two or three
        boolean statements. If op2, left2, and right2 are defined then
        this function returns
        (left_scale*)left op (right_scale*right) AND (left2_scale*)left2 op2 (right2_scale*right2)
        if the "3" parameters are also defined then it's the intersection of all 3 statements
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s
    %(tag_if_true)s
    %(tag_if_false)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    This function is ideally run with one boolean statement at a time so that
    the result of each boolean is logged. For example, it's better to test
    kappa>kappa_elbow and rho>rho_elbow with two separate calls to this function
    so that the results of each test can be easily viewed. That said, particularly for
    the original kundu decision tree, if you're making decisions on components with
    various classifications based on multiple boolean statements, the decision tree
    becomes really messy and the added functionality here is useful.
    Combinations of boolean statements only test with "and" and not "or". This is
    an intentional decision because, if a classification changes if A>B or C>D are true
    then A>B and C>D should be logged separately
    """
    # predefine all outputs that should be logged
    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "used_metrics": set(),
        "used_cross_component_metrics": set(),
        "node_label": None,
        "n_true": None,
        "n_false": None,
    }

    function_name_idx = f"Step {selector.current_node_idx_}: left_op_right"
    # Only select components if the decision tree is being run
    if not only_used_metrics:
        comps2use = selectcomps2use(selector.component_table_, decide_comps)

    def identify_used_metric(val, isnum=False):
        """
        Parse the left or right values or scalers.

        This is done to see if they are an existing used_metric or cross_component_metric.

        If the value already a number, no parse would be needed.

        This is also used on left_scale and right_scale to convert
        a value in cross_component_metrics to a number. Set the isnum
        flag to true for those inputs and this will raise an error
        if a number isn't loaded
        """
        orig_val = val
        if isinstance(val, str):
            if val in selector.component_table_.columns:
                outputs["used_metrics"].update([val])
            elif val in selector.cross_component_metrics_:
                outputs["used_cross_component_metrics"].update([val])
                val = selector.cross_component_metrics_[val]
            # If decision tree is being run, then throw errors or messages
            #  if a component doesn't exist. If this is just getting a list
            #  of metrics to be used, then don't bring up warnings
            elif not only_used_metrics:
                if not comps2use:
                    LGR.info(
                        f"{function_name_idx}: {val} is neither a metric in "
                        "selector.component_table_ nor selector.cross_component_metrics_, "
                        f"but no components with {decide_comps} remain by this node "
                        "so nothing happens"
                    )
                else:
                    raise ValueError(
                        f"{val} is neither a metric in selector.component_table_ "
                        "nor selector.cross_component_metrics_"
                    )
        if isnum:
            if not isinstance(val, (int, float)):
                raise ValueError(f"{orig_val} must be a number. It is {val}")
        return val

    legal_ops = (">", ">=", "==", "<=", "<")

    def confirm_valid_conditional(left_scale, left_val, right_scale, right_val, op_val):
        """
        Confirm that the conditional statement is valid.

        Makes sure the left_scale, left_val, right_scale, right_val, and
        operator variables combine into a valid conditional statement.
        """
        left_val = identify_used_metric(left_val)
        right_val = identify_used_metric(right_val)
        left_scale = identify_used_metric(left_scale, isnum=True)
        right_scale = identify_used_metric(right_scale, isnum=True)

        if op_val not in legal_ops:
            raise ValueError(f"{op_val} is not a binary comparison operator, like > or <")
        return left_scale, left_val, right_scale, right_val

    def operator_scale_descript(val_scale, val):
        """
        Return a string with one element from the mathematical expression.

        If val_scale is not 1, include scaling factor (rounded to 2 decimals).

        If val is a column in the component_table output the column label
        If val is a number (either an inputted number or from cross_component_metrics
        include the number (rounded to 2 decimals)
        This output is used to great a descriptor for visualizing the decision tree
        Unrounded values are saved and rounding here will not affect results.
        """
        if not isinstance(val, str):
            val = str(round(val, 2))
        if val_scale == 1:
            return val
        else:
            return f"{round(val_scale, 2)}*{val}"

    left_scale, left, right_scale, right = confirm_valid_conditional(
        left_scale, left, right_scale, right, op
    )
    descript_left = operator_scale_descript(left_scale, left)
    descript_right = operator_scale_descript(right_scale, right)
    is_compound = 0

    # If any of the values for the second boolean statement are set
    if left2 is not None or right2 is not None or op2 is not None:
        # Check if they're all set & use them all or raise an error
        if left2 is not None and right2 is not None and op2 is not None:
            is_compound = 2
            left2_scale, left2, right2_scale, right2 = confirm_valid_conditional(
                left2_scale, left2, right2_scale, right2, op2
            )
            descript_left2 = operator_scale_descript(left2_scale, left2)
            descript_right2 = operator_scale_descript(right2_scale, right2)
        else:
            raise ValueError(
                "left_op_right can check if a first and second boolean "
                "statement are both true. This call includes some but not "
                "all variables to define the second boolean statement "
                f"left2={left2}, right2={right2}, op2={op2}"
            )

    # If any of the values for the second boolean statement are set
    if left3 or right3 or op3:
        if is_compound == 0:
            raise ValueError(
                "left_op_right is includes parameters for a third conditional "
                "(left3, right3, or op3) statement without setting the "
                "second statement"
            )
        # Check if they're all set & use them all or raise an error
        if left3 and right3 and op3:
            is_compound = 3
            left3_scale, left3, right3_scale, right3 = confirm_valid_conditional(
                left3_scale, left3, right3_scale, right3, op3
            )
            descript_left3 = operator_scale_descript(left3_scale, left3)
            descript_right3 = operator_scale_descript(right3_scale, right3)
        else:
            raise ValueError(
                "left_op_right can check if three boolean "
                "statements are all true. This call includes some but not "
                "all variables to define the third boolean statement "
                f"left3={left3}, right3={right3}, op3={op3}"
            )

    if only_used_metrics:
        return outputs["used_metrics"]

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    elif is_compound == 0:
        outputs["node_label"] = f"{descript_left}{op}{descript_right}"
    elif is_compound == 2:
        outputs["node_label"] = [
            f"{descript_left}{op}{descript_right} & " f"{descript_left2}{op2}{descript_right2}"
        ]
    elif is_compound == 3:
        outputs["node_label"] = [
            f"{descript_left}{op}{descript_right} & "
            f"{descript_left2}{op2}{descript_right2} & "
            f"{descript_left3}{op3}{descript_right3}"
        ]

    # Might want to add additional default logging to functions here
    # The function input will be logged before the function call
    LGR.info(f"{function_name_idx}: {if_true} if {outputs['node_label']}, else {if_false}")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    def parse_vals(val):
        """Get the metric values for the selected components or relevant constant."""
        if isinstance(val, str):
            return selector.component_table_.loc[comps2use, val].copy()
        else:
            return val  # should be a fixed number

    if not comps2use:
        outputs["n_true"] = 0
        outputs["n_false"] = 0
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
            if_true=outputs["n_true"],
            if_false=outputs["n_false"],
        )

    else:
        left1_val = parse_vals(left)  # noqa: F841
        right1_val = parse_vals(right)  # noqa: F841
        decision_boolean = eval(f"(left_scale*left1_val) {op} (right_scale * right1_val)")
        if is_compound >= 2:
            left2_val = parse_vals(left2)  # noqa: F841
            right2_val = parse_vals(right2)  # noqa: F841
            statement1 = decision_boolean.copy()
            statement2 = eval(f"(left2_scale*left2_val) {op2} (right2_scale * right2_val)")
            # logical dot product for compound statement
            decision_boolean = statement1 * statement2
        if is_compound == 3:
            left3_val = parse_vals(left3)  # noqa: F841
            right3_val = parse_vals(right3)  # noqa: F841
            # statement 1 is now the combination of the first two conditional statements
            statement1 = decision_boolean.copy()
            # statement 2 is now the third conditional statement
            statement2 = eval(f"(left3_scale*left3_val) {op2} (right3_scale * right3_val)")
            # logical dot product for compound statement
            decision_boolean = statement1 * statement2

        (
            selector,
            outputs["n_true"],
            outputs["n_false"],
        ) = change_comptable_classifications(
            selector,
            if_true,
            if_false,
            decision_boolean,
            tag_if_true=tag_if_true,
            tag_if_false=tag_if_false,
        )
        # outputs["n_true"] = np.asarray(decision_boolean).sum()
        # outputs["n_false"] = np.logical_not(decision_boolean).sum()

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            n_true=outputs["n_true"],
            n_false=outputs["n_false"],
            if_true=if_true,
            if_false=if_false,
        )

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def dec_variance_lessthan_thresholds(
    selector,
    if_true,
    if_false,
    decide_comps,
    var_metric="variance explained",
    single_comp_threshold=0.1,
    all_comp_threshold=1.0,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag_if_true=None,
    tag_if_false=None,
):
    """Change classifications for components with variance<single_comp_threshold.

    If the sum of the variance for all components that meet this criteria
    is greater than all_comp_threshold then only change classifications for the
    lowest variance components where the sum of their variances is less than
    all_comp_threshold

    Parameters
    ----------
    %(selector)s
    %(tag_if_true)s
    %(tag_if_false)s
    %(decide_comps)s
    var_metric : :obj:`str`
        The name of the metric in component_table for variance. Default="variance explained"
        This is an option so that it is possible to use "normalized variance explained"
        or another metric
    single_comp_threshold : :obj:`float`
        The threshold for which all components need to have lower variance.
        Default=0.1
    all_comp_threshold : :obj: `float`
        The number of the variance for all components less than single_comp_threshold
        needs to be under this threshold. Default=1.0
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s
    %(tag_if_true)s
    %(tag_if_false)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s
    """
    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "used_metrics": {var_metric},
        "node_label": None,
        "n_true": None,
        "n_false": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = f"Step {selector.current_node_idx_}: variance_lt_thresholds"
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = (
            f"{var_metric}<{single_comp_threshold}. All variance<{all_comp_threshold}"
        )

    LGR.info(f"{function_name_idx}: {if_true} if {outputs['node_label']}, else {if_false}")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        outputs["n_true"] = 0
        outputs["n_false"] = 0
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
            if_true=outputs["n_true"],
            if_false=outputs["n_false"],
        )
    else:
        variance = selector.component_table_.loc[comps2use, var_metric]
        decision_boolean = variance < single_comp_threshold
        # if all the low variance components sum above all_comp_threshold
        # keep removing the highest remaining variance component until
        # the sum is below all_comp_threshold. This is an inefficient
        # way to do this, but it works & should never cause an infinite loop
        if variance[decision_boolean].sum() > all_comp_threshold:
            while variance[decision_boolean].sum() > all_comp_threshold:
                tmpmax = variance == variance[decision_boolean].max()
                decision_boolean[tmpmax] = False
        (
            selector,
            outputs["n_true"],
            outputs["n_false"],
        ) = change_comptable_classifications(
            selector,
            if_true,
            if_false,
            decision_boolean,
            tag_if_true=tag_if_true,
            tag_if_false=tag_if_false,
        )

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            n_true=outputs["n_true"],
            n_false=outputs["n_false"],
            if_true=if_true,
            if_false=if_false,
        )

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs
    return selector


@fill_doc
def calc_median(
    selector,
    decide_comps,
    metric_name,
    median_label,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """Calculate the median across components for the metric defined by metric_name.

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    metric_name : :obj:`str`
        The name of a column in selector.component_table. The median of
        the values in this column will be calculated
    median_label : :obj:`str`
        The median will be saved in "median_(median_label)"
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s
    """
    function_name_idx = f"Step {selector.current_node_idx_}: calc_median"
    if not isinstance(median_label, str):
        raise ValueError(
            f"{function_name_idx}: median_label must be a string. It is: {median_label}"
        )
    else:
        label_name = f"median_{median_label}"

    if not isinstance(metric_name, str):
        raise ValueError(
            f"{function_name_idx}: metric_name must be a string. It is: {metric_name}"
        )

    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "node_label": None,
        label_name: None,
        "used_metrics": {metric_name},
        "calc_cross_comp_metrics": [label_name],
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    if label_name in selector.cross_component_metrics_:
        LGR.warning(
            f"{label_name} already calculated. Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = f"Median({label_name})"

    LGR.info(f"{function_name_idx}: {outputs['node_label']}")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:
        outputs[label_name] = np.median(selector.component_table_.loc[comps2use, metric_name])

        selector.cross_component_metrics_[label_name] = outputs[label_name]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def calc_kappa_elbow(
    selector,
    decide_comps,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """Calculate elbow for kappa across components.

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    This function is currently hard coded for a specific way to calculate the kappa elbow
    based on the method by Kundu in the MEICA v2.5 code. This uses the minimum of
    a kappa elbow calculation on all components and on a subset of kappa values below
    a significance threshold. To get the same functionality as in MEICA v2.5,
    decide_comps must be 'all'.

    varex_upper_p isn't used for anything in this function, but it is calculated
    on kappa values and is used in rho_elbow_kundu_liberal and
    dec_reclassify_high_var_comps. For several reasons it made more sense to calculate here.
    This also means the kappa elbow should be calculated before those two other functions
    are called
    """
    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "node_label": None,
        "n_echos": selector.cross_component_metrics_["n_echos"],
        "used_metrics": {"kappa"},
        "calc_cross_comp_metrics": [
            "kappa_elbow_kundu",
            "kappa_allcomps_elbow",
            "kappa_nonsig_elbow",
            "varex_upper_p",
        ],
        "kappa_elbow_kundu": None,
        "kappa_allcomps_elbow": None,
        "kappa_nonsig_elbow": None,
        "varex_upper_p": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = f"Step {selector.current_node_idx_}: calc_kappa_elbow"

    if ("kappa_elbow_kundu" in selector.cross_component_metrics_) and (
        "kappa_elbow_kundu" in outputs["calc_cross_comp_metrics"]
    ):
        LGR.warning(
            "kappa_elbow_kundu already calculated."
            f"Overwriting previous value in {function_name_idx}"
        )

    if "varex_upper_p" in selector.cross_component_metrics_:
        LGR.warning(
            f"varex_upper_p already calculated. Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Calc Kappa Elbow"

    LGR.info(f"{function_name_idx}: {outputs['node_label']}")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:
        (
            outputs["kappa_elbow_kundu"],
            outputs["kappa_allcomps_elbow"],
            outputs["kappa_nonsig_elbow"],
            outputs["varex_upper_p"],
        ) = kappa_elbow_kundu(
            selector.component_table_,
            selector.cross_component_metrics_["n_echos"],
            comps2use=comps2use,
        )
        selector.cross_component_metrics_["kappa_elbow_kundu"] = outputs["kappa_elbow_kundu"]
        selector.cross_component_metrics_["kappa_allcomps_elbow"] = outputs["kappa_allcomps_elbow"]
        selector.cross_component_metrics_["kappa_nonsig_elbow"] = outputs["kappa_nonsig_elbow"]
        selector.cross_component_metrics_["varex_upper_p"] = outputs["varex_upper_p"]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def calc_rho_elbow(
    selector,
    decide_comps,
    subset_decide_comps="unclassified",
    rho_elbow_type="kundu",
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """Calculate elbow for rho across components.

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    subset_decide_comps : :obj:`str`
        This is a string with a single component classification label. For the
        elbow calculation used by Kundu in MEICA v.27 thresholds are based
        on all components and on unclassified components.
        Default='unclassified'.
    rho_elbow_type : :obj:`str`
        The algorithm used to calculate the rho elbow. Current options are:
        'kundu' and 'liberal'. Default='kundu'.
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    This script is currently hard coded for a specific way to calculate the rho elbow
    based on the method by Kundu in the MEICA v2.5 code. To get the same functionality
    in MEICA v2.5, decide_comps must be 'all' and subset_decide_comps must be
    'unclassified' See :obj:`tedana.selection.selection_utils.rho_elbow_kundu_liberal`
    for a more detailed explanation of the difference between the kundu and liberal
    options.
    """
    function_name_idx = f"Step {selector.current_node_idx_}: calc_rho_elbow"

    if rho_elbow_type == "kundu":
        elbow_name = "rho_elbow_kundu"
    elif rho_elbow_type == "liberal":
        elbow_name = "rho_elbow_liberal"
    else:
        raise ValueError(
            f"{function_name_idx}: rho_elbow_type must be 'kundu' or 'liberal' "
            f"It is {rho_elbow_type} "
        )

    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "node_label": None,
        "n_echos": selector.cross_component_metrics_["n_echos"],
        "calc_cross_comp_metrics": [
            elbow_name,
            "rho_allcomps_elbow",
            "rho_unclassified_elbow",
            "elbow_f05",
        ],
        "used_metrics": {"kappa", "rho", "variance explained"},
        elbow_name: None,
        "rho_allcomps_elbow": None,
        "rho_unclassified_elbow": None,
        "elbow_f05": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    if (elbow_name in selector.cross_component_metrics_) and (
        elbow_name in outputs["calc_cross_comp_metrics"]
    ):
        LGR.warning(
            f"{elbow_name} already calculated."
            f"Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Calc Rho Elbow"

    LGR.info(f"{function_name_idx}: {outputs['node_label']}")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    subset_comps2use = selectcomps2use(selector.component_table_, subset_decide_comps)

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:
        (
            outputs[elbow_name],
            outputs["rho_allcomps_elbow"],
            outputs["rho_unclassified_elbow"],
            outputs["elbow_f05"],
        ) = rho_elbow_kundu_liberal(
            selector.component_table_,
            selector.cross_component_metrics_["n_echos"],
            rho_elbow_type=rho_elbow_type,
            comps2use=comps2use,
            subset_comps2use=subset_comps2use,
        )
        selector.cross_component_metrics_[elbow_name] = outputs[elbow_name]
        selector.cross_component_metrics_["rho_allcomps_elbow"] = outputs["rho_allcomps_elbow"]
        selector.cross_component_metrics_["rho_unclassified_elbow"] = outputs[
            "rho_unclassified_elbow"
        ]
        selector.cross_component_metrics_["elbow_f05"] = outputs["elbow_f05"]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def dec_classification_doesnt_exist(
    selector,
    new_classification,
    decide_comps,
    class_comp_exists,
    at_least_num_exist=1,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag=None,
):
    """
    Change the classification of all components in decide_comps.

    This is done if there are no components with a classification specified in
    class_comp_exists.

    Parameters
    ----------
    %(selector)s
    new_classification : :obj:`str`
        Assign all components identified in decide_comps the classification
        in new_classification.
    %(decide_comps)s
    class_comp_exists : :obj:`str` or :obj:`list[str]` or :obj:`int` or :obj:`list[int]`
        This has the same structure options as decide_comps. This function tests
        whether any components in decide_comps have the classifications defined in this
        variable.
    at_least_num_exist : :obj:`int`
        Instead of just testing whether a classification exists, test whether at least
        this number of components have that classification. Default=1
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s
    tag : :obj:`str`
        A classification tag to assign to all components being reclassified.
        This should be one of the tags defined by classification_tags in
        the decision tree specification. Default="".

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    This function is useful to end the component selection process early
    even if there are additional nodes. For example, in the original
    kundu tree, if 0 or 1 components are identified with kappa>elbow and
    rho>elbow then, instead of removing everything, it effectively says
    something's wrong and conservatively keeps everything. Similarly,
    later in the kundu tree, there are several steps deciding how to
    classify any remaining provisional components. If none of the
    remaining components are "provisionalreject" then it skips those
    steps and accepts everything left.
    """
    # predefine all outputs that should be logged
    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "used_metrics": set(),
        "used_cross_comp_metrics": set(),
        "node_label": None,
        "n_true": None,
        "n_false": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = f"Step {selector.current_node_idx_}: classification_doesnt_exist"
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    elif at_least_num_exist == 1:
        outputs["node_label"] = (
            f"Change {decide_comps} to {new_classification} if {class_comp_exists} doesn't exist"
        )
    else:
        outputs["node_label"] = (
            f"Change {decide_comps} to {new_classification} if less than "
            f"{at_least_num_exist} components with {class_comp_exists} exist"
        )

    LGR.info(f"{function_name_idx}: {outputs['node_label']}")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    if_true = new_classification
    if_false = "nochange"

    comps2use = selectcomps2use(selector.component_table_, decide_comps)

    do_comps_exist = selectcomps2use(selector.component_table_, class_comp_exists)

    if (not comps2use) or (len(do_comps_exist) >= at_least_num_exist):
        outputs["n_true"] = 0
        # If nothing chanages, then assign the number of components in comps2use to n_false
        outputs["n_false"] = len(comps2use)
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
            if_true=outputs["n_true"],
            if_false=outputs["n_false"],
        )
    else:  # do_comps_exist is None:
        decision_boolean = pd.Series(True, index=comps2use)

        selector, outputs["n_true"], outputs["n_false"] = change_comptable_classifications(
            selector,
            if_true,
            if_false,
            decision_boolean,
            tag_if_true=tag,
        )

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            n_true=outputs["n_true"],
            n_false=outputs["n_false"],
            if_true=if_true,
            if_false=if_false,
        )

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def dec_reclassify_high_var_comps(
    selector,
    new_classification,
    decide_comps,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    tag=None,
):
    """
    Identify and reclassify a couple components with the largest gaps in variance.

    Parameters
    ----------
    %(selector)s
    new_classification : :obj:`str`
        Assign all components identified in decide_comps the classification
        in new_classification.
    %(decide_comps)s
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s
    tag : :obj:`str`
        A classification tag to assign to all components being reclassified.
        This should be one of the tags defined by classification_tags in
        the decision tree specification. Default="".

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    This function should not exist, but with the goal of maintaining the results of
    the original MEICA decision tree it is necessary, so here it is.
    It is a quirky and brittle step that is used to remove a few higher variance
    components from the calculation of the rho elbow. In the kundu decision tree,
    these components are also excluded from being provisionally accepted if
    kappa>kappa_elbow and rho<rho_elbow.
    """
    # predefine all outputs that should be logged
    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "used_metrics": {"variance explained"},
        "used_cross_comp_metrics": {"varex_upper_p"},
        "node_label": None,
        "n_true": None,
        "n_false": None,
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = f"Step {selector.current_node_idx_}: reclassify_high_var_comps"
    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = (
            f"Change {decide_comps} to {new_classification} for the top couple "
            f"of components with the highest jumps in variance"
        )

    LGR.info(f"{function_name_idx}: {outputs['node_label']}")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    if_true = new_classification
    if_false = "nochange"

    comps2use = selectcomps2use(selector.component_table_, decide_comps)

    if "varex_upper_p" not in selector.cross_component_metrics_:
        if not comps2use:
            LGR.info(
                f"{function_name_idx}: varex_upper_p is not in "
                "selector.cross_component_metrics_, but no components with "
                f"{decide_comps} remain by this node so nothing happens"
            )
        else:
            raise ValueError(
                f"varex_upper_p not in cross_component_metrics. "
                f"It needs to be calculated before {function_name_idx}"
            )

    if not comps2use:
        outputs["n_true"] = 0
        # If nothing changes, then assign the number of components in comps2use to n_false
        outputs["n_false"] = len(comps2use)
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
            if_true=outputs["n_true"],
            if_false=outputs["n_false"],
        )
    else:
        keep_comps2use = comps2use.copy()
        for i_loop in range(3):
            temp_comptable = selector.component_table_.loc[keep_comps2use].sort_values(
                by=["variance explained"], ascending=False
            )
            diff_vals = temp_comptable["variance explained"].diff(-1)
            diff_vals = diff_vals.fillna(0)
            keep_comps2use = temp_comptable.loc[
                diff_vals < selector.cross_component_metrics_["varex_upper_p"]
            ].index.values
        # Everything that should be kept as unclassified is False while the few
        #   that are not in keep_comps2use should be True
        decision_boolean = pd.Series(True, index=comps2use)
        decision_boolean.loc[keep_comps2use] = False

        selector, outputs["n_true"], outputs["n_false"] = change_comptable_classifications(
            selector,
            if_true,
            if_false,
            decision_boolean,
            tag_if_true=tag,
        )

        log_decision_tree_step(
            function_name_idx,
            comps2use,
            n_true=outputs["n_true"],
            n_false=outputs["n_false"],
            if_true=if_true,
            if_false=if_false,
        )

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def calc_varex_thresh(
    selector,
    decide_comps,
    thresh_label,
    percentile_thresh,
    num_highest_var_comps=None,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """Calculate the variance explained threshold to use in the kundu decision tree.

    Will save a high or low percentile threshold depending on highlow_thresh

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    thresh_label : :obj:`str`
        The threshold will be saved in "varex_(thresh_label)_thresh"
        In the original kundu decision tree this was either "upper" or "lower"
        If passed an empty string, will be saved as "varex_thresh"
    percentile_thresh : :obj:`int`
        A percentile threshold to apply to components to set the variance threshold.
        In the original kundu decision tree this was 90 for varex_upper_thresh and
        25 for varex_lower_thresh
    num_highest_var_comps : :obj:`str` :obj:`int`
        percentile can be calculated on the num_highest_var_comps components with the
        lowest variance. Either input an integer directly or input a string that is
        a parameter stored in ``selector.cross_component_metrics_`` ("num_acc_guess" in
        original decision tree). Default=None
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s
    """
    function_name_idx = f"Step {selector.current_node_idx_}: calc_varex_thresh"
    thresh_label = thresh_label.lower()
    if thresh_label is None or thresh_label == "":
        varex_name = "varex_thresh"
        perc_name = "perc"
    else:
        varex_name = f"varex_{thresh_label}_thresh"
        perc_name = f"{thresh_label}_perc"

    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "node_label": None,
        varex_name: None,
        "num_highest_var_comps": num_highest_var_comps,
        "used_metrics": {"variance explained"},
    }
    if (
        isinstance(percentile_thresh, (int, float))
        and (percentile_thresh > 0)
        and (percentile_thresh < 100)
    ):
        outputs["calc_cross_comp_metrics"] = [varex_name, perc_name]
        outputs[perc_name] = percentile_thresh
    else:
        raise ValueError(
            f"percentile_thresh must be a number between 0 & 100. It is: {percentile_thresh}"
        )

    if only_used_metrics:
        return outputs["used_metrics"]

    if varex_name in selector.cross_component_metrics_:
        LGR.warning(
            f"{varex_name} already calculated. Overwriting previous value in {function_name_idx}"
        )

    if perc_name in selector.cross_component_metrics_:
        LGR.warning(
            f"{perc_name} already calculated. Overwriting previous value in {function_name_idx}"
        )

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    if num_highest_var_comps is not None:
        if isinstance(num_highest_var_comps, str):
            if num_highest_var_comps in selector.cross_component_metrics_:
                num_highest_var_comps = selector.cross_component_metrics_[num_highest_var_comps]
            elif not comps2use:
                # Note: It is possible the comps2use requested for this function
                #  is not empty, but the comps2use requested to calculate
                # {num_highest_var_comps} was empty. Given the way this node is
                # used, that's unlikely, but worth a comment.
                LGR.info(
                    f"{function_name_idx}:  num_highest_var_comps ( {num_highest_var_comps}) "
                    "is not in selector.cross_component_metrics_, but no components with "
                    f"{decide_comps} remain by this node so nothing happens"
                )
            else:
                raise ValueError(
                    f"{function_name_idx}: num_highest_var_comps ( {num_highest_var_comps}) "
                    "is not in selector.cross_component_metrics_"
                )
        if not isinstance(num_highest_var_comps, int) and comps2use:
            raise ValueError(
                f"{function_name_idx}: num_highest_var_comps ( {num_highest_var_comps}) "
                "is used as an array index and should be an integer"
            )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = f"Calc {varex_name}, {percentile_thresh}th percentile threshold"

    LGR.info(f"{function_name_idx}: {outputs['node_label']}")
    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:
        if num_highest_var_comps is None:
            outputs[varex_name] = scoreatpercentile(
                selector.component_table_.loc[comps2use, "variance explained"], percentile_thresh
            )
        else:
            # Using only the first num_highest_var_comps components sorted to include
            # lowest variance
            if num_highest_var_comps > len(comps2use):
                # NOTE: This was originally an error, but the original tedana code has no check
                #       at all and it looks like sorted_varex[:num_highest_var_comps] does not
                #       crash and always maxes out at the length of sorted_varex. Since this is
                #       an edge case, decided to print an info message and change the value even
                #       if this won't affect functionality
                LGR.info(
                    f"{function_name_idx}: num_highest_var_comps ({num_highest_var_comps}) > "
                    f"len(comps2use) ({len(comps2use)}). Setting to equal len(comps2use) since "
                    "selection should not use more components than exist"
                )
                num_highest_var_comps = len(comps2use)

            sorted_varex = np.flip(
                np.sort(
                    (selector.component_table_.loc[comps2use, "variance explained"]).to_numpy()
                )
            )
            outputs[varex_name] = scoreatpercentile(
                sorted_varex[:num_highest_var_comps], percentile_thresh
            )

        selector.cross_component_metrics_[varex_name] = outputs[varex_name]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def calc_extend_factor(
    selector,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
    extend_factor=None,
):
    """Calculate the scalar used to set a threshold for d_table_score.

    2 if fewer than 90 fMRI volumes, 3 if more than 110 and linear in-between
    The explanation for the calculation is in
    :obj:`tedana.selection.selection_utils.get_extend_factor`

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s
    extend_factor : :obj:`float`
        If a number, then use rather than calculating anything.
        If None than calculate. default=None

    Returns
    -------
    %(selector)s
    %(used_metrics)s
    """
    outputs = {
        "used_metrics": set(),
        "decision_node_idx": selector.current_node_idx_,
        "node_label": None,
        "extend_factor": None,
        "calc_cross_comp_metrics": ["extend_factor"],
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    function_name_idx = f"Step {selector.current_node_idx_}: calc_extend_factor"

    if "extend_factor" in selector.cross_component_metrics_:
        LGR.warning(
            f"extend_factor already calculated. Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Calc extend_factor"

    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    outputs["extend_factor"] = get_extend_factor(
        n_vols=selector.cross_component_metrics_["n_vols"], extend_factor=extend_factor
    )

    selector.cross_component_metrics_["extend_factor"] = outputs["extend_factor"]

    log_decision_tree_step(function_name_idx, -1, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def calc_max_good_meanmetricrank(
    selector,
    decide_comps,
    metric_suffix=None,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """Calculate the metric "max_good_meanmetricrank".

    Calculates the max_good_meanmetricrank to use in the kundu decision tree.
    This is the number of components selected with decide_comps * the extend_factor
    calculated in calc_extend_factor

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    metric_suffix : :obj:`str`
        By default, this will output a value called "max_good_meanmetricrank"
        If this variable is not None or "" then it will output:
        "max_good_meanmetricrank_[metric_suffix]". Default=None
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    "meanmetricrank" is the same as "d_table_score" and is used to set a threshold for
    the "d_table" values in the component table. This metric ranks the components based
    on 5 metrics and then outputs the mean rank across the 5 metrics.
    Thus "meanmetricrank" is a slightly more descriptive name but d_table was used in
    earlier versions of this code. It might be worth consistently using the same term,
    but this note will hopefully suffice for now.
    """
    function_name_idx = f"Step {selector.current_node_idx_}: calc_max_good_meanmetricrank"

    if (metric_suffix is not None) and (metric_suffix != "") and isinstance(metric_suffix, str):
        metric_name = f"max_good_meanmetricrank_{metric_suffix}"
    else:
        metric_name = "max_good_meanmetricrank"

    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "node_label": None,
        metric_name: None,
        "used_metrics": set(),
        "calc_cross_comp_metrics": [metric_name],
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    if metric_name in selector.cross_component_metrics_:
        LGR.warning(
            "max_good_meanmetricrank already calculated."
            f"Overwriting previous value in {function_name_idx}"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = f"Calc {metric_name}"

    if log_extra_info:
        LGR.info(f"{function_name_idx} {log_extra_info}")

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:
        num_prov_accept = len(comps2use)
        if "extend_factor" in selector.cross_component_metrics_:
            extend_factor = selector.cross_component_metrics_["extend_factor"]
            outputs[metric_name] = extend_factor * num_prov_accept
        else:
            raise ValueError(
                f"extend_factor needs to be in cross_component_metrics for {function_name_idx}"
            )

        selector.cross_component_metrics_[metric_name] = outputs[metric_name]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def calc_varex_kappa_ratio(
    selector,
    decide_comps,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """Calculate the cross-component metric "kappa_rate".

    Calculates the cross_component_metric kappa_rate for the components in decide_comps
    and then calculate the variance explained / kappa ratio for ALL components
    and adds those values to a new column in the component_table titled "varex kappa ratio".

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    These measures are used in the original kundu decision tree.
    kappa_rate = (max-min kappa values of selected components)/(max-min variance explained)
    varex_k.
    varex kappa ratio = kappa_rate * "variance explained"/"kappa" for each component.
    Components with larger variance and smaller kappa are more likely to be rejected.
    This metric sometimes causes issues with high magnitude BOLD responses
    such as the V1 response to a block-design flashing checkerboard
    """
    function_name_idx = f"Step {selector.current_node_idx_}: calc_varex_kappa_ratio"

    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "node_label": None,
        "kappa_rate": None,
        "used_metrics": {"kappa", "variance explained"},
        "calc_cross_comp_metrics": ["kappa_rate"],
        "added_component_table_metrics": ["varex kappa ratio"],
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    if "kappa_rate" in selector.cross_component_metrics_:
        LGR.warning(
            f"kappa_rate already calculated. Overwriting previous value in {function_name_idx}"
        )

    if "varex kappa ratio" in selector.component_table_:
        raise ValueError(
            "'varex kappa ratio' is already a column in the component_table."
            f"Recalculating in {function_name_idx} can cause problems since these "
            "are only calculated on a subset of components"
        )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Calc varex kappa ratio"

    if log_extra_info:
        LGR.info(f"{function_name_idx}: {log_extra_info}")

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:
        kappa_rate = (
            np.nanmax(selector.component_table_.loc[comps2use, "kappa"])
            - np.nanmin(selector.component_table_.loc[comps2use, "kappa"])
        ) / (
            np.nanmax(selector.component_table_.loc[comps2use, "variance explained"])
            - np.nanmin(selector.component_table_.loc[comps2use, "variance explained"])
        )
        outputs["kappa_rate"] = kappa_rate
        LGR.debug(
            f"{function_name_idx} Kappa rate found to be {kappa_rate} from components "
            f"{comps2use}"
        )
        # NOTE: kappa_rate is calculated on a subset of components while
        #     "varex kappa ratio" is calculated for all compnents
        selector.component_table_["varex kappa ratio"] = (
            kappa_rate
            * selector.component_table_["variance explained"]
            / selector.component_table_["kappa"]
        )
        # Unclear if necessary, but this may clean up a weird issue on passing
        # references in a data frame.
        # See longer comment in selection_utils.comptable_classification_changer
        selector.component_table_ = selector.component_table_.copy()

        selector.cross_component_metrics_["kappa_rate"] = outputs["kappa_rate"]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector


@fill_doc
def calc_revised_meanmetricrank_guesses(
    selector,
    decide_comps,
    restrict_factor=2,
    log_extra_info="",
    custom_node_label="",
    only_used_metrics=False,
):
    """Calculate a new d_table_score (meanmetricrank).

    Parameters
    ----------
    %(selector)s
    %(decide_comps)s
    restrict_factor : :obj:`int` or :obj:`float`
        A scaling factor to scale between num_acc_guess and conservative_guess.
        Default=2.

    %(log_extra_info)s
    %(custom_node_label)s
    %(only_used_metrics)s

    Returns
    -------
    %(selector)s
    %(used_metrics)s

    Note
    ----
    These measures are used in the original kundu decision tree.
    Since the d_table_rank is a mean rank across 5 metrics, those ranks
    will change when they're calculated on a subset of components. It's
    unclear how much the relative magnitudes will change and when the
    recalculation will affect results, but this was in the original
    kundu tree and will be replicated here to allow for comparisions

    This also hard-codes for kappa_elbow_kundu and rho_elbow_kundu in
    the cross component metrics. If someone decides to keep using
    this function with other elbow thresholds, the code would need to
    be altered to account for that.

    This function also saves the following cross_component_metrics:
      - ``num_acc_guess``, a guess on the final number of accepted components,
      - ``restrict_factor``, an input to this function used for scaling,
      - ``conservative_guess``, a conservative guess of the final number of
        accepted components calculated as the ratio of ``num_acc_guess`` to
        ``restrict_factor``.
    """
    function_name_idx = f"Step {selector.current_node_idx_}: calc_revised_meanmetricrank_guesses"

    outputs = {
        "decision_node_idx": selector.current_node_idx_,
        "node_label": None,
        "num_acc_guess": None,
        "conservative_guess": None,
        "restrict_factor": None,
        "used_metrics": {
            "kappa",
            "dice_FT2",
            "signal-noise_t",
            "countnoise",
            "countsigFT2",
            "rho",
        },
        "used_cross_component_metrics": {"kappa_elbow_kundu", "rho_elbow_kundu"},
        "calc_cross_comp_metrics": ["num_acc_guess", "conservative_guess", "restrict_factor"],
        "added_component_table_metrics": [f"d_table_score_node{selector.current_node_idx_}"],
    }

    if only_used_metrics:
        return outputs["used_metrics"]

    if "num_acc_guess" in selector.cross_component_metrics_:
        LGR.warning(
            f"num_acc_guess already calculated. Overwriting previous value in {function_name_idx}"
        )

    if "conservative_guess" in selector.cross_component_metrics_:
        LGR.warning(
            "conservative_guess already calculated. "
            f"Overwriting previous value in {function_name_idx}"
        )

    if "restrict_factor" in selector.cross_component_metrics_:
        LGR.warning(
            "restrict_factor already calculated. "
            f"Overwriting previous value in {function_name_idx}"
        )
    if not isinstance(restrict_factor, (int, float)):
        raise ValueError(f"restrict_factor needs to be a number. It is: {restrict_factor}")

    if f"d_table_score_node{selector.current_node_idx_}" in selector.component_table_:
        raise ValueError(
            f"d_table_score_node{selector.current_node_idx_} is already a column"
            f"in the component_table. Recalculating in {function_name_idx} can "
            "cause problems since these are only calculated on a subset of components"
        )

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    for xcompmetric in outputs["used_cross_component_metrics"]:
        if xcompmetric not in selector.cross_component_metrics_:
            if not comps2use:
                LGR.info(
                    f"{function_name_idx}: {xcompmetric} is not in "
                    "selector.cross_component_metrics_, but no components with "
                    f"{decide_comps} remain by this node so nothing happens"
                )
            else:
                raise ValueError(
                    f"{xcompmetric} not in cross_component_metrics. "
                    f"It needs to be calculated before {function_name_idx}"
                )

    if custom_node_label:
        outputs["node_label"] = custom_node_label
    else:
        outputs["node_label"] = "Calc revised d_table_score & num accepted component guesses"

    LGR.info(f"{function_name_idx}: {outputs['node_label']}")
    if log_extra_info:
        LGR.info(f"{function_name_idx}: {log_extra_info}")

    comps2use = selectcomps2use(selector.component_table_, decide_comps)
    confirm_metrics_exist(
        selector.component_table_, outputs["used_metrics"], function_name=function_name_idx
    )

    if not comps2use:
        log_decision_tree_step(
            function_name_idx,
            comps2use,
            decide_comps=decide_comps,
        )
    else:
        outputs["restrict_factor"] = restrict_factor
        outputs["num_acc_guess"] = int(
            np.mean(
                [
                    np.sum(
                        (
                            selector.component_table_.loc[comps2use, "kappa"]
                            > selector.cross_component_metrics_["kappa_elbow_kundu"]
                        )
                        & (
                            selector.component_table_.loc[comps2use, "rho"]
                            < selector.cross_component_metrics_["rho_elbow_kundu"]
                        )
                    ),
                    np.sum(
                        selector.component_table_.loc[comps2use, "kappa"]
                        > selector.cross_component_metrics_["kappa_elbow_kundu"]
                    ),
                ]
            )
        )
        outputs["conservative_guess"] = outputs["num_acc_guess"] / outputs["restrict_factor"]

        tmp_kappa = selector.component_table_.loc[comps2use, "kappa"].to_numpy()
        tmp_dice_ft2 = selector.component_table_.loc[comps2use, "dice_FT2"].to_numpy()
        tmp_signal_m_noise_t = selector.component_table_.loc[
            comps2use, "signal-noise_t"
        ].to_numpy()
        tmp_countnoise = selector.component_table_.loc[comps2use, "countnoise"].to_numpy()
        tmp_countsig_ft2 = selector.component_table_.loc[comps2use, "countsigFT2"].to_numpy()
        tmp_d_table_score = generate_decision_table_score(
            tmp_kappa, tmp_dice_ft2, tmp_signal_m_noise_t, tmp_countnoise, tmp_countsig_ft2
        )
        selector.component_table_[f"d_table_score_node{selector.current_node_idx_}"] = np.NaN
        selector.component_table_.loc[
            comps2use, f"d_table_score_node{selector.current_node_idx_}"
        ] = tmp_d_table_score
        # Unclear if necessary, but this may clean up a weird issue on passing
        # references in a data frame.
        # See longer comment in selection_utils.comptable_classification_changer
        selector.component_table_ = selector.component_table_.copy()

        selector.cross_component_metrics_["conservative_guess"] = outputs["conservative_guess"]
        selector.cross_component_metrics_["num_acc_guess"] = outputs["num_acc_guess"]
        selector.cross_component_metrics_["restrict_factor"] = outputs["restrict_factor"]

        log_decision_tree_step(function_name_idx, comps2use, calc_outputs=outputs)

    selector.tree["nodes"][selector.current_node_idx_]["outputs"] = outputs

    return selector
