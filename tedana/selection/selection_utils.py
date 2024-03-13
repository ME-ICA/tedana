"""Utility functions for tedana.selection."""

import logging

import numpy as np

from tedana.stats import getfbounds

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

##############################################################
# Functions that are used for interacting with component_table
##############################################################


def selectcomps2use(component_table, decide_comps):
    """Get a list of component numbers that fit the classification types in ``decide_comps``.

    Parameters
    ----------
    component_table : :obj:`~pandas.DataFrame`
        The component_table with metrics and labels for each ICA component
    decide_comps : :obj:`str` or :obj:`list[str]` or :obj:`list[int]`
        This is string or a list of strings describing what classifications
        of components to operate on, using default or intermediate_classification
        labels. For example: decide_comps='unclassified' means to operate only on
        unclassified components. The label 'all' will operate on all components
        regardess of classification. This can also be used to pass through a list
        of component indices to comps2use

    Returns
    -------
    comps2use : :obj:`list[int]`
        A list of component indices with classifications included in decide_comps
    """
    if "classification" not in component_table:
        raise ValueError("component_table needs a 'classification' column to run selectcomps2use")

    if isinstance(decide_comps, (str, int)):
        decide_comps = [decide_comps]

    if isinstance(decide_comps, list) and (decide_comps[0] == "all"):
        # All components with any string in the classification field are set to True
        comps2use = list(range(component_table.shape[0]))

    elif isinstance(decide_comps, list) and all(isinstance(elem, str) for elem in decide_comps):
        comps2use = []
        for didx in range(len(decide_comps)):
            newcomps2use = component_table.index[
                component_table["classification"] == decide_comps[didx]
            ].tolist()
            comps2use = list(set(comps2use + newcomps2use))

    elif isinstance(decide_comps, list) and all(isinstance(elem, int) for elem in decide_comps):
        # decide_comps is already a list of indices
        if len(component_table) <= max(decide_comps):
            raise ValueError(
                "decide_comps for selectcomps2use is selecting for a component with index"
                f"{max(decide_comps)} (0 indexing) which is greater than the number "
                f"of components: {len(component_table)}"
            )
        elif min(decide_comps) < 0:
            raise ValueError(
                "decide_comps for selectcomps2use is selecting for a component "
                f"with index {min(decide_comps)}, which is less than 0"
            )
        else:
            comps2use = decide_comps
    else:
        raise ValueError(
            "decide_comps in selectcomps2use needs to be a list or a single element "
            f"of strings or integers. It is {decide_comps}"
        )

    # If no components are selected, then return None.
    # The function that called this can check for None and exit before
    # attempting any computations on no data
    # if not comps2use:
    #     comps2use = None

    return comps2use


def change_comptable_classifications(
    selector,
    if_true,
    if_false,
    decision_boolean,
    tag_if_true=None,
    tag_if_false=None,
    dont_warn_reclassify=False,
):
    """
    Change or don't change the component classification.

    This happens based on the information on whether a decision critereon is true or
    false for each component.

    Parameters
    ----------
    selector : :obj:`tedana.selection.component_selector.ComponentSelector`
        The attributes used are ``component_table_``, ``component_status_table_``, and
        ``current_node_idx_``
    if_true, if_false : :obj:`str`
        If the condition in this step is true or false, give the component
        the label in this string. Options are 'accepted', 'rejected',
        'nochange', or intermediate_classification labels predefined in the
        decision tree. If 'nochange' then don't change the current component
        classification
    decision_boolean : :obj:`pd.Series(bool)`
        A dataframe column of equal length to component_table where each value
        is True or False.
    tag_if_true, tag_if_false : :obj:`str`
        A string containing a label in classification_tags that will be added to
        the classification_tags column in component_table if a component is
        classified as true or false. default=None
    dont_warn_reclassify : :obj:`bool`
        If this function changes a component classification from accepted or
        rejected to something else, it gives a warning. If this is True, that
        warning is suppressed. default=False

    Returns
    -------
    selector : :obj:`tedana.selection.component_selector.ComponentSelector`
        ``component_table_["classifications"]`` will reflect any new
        classifications.
        ``component_status_table_`` will have a new column titled
        "Node ``current_node_idx_``" that is a copy of the updated classifications
        column.
        ``component_table_["classification_tags"]`` will be updated to include any
        new tags. Each tag should appear only once in the string and tags will
        be separated by commas.
    n_true, n_false : :obj:`int`
        The number of True and False components in decision_boolean

    Note
    ----
    If a classification is changed away from accepted or rejected and
    dont_warn_reclassify is False, then a warning is logged
    """
    selector = comptable_classification_changer(
        selector,
        True,
        if_true,
        decision_boolean,
        tag_if=tag_if_true,
        dont_warn_reclassify=dont_warn_reclassify,
    )
    selector = comptable_classification_changer(
        selector,
        False,
        if_false,
        decision_boolean,
        tag_if=tag_if_false,
        dont_warn_reclassify=dont_warn_reclassify,
    )

    selector.component_status_table_[f"Node {selector.current_node_idx_}"] = (
        selector.component_table_["classification"]
    )

    n_true = decision_boolean.sum()
    n_false = np.logical_not(decision_boolean).sum()
    return selector, n_true, n_false


def comptable_classification_changer(
    selector,
    boolstate,
    classify_if,
    decision_boolean,
    tag_if=None,
    dont_warn_reclassify=False,
):
    """Implement the component classification changes from ``change_comptable_classifications``.

    Parameters
    ----------
    selector : :obj:`tedana.selection.component_selector.ComponentSelector`
        The attributes used are ``component_table_``, ``component_status_table_``, and
        ``current_node_idx_``
    boolstate : :obj:`bool`
        Change classifications only for True or False components in
        decision_boolean based on this variable
    classify_if : :obj:`str`
        This should be if_True or if_False to match boolstate.
        If the condition in this step is true or false, give the component
        the label in this string. Options are 'accepted', 'rejected',
        'nochange', or intermediate_classification labels predefined in the
        decision tree. If 'nochange' then don't change the current component
        classification
    decision_boolean : :obj:`pd.Series(bool)`
        A dataframe column of equal length to component_table where each value
        is True or False.
    tag_if : :obj:`str`
        This should be tag_if_true or tag_if_false to match boolstate
        A string containing a label in classification_tags that will be added to
        the classification_tags column in component_table if a component is
        classified as true or false. default=None
    dont_warn_reclassify : :obj:`bool`
        If this function changes a component classification from accepted or
        rejected to something else, it gives a warning. If this is True, that
        warning is suppressed. default=False

    Returns
    -------
    selector : :obj:`tedana.selection.component_selector.ComponentSelector`
        Operates on the True OR False components depending on boolstate
        ``component_table_["classifications"]`` will reflect any new
        classifications.
        ``component_status_table_`` will have a new column titled
        "Node ``current_node_idx_``" that is a copy of the updated classifications
        column.
        component_table_["classification_tags"] will be updated to include any
        new tags. Each tag should appear only once in the string and tags will
        be separated by commas.

    Warns
    -----
    UserWarning
        If a classification is changed away from accepted or rejected and
        dont_warn_reclassify is False, then a warning is logged

    Note
    ----
    This is designed to be run by
    :func:`~tedana.selection.selection_utils.change_comptable_classifications`.
    This function is run twice, ones for changes to make of a component is
    True and again for components that are False.
    """
    if classify_if != "nochange":
        changeidx = decision_boolean.index[np.asarray(decision_boolean) == boolstate]
        if not changeidx.empty:
            current_classifications = set(
                selector.component_table_.loc[changeidx, "classification"].tolist()
            )
            if current_classifications.intersection({"accepted", "rejected"}):
                if not dont_warn_reclassify:
                    # don't make a warning if classify_if matches the current classification
                    # That is reject->reject shouldn't throw a warning
                    if (
                        ("accepted" in current_classifications) and (classify_if != "accepted")
                    ) or (("rejected" in current_classifications) and (classify_if != "rejected")):
                        LGR.warning(
                            f"Step {selector.current_node_idx_}: Some classifications are"
                            " changing away from accepted or rejected. Once a component is "
                            "accepted or rejected, it shouldn't be reclassified"
                        )
            selector.component_table_.loc[changeidx, "classification"] = classify_if
            # NOTE: CAUTION: extremely bizarre pandas behavior violates guarantee
            # that df['COLUMN'] matches the df as a a whole in this case.
            # We cannot replicate this consistently, but it seems to happen in some
            # datasets where decide_comps does not select all components. We strongly
            # suspect it has something to do with passing via reference a pandas
            # data series.
            # We do not understand why, but copying the table and thus removing references
            # to past memory locations seems to reliably solve this issue.
            # TODO: understand why this happens and avoid the problem without this hack.
            #   Comment line below to re-introduce original bug. For the kundu decision
            #   tree it happens on node 6 which is the first time decide_comps is for
            #   a subset of components
            selector.component_table_ = selector.component_table_.copy()

            if tag_if is not None:  # only run if a tag is provided
                for idx in changeidx:
                    tmpstr = selector.component_table_.loc[idx, "classification_tags"]
                    if tmpstr == "" or isinstance(tmpstr, float):
                        tmpset = {tag_if}
                    else:
                        tmpset = set(tmpstr.split(","))
                        tmpset.update([tag_if])
                    selector.component_table_.loc[idx, "classification_tags"] = ",".join(
                        str(s) for s in tmpset
                    )
        else:
            LGR.info(
                f"Step {selector.current_node_idx_}: No components fit criterion "
                f"{boolstate} to change classification"
            )
    return selector


def clean_dataframe(component_table):
    """
    Reorder columns in component table.

    The reordering is done so that "classification" and "classification_tags" are last.

    Parameters
    ----------
    component_table : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric

    Returns
    -------
    component_table : (C x M) :obj:`pandas.DataFrame`
        Same data as input, but the final two columns are "classification"
        and "classification_tags"
    """
    cols_at_end = ["classification", "classification_tags"]
    component_table = component_table[
        [c for c in component_table if c not in cols_at_end]
        + [c for c in cols_at_end if c in component_table]
    ]

    return component_table


#################################################
# Functions to validate inputs or log information
#################################################


def confirm_metrics_exist(component_table, necessary_metrics, function_name=None):
    """Confirm that all metrics declared in necessary_metrics are already included in comptable.

    Parameters
    ----------
    component_table : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    necessary_metrics : :obj:`list`
        A list of strings of metric names.
    function_name : :obj:`str`
        Text identifying the function name that called this function.

    Raises
    ------
    ValueError
        If ``metrics_exist`` is False then raise an error and end the program.

    Notes
    -----
    This doesn't check if there are data in each metric's column, just that the columns exist.
    Also, the string in ``necessary_metrics`` and the column labels in ``component_table`` will
    only be matched if they're identical.
    """
    missing_metrics = set(necessary_metrics) - set(component_table.columns)
    if missing_metrics:
        function_name = function_name or "unknown function"
        raise ValueError(
            f"Necessary metrics for {function_name}: {necessary_metrics}. "
            f"Comptable metrics: {set(component_table.columns)}. "
            f"MISSING METRICS: {missing_metrics}."
        )


def log_decision_tree_step(
    function_name_idx,
    comps2use,
    decide_comps=None,
    n_true=None,
    n_false=None,
    if_true=None,
    if_false=None,
    calc_outputs=None,
):
    """Log text to add after every decision tree calculation.

    Parameters
    ----------
    function_name_idx : :obj:`str`
        The name of the function that should be logged. By convention, this
        be "Step ``current_node_idx_``: function_name"
    comps2use : :obj:`list[int]` or -1
        A list of component indices that should be used by a function.
        Only used to report no components found if empty and report
        the number of components found if not empty.
        Note: ``calc_`` functions that don't use component metrics do not
        need to use the component_table and may not require selecting
        components. For those functions, set comps2use==-1 to avoid
        logging a warning that no components were found. Currently,
        this is only used by `calc_extend_factor`
    decide_comps : :obj:`str` or :obj:`list[str]` or :obj:`list[int]`
        This is string or a list of strings describing what classifications
        of components to operate on. Only used in this function to report
        its contents if no components with these classifications were found
    n_true, n_false : :obj:`int`
        The number of components classified as True or False
    if_true, if_false : :obj:`str`
        If a component is true or false, the classification to assign that
        component
    calc_outputs : :obj:`dict`
        A dictionary with output information from the function. If it contains a key
        "calc_cross_comp_metrics" then the value for that key is a list of
        cross component metrics (i.e. kappa or rho elbows) that were calculated
        within the function. Each of those metrics will also be a key in calc_outputs
        and those keys and values will be logged by this function

    Returns
    -------
    Information is added to the LGR.info logger. This either logs that
    nothing was changed, the number of components classified as true or
    false and what they changed to, or the cross component metrics that were
    calculated
    """
    if not (comps2use == -1) and not comps2use:
        LGR.info(
            f"{function_name_idx} not applied because no remaining components were "
            f"classified as {decide_comps}"
        )

    if if_true or if_false:
        LGR.info(
            f"{function_name_idx} applied to {len(comps2use)} components. "
            f"{n_true} True -> {if_true}. "
            f"{n_false} False -> {if_false}."
        )

    if calc_outputs:
        if "calc_cross_comp_metrics" in calc_outputs:
            calc_summaries = [
                f"{metric_name}={calc_outputs[metric_name]}"
                for metric_name in calc_outputs["calc_cross_comp_metrics"]
            ]
            LGR.info(f"{function_name_idx} calculated: {', '.join(calc_summaries)}")
        else:
            LGR.warning(
                f"{function_name_idx} logged to write out cross_component_metrics, "
                "but none were calculated"
            )


def log_classification_counts(decision_node_idx, component_table):
    """Log the total counts for each component classification in component_table.

    Parameters
    ----------
    decision_node_idx : :obj:`int`
        The index number for the function in the decision tree that just
        finished executing
    component_table : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Only the "classification" column is usd in this function

    Returns
    -------
    The LGR.info logger will add a line like : \
    'Step 4 : Total component classifications: 10 accepted, 5 provisionalreject, 8 rejected'
    """
    classification_labels, label_counts = np.unique(
        component_table["classification"].values, return_counts=True
    )
    label_summaries = [
        f"{label_counts[i]} {label}" for i, label in enumerate(classification_labels)
    ]
    prelude = f"Step {decision_node_idx}: Total component classifications:"
    out_str = f"{prelude} {', '.join(label_summaries)}"
    LGR.info(out_str)


#######################################################
# Calculations that are used in decision tree functions
#######################################################
def getelbow_cons(arr, return_val=False):
    """Elbow using mean/variance method - conservative.

    Parameters
    ----------
    arr : (C,) array_like
        Metric (e.g., Kappa or Rho) values.
    return_val : :obj:`bool`, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    : obj:`int` or :obj:`float`
        Either the elbow index (if return_val is True) or the values at the
        elbow index (if return_val is False)
    """
    if arr.ndim != 1:
        raise ValueError(f"Parameter arr should be 1d, not {arr.ndim}d")

    if not arr.size:
        raise ValueError(
            "Empty array detected during elbow calculation. "
            "This error happens when getelbow_cons is incorrectly called on no components. "
            "If you see this message, please open an issue at "
            "https://github.com/ME-ICA/tedana/issues with the full traceback and any data "
            "necessary to reproduce this error, so that we create additional data checks to "
            "prevent this from happening."
        )

    arr = np.sort(arr)[::-1]
    nk = len(arr)
    temp1 = [
        (arr[nk - 5 - ii - 1] > arr[nk - 5 - ii : nk].mean() + 2 * arr[nk - 5 - ii : nk].std())
        for ii in range(nk - 5)
    ]
    ds = np.array(temp1[::-1], dtype=np.int32)
    dsum = []
    c_ = 0
    for d_ in ds:
        c_ = (c_ + d_) * d_
        dsum.append(c_)

    e2 = np.argmax(np.array(dsum))
    elind = np.max([getelbow(arr), e2])

    if return_val:
        return arr[elind]
    else:
        return elind


def getelbow(arr, return_val=False):
    """Get elbow using linear projection method - moderate.

    Parameters
    ----------
    arr : (C,) array_like
        Metric (e.g., Kappa or Rho) values.
    return_val : :obj:`bool`, optional
        Return the value of the elbow instead of the index. Default: False

    Returns
    -------
    : obj:`int` or :obj:`float`
        Either the elbow index (if return_val is True) or the values at the
        elbow index (if return_val is False)
    """
    if arr.ndim != 1:
        raise ValueError(f"Parameter arr should be 1d, not {arr.ndim}d")

    if not arr.size:
        raise ValueError(
            "Empty array detected during elbow calculation. "
            "This error happens when getelbow is incorrectly called on no components. "
            "If you see this message, please open an issue at "
            "https://github.com/ME-ICA/tedana/issues with the full traceback and any data "
            "necessary to reproduce this error, so that we create additional data checks to "
            "prevent this from happening."
        )

    arr = np.sort(arr)[::-1]
    n_components = arr.shape[0]
    coords = np.array([np.arange(n_components), arr])
    p = coords - coords[:, 0].reshape(2, 1)
    b = p[:, -1]
    b_hat = np.reshape(b / np.sqrt((b**2).sum()), (2, 1))
    proj_p_b = p - np.dot(b_hat.T, p) * np.tile(b_hat, (1, n_components))
    d = np.sqrt((proj_p_b**2).sum(axis=0))
    k_min_ind = d.argmax()

    if return_val:
        return arr[k_min_ind]
    else:
        return k_min_ind


def kappa_elbow_kundu(component_table, n_echos, comps2use=None):
    """
    Calculate an elbow for kappa.

    Uses the approach originally in Prantik Kundu's MEICA v2.5 code.

    Parameters
    ----------
    component_table : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
        Only the 'kappa' column is used in this function
    n_echos : :obj:`int`
        The number of echos in the multi-echo data
    comps2use : :obj:`list[int]`
        A list of component indices used to calculate the elbow
        default=None which means use all components

    Returns
    -------
    kappa_elbow : :obj:`float`
        The 'elbow' value for kappa values, above which components are considered
        more likely to contain T2* weighted signals.
        minimum of kappa_allcomps_elbow and kappa_nonsig_elbow
    kappa_allcomps_elbow : :obj:`float`
        The elbow for kappa values using all components in comps2use
    kappa_nonsig_elbow : :obj:`float`
        The elbow for kappa values excluding kappa values above a threshold
        None if there are fewer than 6 values remaining after thresholding
    varex_upper_p : :obj:`float`
        This is the median "variance explained" across components with kappa values
        greater than the kappa_elbow calculated using all components

    Note
    ----
    The kappa elbow calculation in Kundu's original meica code calculates
    one elbow using all components' kappa values, one elbow excluding kappa
    values above a threshold, and then selects the lower of the two thresholds.
    This is replicated by setting comps2use to None or by giving a list that
    includes all component numbers. If comps2use includes indices for only a
    subset of components then the kappa values from just those components
    will be used for both elbow calculations.

    varex_upper_p isn't used for anything in this function, but it is calculated
    on kappa values and is used in rho_elbow_kundu_liberal. For several reasons
    it made more sense to calculate here.
    """
    # If comps2use is None then set to a list of all component numbers
    if not comps2use:
        comps2use = list(range(component_table.shape[0]))
    kappas2use = component_table.loc[comps2use, "kappa"].to_numpy()

    # low kappa threshold
    _, _, f01 = getfbounds(n_echos)
    # get kappa values for components below a significance threshold
    kappas_nonsig = kappas2use[kappas2use < f01]

    kappa_allcomps_elbow = getelbow(kappas2use, return_val=True)
    # How often would an elbow from all Kappa values ever be lower than one from
    # a subset of lower values?
    # Note: Only use the subset of values if it includes at least 6 data points
    #  That is enough to calculate an elbow of a curve
    #  This is an arbitrary threshold not from the original meica and is
    #  worth reconsidering at some point
    if kappas_nonsig.size >= 6:
        kappa_nonsig_elbow = getelbow(kappas_nonsig, return_val=True)

        kappa_elbow = np.min((kappa_nonsig_elbow, kappa_allcomps_elbow))
        LGR.info("Calculating kappa elbow based on min of all and nonsig components.")
    else:
        kappa_elbow = kappa_allcomps_elbow
        kappa_nonsig_elbow = None
        LGR.info("Calculating kappa elbow based on all components.")

    # Calculating varex_upper_p
    # Upper limit for variance explained is median across components with high
    # Kappa values. High Kappa is defined as Kappa above Kappa elbow.
    high_kappa_idx = np.squeeze(np.argwhere(kappas2use > kappa_allcomps_elbow))
    # list(kappa_comps2use.index[kappas2use > kappa_allcomps_elbow])
    varex_upper_p = np.median(
        component_table.loc[
            high_kappa_idx,
            "variance explained",
        ]
    )

    return kappa_elbow, kappa_allcomps_elbow, kappa_nonsig_elbow, varex_upper_p


def rho_elbow_kundu_liberal(
    component_table, n_echos, rho_elbow_type="kundu", comps2use=None, subset_comps2use=-1
):
    """
    Calculate an elbow for rho.

    Uses the approach originally in Prantik Kundu's MEICA v2.5 code
    and with a slightly more liberal threshold.

    Parameters
    ----------
    component_table : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
        Only the 'kappa' column is used in this function
    n_echos : :obj:`int`
        The number of echos in the multi-echo data
    rho_elbow_type : :obj:`str`
        The algorithm used to calculate the rho elbow. Current options are
        'kundu' and 'liberal'.
    comps2use : :obj:`list[int]`
        A list of component indices used to calculate the elbow
        default=None which means use all components
    subset_comps2use : :obj:`list[int]`
        A list of component indices used to calculate the elbow
        If None then only calculate a threshold using all components
        default=-1 which means use only 'unclassified' components

    Returns
    -------
    rho_elbow : :obj:`float`
        The 'elbow' value for rho values, above which components are considered
        more likely to contain S0 weighted signals
    rho_allcomps_elbow : :obj:`float`
        rho elbow calculated using all components in comps2use
    rho_unclassified_elbow : :obj:`float`
        rho elbow clculated using all components in subset_comps2use
        None if subset_comps2use is None
    elbow_f05 : :obj:`float`
        A significant threshold based on the number of echoes. Used
        as part of the mean for rho_elbow_type=='kundu'

    Note
    ----
    The rho elbow calculation in Kundu's original meica code calculates
    one elbow using all components' rho values, one elbow using only
    unclassified components (excluding 2-3 remaining high variance componetns),
    on threshold based on the number of echos, and takes the mean of those 3 values
    To replicate the original code, comps2use should include indices for all components
    and subset_comps2use should includes indices for unclassified components

    Also, in practice, one of these elbows is sometimes extremely low and the
    mean creates an overly agressive rho threshold (values >rho_elbow are more
    likely rejected). The liberal threshold option takes the max of the two
    elbows based on rho values. The assumption is that the threshold on
    unclassified components is always lower and can likely be excluded. Both
    rho elbows are now logged so that it will be possible to confirm this with
    data & make additional adjustments to this threshold.

    Additionally, the liberal threshold does not exclude 2-3 high variance components
    from the unclassified threshold. This was done as a practical matter because
    those components are now removed in a separate node, dec_reclassify_high_var_comps,
    and adding that separate node to the minimal tree would make it less minimal, but
    it also seems reasonable since there was no clear reason why they elbow with them
    removed was reliably better than the elbow containing them. More direct comparisons
    between these two arbitrary thresholds might be useful at some point.
    """
    if rho_elbow_type not in ["kundu", "liberal"]:
        raise ValueError(
            f"rho_elbow_kundu_liberal: rho_elbow_type must be 'kundu' or 'liberal'"
            f"It is {rho_elbow_type} "
        )

    # If comps2use is None then set to a list of all component numbers
    if not comps2use:
        comps2use = list(range(component_table.shape[0]))

    # If subset_comps2use is -1 then set to a list of all unclassified components
    if subset_comps2use == -1:
        subset_comps2use = component_table.index[
            component_table["classification"] == "unclassified"
        ].tolist()

    # One rho elbow threshold set just on the number of echoes
    elbow_f05, _, _ = getfbounds(n_echos)

    # One rho elbow threshold set using all componets in comps2use
    rhos_comps2use = component_table.loc[comps2use, "rho"].to_numpy()
    rho_allcomps_elbow = getelbow(rhos_comps2use, return_val=True)

    # low kappa threshold
    # get kappa values for components below a significance threshold
    # kappas_nonsig = kappas2use[kappas2use < f01]

    # Only calculate
    if not subset_comps2use:
        LGR.warning(
            "No unclassified components for rho elbow calculation only elbow based "
            "on all components is used"
        )
        rho_unclassified_elbow = None
        rho_elbow = rho_allcomps_elbow

    else:
        rho_unclassified_elbow = getelbow(
            component_table.loc[subset_comps2use, "rho"], return_val=True
        )

        if rho_elbow_type == "kundu":
            rho_elbow = np.mean((rho_allcomps_elbow, rho_unclassified_elbow, elbow_f05))
        else:  # rho_elbow_type == 'liberal'
            rho_elbow = np.maximum(rho_allcomps_elbow, rho_unclassified_elbow)

    return rho_elbow, rho_allcomps_elbow, rho_unclassified_elbow, elbow_f05


def get_extend_factor(n_vols=None, extend_factor=None):
    """
    Get the extend_factor for the kundu decision tree.

    Extend_factor is a scaler used to set a threshold for the d_table_score in
    the kundu decision tree.

    It is either defined by the number of volumes in the time series or directly
    defined by the user. If it is defined by the user, that takes precedence over
    using the number of volumes in a calculation

    Parameters
    ----------
    n_vols : :obj:`int`
        The number of volumes in an fMRI time series. default=None
        In the MEICA code, extend_factor was hard-coded to 2 for data with more
        than 100 volumes and 3 for data with less than 100 volumes.
        Now is linearly ramped from 2-3 for vols between 90 & 110
    extend_factor : :obj:`float`
        The scaler used to set a threshold for d_table_score. default=None

    Returns
    -------
    extend_factor : :obj:`float`

    Note
    ----
    Either n_vols OR extend_factor is a required input
    """
    if extend_factor:
        if isinstance(extend_factor, int):
            extend_factor = float(extend_factor)
        LGR.info(f"extend_factor={extend_factor}, as defined by user")
    elif n_vols:
        if n_vols < 90:
            extend_factor = 3.0
        elif n_vols < 110:
            extend_factor = 2.0 + (n_vols - 90) / 20.0
        else:
            extend_factor = 2.0
        LGR.info(f"extend_factor={extend_factor}, based on number of fMRI volumes")
    else:
        error_msg = "get_extend_factor need n_vols or extend_factor as an input"
        raise ValueError(error_msg)

    return extend_factor
