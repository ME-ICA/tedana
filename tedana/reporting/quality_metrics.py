"""File to assess quality metrics for components."""

import numpy as np

from tedana.stats import fit_model


def calculate_rejected_components_impact(selector, mixing):
    """Calculate the % variance explained by the rejected components for accepted components.

    The final metric is the weighted sum across accepted components of
    the total variance explained by each accepted component times
    the varianced explained (100*R^2) of each accepted component by the rejected comp time series.
    This quantifies the impact of rejected components on the overall variance explained
    by accepted components.

    Parameters
    ----------
    selector : :obj: tedana.selection.component_selector.ComponentSelector
        Uses `variance explained` and `classification` columns in ``selector.component_table_``.
    mixing : (T [x C]) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is time

    Returns
    -------
    None
        Updates elements the selector object:
        ``selector.component_table_`` has an added column for "Var Exp of rejected to accepted"
        ``selector.cross_component_metrics_`` has an added value
        `total_var_exp_rejected_components_on_accepted`
        `variance explained` is a percentage ranging from 0-100% and
        `total_var_exp_rejected_components_on_accepted` is also a percent with the same range.
    """
    component_table = selector.component_table_

    rej = component_table.loc[component_table["classification"] == "rejected"].index
    acc = component_table.loc[component_table["classification"] == "accepted"].index

    if len(rej) == 0 or len(acc) == 0:
        component_table["Var Exp of rejected to accepted"] = np.nan
        selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"] = np.nan
        return

    rej_arrs, acc_arrs = mixing[:, rej], mixing[:, acc]

    # Calculate the R-squared values for each accepted component
    _, sse, _ = fit_model(rej_arrs, acc_arrs)
    # Note: Since the mixing matrix time series are typically mean centered with stdev 1,
    #    ss_total will be the number of time points (+ rounding errors)
    ss_total = np.sum(
        (acc_arrs - np.tile(np.mean(acc_arrs, axis=0), [acc_arrs.shape[0], 1])) ** 2, axis=0
    )
    r2 = 1 - (sse / ss_total)

    if "Var Exp of rejected to accepted" in component_table.columns:
        # Reset to NaN before calculating for accepted components
        component_table["Var Exp of rejected to accepted"] = np.nan
    else:
        # initialize new column in component table that's left of the classification & tag columns
        num_columns = len(component_table.columns)
        component_table.insert(num_columns - 2, "Var Exp of rejected to accepted", np.nan)

    # Calculated r2 is assigned to accepted components and the rest remain NaN
    component_table.loc[acc, "Var Exp of rejected to accepted"] = 100 * r2

    # Update selector with the overall impact measure
    selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"] = (
        np.sum(
            component_table["Var Exp of rejected to accepted"][acc]
            * component_table["variance explained"][acc]
        )
        / 100
    )
