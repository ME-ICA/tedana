import numpy as np
import pandas as pd
from tedana.stats import fit_model


def calculate_rejected_components_impact(selector, mixing):
    """
    This function calculates the % variance explained by the rejected components for
    each of the accepted components. The final metric is the weighted sum of the variances.
    This quantifies the impact of rejected components on the overall variance explained by
    accepted components.

    Parameters
    ----------
    selector : :obj: tedana.selection.component_selector.ComponentSelector
        Contains component classifications in component_table.
    mixing : (T [x C]) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`

    Returns
    -------
    None
        Updates the selector object's cross_component_metrics_ and component_table_
    """
    component_table = selector.component_table_

    rej = component_table.loc[component_table["classification"] == "rejected"].index
    acc = component_table.loc[component_table["classification"] == "accepted"].index

    rej_arrs, acc_arrs = mixing[:, rej], mixing[:, acc]

    all_rvals = {}

    # Calculate the R-squared values for each accepted component
    for i, accepted_column in enumerate(acc):
        acc_arr = acc_arrs[:, i]

        _, sse, _ = fit_model(rej_arrs, acc_arr)
        ss_total = np.sum((acc_arr - np.mean(acc_arr)) ** 2)
            
        r2 = 1 - (sse / ss_total)

        all_rvals[accepted_column] = r2

    # Update component table
    component_table["R2 of fit of rejected to accepted"] = component_table.index.map(all_rvals)

    measures = []
    vars_explained = []

    # Calculate the final weighted sum of the variances
    for component, val in all_rvals.items():
        var_explained = component_table.loc[component, "variance explained"].item() / 100
        measures.append(var_explained * val)
        vars_explained.append(var_explained)
    
    # Final QC metric as the sum of the weighted measures
    rejected_components_impact = np.sum(measures)

    # Update selector
    selector.cross_component_metrics_["rejected_components_impact"] = rejected_components_impact