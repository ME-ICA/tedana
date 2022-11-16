"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging

import numpy as np

from tedana.metrics import collect
from tedana.selection.ComponentSelector import ComponentSelector
from tedana.selection.selection_utils import clean_dataframe

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def manual_selection(comptable, acc=None, rej=None):
    """
    Perform manual selection of components.

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table, where `C` is components and `M` is metrics
    acc : :obj:`list`, optional
        List of accepted components. Default is None.
    rej : :obj:`list`, optional
        List of rejected components. Default is None.

    Returns
    -------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table with classification.
    metric_metadata : :obj:`dict`
        Dictionary with metadata about calculated metrics.
        Each entry corresponds to a column in ``comptable``.
    """
    LGR.info("Performing manual ICA component selection")
    RepLGR.info(
        "Next, components were manually classified as "
        "BOLD (TE-dependent), non-BOLD (TE-independent), or "
        "uncertain (low-variance)."
    )
    # NOTE: during a merge conflict this got split oddly in a diff
    # Please pay attention to this part to make sure it makes sense
    if (
        "classification" in comptable.columns
        and "original_classification" not in comptable.columns
    ):
        comptable["original_classification"] = comptable["classification"]
        # comptable["original_rationale"] = comptable["rationale"]

    # comptable["rationale"] = ""

    all_comps = comptable.index.values
    if acc is not None:
        acc = [int(comp) for comp in acc]

    if rej is not None:
        rej = [int(comp) for comp in rej]

    if acc is not None and rej is None:
        rej = sorted(np.setdiff1d(all_comps, acc))
    elif acc is None and rej is not None:
        acc = sorted(np.setdiff1d(all_comps, rej))
    elif acc is None and rej is None:
        LGR.info("No manually accepted or rejected components supplied. Accepting all components.")
        # Accept all components if no manual selection provided
        acc = all_comps[:]
        rej = []

    ign = np.setdiff1d(all_comps, np.union1d(acc, rej))
    comptable.loc[acc, "classification"] = "accepted"
    comptable.loc[rej, "classification"] = "rejected"
    # TODO Need to fix classification_tags here to better interact with any previous tags
    # comptable.loc[rej, "classification_tags"] += "Manual"
    comptable.loc[ign, "classification"] = "ignored"
    # comptable.loc[ign, "classification_tags"] += "Manual"

    # Move decision columns to end
    comptable = clean_dataframe(comptable)
    metric_metadata = collect.get_metadata(comptable)
    return comptable, metric_metadata


def automatic_selection(comptable, n_echos, n_vols, tree="minimal"):
    """Classify components based on component table and tree type.

    Parameters
    ----------
    comptable: pd.DataFrame
        The component table to classify
    n_echos: int
        The number of echoes in this dataset
    tree: str
        The type of tree to use for the ComponentSelector object

    Returns
    -------
    A dataframe of the component table, after classification and reorder
    The metadata associated with the component table

    See Also
    --------
    ComponentSelector, the class used to represent the classification process
    Notes
    -----
    The selection algorithm used in this function was originated in ME-ICA
    by Prantik Kundu, and his original implementation is available at:
    https://github.com/ME-ICA/me-ica/blob/\
    b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model.py

    The appropriate citation is :footcite:t:`kundu2013integrated`.

    This component selection process uses multiple, previously calculated
    metrics that include kappa, rho, variance explained, noise and spatial
    frequency metrics, and measures of spatial overlap across metrics.

    Prantik began to update these selection criteria to use SVMs to distinguish
    components, a hypercommented version of this attempt is available at:
    https://gist.github.com/emdupre/ca92d52d345d08ee85e104093b81482e

    References
    ----------
    .. footbibliography::
    """
    LGR.info("Performing ICA component selection with Kundu decision tree v2.5")
    RepLGR.info(
        "Next, component selection was performed to identify "
        "BOLD (TE-dependent), non-BOLD (TE-independent), and "
        "uncertain (low-variance) components using the Kundu "
        "decision tree (v2.5) \\citep{kundu2013integrated}."
    )

    comptable["classification_tags"] = ""
    xcomp = {
        "n_echos": n_echos,
        "n_vols": n_vols,
    }
    selector = ComponentSelector(tree, comptable, cross_component_metrics=xcomp)
    selector.select()
    selector.metadata = collect.get_metadata(selector.component_table)

    return selector
