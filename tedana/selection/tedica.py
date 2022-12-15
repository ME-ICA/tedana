"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging

from tedana.metrics import collect
from tedana.selection.component_selector import ComponentSelector

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def automatic_selection(component_table, n_echos, n_vols, tree="kundu"):
    """Classify components based on component table and decision tree type.

    Parameters
    ----------
    component_table: :obj:`pd.DataFrame`
        The component table to classify
    n_echos: :obj:`int`
        The number of echoes in this dataset
    tree: :obj:`str`
        The type of tree to use for the ComponentSelector object. Default="kundu"

    Returns
    -------

    selector: :obj:`tedana.selection.component_selector.ComponentSelector`
        Contains component classifications in a component_table and provenance
        and metadata from the component selection process

    Notes
    -----
    If tree=kundu, the selection algorithm used in this function was originated in ME-ICA
    by Prantik Kundu, and his original implementation is available at:
    https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model.py

    The appropriate citation is :footcite:t:`kundu2013integrated`.

    This component selection process uses multiple, previously calculated
    metrics that include kappa, rho, variance explained, noise and spatial
    frequency metrics, and measures of spatial overlap across metrics.

    Prantik began to update these selection criteria to use SVMs to distinguish
    components, a hypercommented version of this attempt is available at:
    https://gist.github.com/emdupre/ca92d52d345d08ee85e104093b81482e

    If tree==minimal, the selection algorithm based on the kundu tree with differences
    described in the `FAQ`_

    References
    ----------
    .. footbibliography::

    .. _FAQ: faq.html
    """
    LGR.info("Performing ICA component selection with Kundu decision tree v2.5")
    RepLGR.info(
        "Next, component selection was performed to identify "
        "BOLD (TE-dependent), non-BOLD (TE-independent), and "
        "uncertain (low-variance) components using the Kundu "
        "decision tree (v2.5) \\citep{kundu2013integrated}."
    )

    component_table["classification_tags"] = ""
    xcomp = {
        "n_echos": n_echos,
        "n_vols": n_vols,
    }
    selector = ComponentSelector(tree, component_table, cross_component_metrics=xcomp)
    selector.select()
    selector.metadata = collect.get_metadata(selector.component_table)

    return selector
