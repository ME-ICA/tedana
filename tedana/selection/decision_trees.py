"""
Functions that include workflows to identify
TE-dependent and TE-independent components.
"""
import logging
import numpy as np
# from scipy import stats

# from tedana.stats import getfbounds
from tedana.selection._utils import (
    clean_dataframe, confirm_metrics_calculated,
    are_only_necessary_metrics_used)  # getelbow
from tedana.selection.selection_nodes import RhoGtKappa

LGR = logging.getLogger(__name__)


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
    """
    LGR.info('Performing manual ICA component selection')
    if ('classification' in comptable.columns and
            'original_classification' not in comptable.columns):
        comptable['original_classification'] = comptable['classification']
        comptable['original_rationale'] = comptable['rationale']

    comptable['classification'] = 'accepted'
    comptable['rationale'] = ''

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
        LGR.info('No manually accepted or rejected components supplied. '
                 'Accepting all components.')
        # Accept all components if no manual selection provided
        acc = all_comps[:]
        rej = []

    ign = np.setdiff1d(all_comps, np.union1d(acc, rej))
    comptable.loc[acc, 'classification'] = 'accepted'
    comptable.loc[rej, 'classification'] = 'rejected'
    comptable.loc[rej, 'rationale'] += 'I001;'
    comptable.loc[ign, 'classification'] = 'ignored'
    comptable.loc[ign, 'rationale'] += 'I001;'

    # Move decision columns to end
    comptable = clean_dataframe(comptable)
    return comptable


def minimal_decision_tree_A(comptable, n_echos):
    """
    Classify components as "accepted," "rejected," or "ignored" based on
    relevant metrics.

    The selection process uses previously calculated parameters listed in
    comptable for each ICA component such as Kappa (a T2* weighting metric),
    Rho (an S0 weighting metric), and variance explained.
    See `Notes` for additional calculated metrics used to classify each
    component into one of the listed groups.

    Parameters
    ----------
    comptable : (C x M) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    n_echos : :obj:`int`
        Number of echos in original data
        Only used to get threshold for F statistic related to the "elbow" calculation

    Returns
    -------
    comptable : :obj:`pandas.DataFrame`
        Updated component table with additional metrics and with
        classification (accepted, rejected, or ignored)

    Notes
    -----
    The selection algorithm used in this function is a minimalist version based on
    ME-ICA by Prantik Kundu, and his original implementation is available at:
    https://github.com/ME-ICA/me-ica/blob/b2781dd087ab9de99a2ec3925f04f02ce84f0adc/meica.libs/select_model.py

    This component selection process uses multiple, previously calculated
    metrics that include kappa, rho, variance explained, noise and spatial
    frequency metrics, and measures of spatial overlap across metrics.

    For this decision tree:
        4 extreme rejection metrics are applied
        A kappa and rho elbow are calculated and used to reject components
        Potentially rejected components with very low variance explained
        are moved to ignored
    """

    # Documentation of what is happening in this decision tree
    functionname = "minimal_decision_tree_A"
    LGR.info('Performing ICA component selection with ' + functionname)
    LGR.report(" Next, component selection was performed to identify "
               "BOLD (TE-dependent), non-BOLD (TE-independent), and "
               "uncertain (low-variance) components using a reduced, "
               " and conservative version of the Kundu "
               "decision tree (v2.5; Kundu et al., 2013). "
               "4 extreme rejection metrics are applied, "
               "A kappa and rho elbow are calculated and used to "
               "reject components, then potentially rejected "
               "components with very low variance explained "
               "are moved to ignored")
    LGR.refs("Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., "
             "Vértes, P. E., Inati, S. J., ... & Bullmore, E. T. "
             "(2013). Integrated strategy for improving functional "
             "connectivity mapping using multiecho fMRI. Proceedings "
             "of the National Academy of Sciences, 110(40), "
             "16187-16192.")

    necessary_metrics = list(set(['kappa', 'rho',
                                  # currently countsigR2 and countsigS0
                                  'count_sig_in_T2cluster', 'count_sig_in_S0cluster',
                                  'DICE_FT2', 'DICE_FS0',
                                  'T2sig_inside-outside_clusters_T',  # currently sigalnoise_t
                                  'varexp']))
    used_metrics = []

    # This will crash the program with an error message if not all necessary_metrics
    # are in the comptable
    confirm_metrics_calculated(comptable, necessary_metrics, functionname=functionname)

    # A decision tree can theoretically start with some things already classified
    # but the current decision tree is assuming a blank slate
    comptable['classification'] = 'unclassified'
    comptable['rationale'] = ''

    # The comptable values for classification & rationale may be updated
    # A new element is added to decision_tree_steps with an index incremented by 1
    # necessary_metrics is a list of metrics used in this specific function

    # Step 1: Rho>Kappa
    (comptable,
     decision_tree_steps,
     tmp_necessary_metrics) = RhoGtKappa(comptable, iftrue='reject', iffalse='no_change',
                                         decide_comps='all', kappa_scale=1,
                                         decision_tree_steps=None)
    used_metrics = list(set(used_metrics + tmp_necessary_metrics))

    # Step 2:
    # (comptable,
    #  decision_tree_steps,
    #  tmp_necessary_metrics) = KappaGtElbow(comptable, iftrue='reject', iffalse='no_change',
    #                                        decide_comps='all', kappa_scale=1,
    #                                        decision_tree_steps=decision_tree_steps)
    # used_metrics = list(set(used_metrics + tmp_necessary_metrics))

    # This function checks if all metrics that are declared as necessary
    # are actually used and if any used_metrics weren't explicitly declared
    # If either of these happen, a warning is added to the logger
    are_only_necessary_metrics_used()

    # Move decision columns to end
    comptable = clean_dataframe(comptable)

    return comptable, decision_tree_steps
