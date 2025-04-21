"""Functions to identify TE-dependent and TE-independent components."""

import logging

import numpy as np

from tedana import utils
from tedana.metrics import collect
from tedana.selection.selection_utils import clean_dataframe, getelbow, getelbow_cons
from tedana.stats import getfbounds

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

F_MAX = 500


def kundu_tedpca(
    component_table, n_echos, n_independent_echos=None, kdaw=10.0, rdaw=1.0, stabilize=False
):
    """Select PCA components using Kundu's decision tree approach.

    Parameters
    ----------
    component_table : :obj:`pandas.DataFrame`
        Component table with relevant metrics: kappa, rho, and normalized
        variance explained. Component number should be the index.
    n_echos : :obj:`int`
        Number of echoes in dataset.
    n_independent_echos : int
        Number of independent echoes to use in goodness of fit metrics (fstat).
        Primarily used for EPTI acquisitions.
        If None, number of echoes will be used. Default is None.
    kdaw : :obj:`float`, optional
        Kappa dimensionality augmentation weight. Must be a non-negative float,
        or -1 (a special value). Default is 10.
    rdaw : :obj:`float`, optional
        Rho dimensionality augmentation weight. Must be a non-negative float,
        or -1 (a special value). Default is 1.
    stabilize : :obj:`bool`, optional
        Whether to stabilize convergence by reducing dimensionality, for low
        quality data. Default is False.

    Returns
    -------
    component_table : :obj:`pandas.DataFrame`
        Component table with components classified as 'accepted', 'rejected',
        or 'ignored'.
    metric_metadata : :obj:`dict`
        Dictionary with metadata about calculated metrics.
        Each entry corresponds to a column in ``component_table``.
    """
    LGR.info("Performing PCA component selection with Kundu decision tree")
    component_table["classification"] = "accepted"
    component_table["rationale"] = ""

    eigenvalue_elbow = getelbow(component_table["normalized variance explained"], return_val=True)

    diff_varex_norm = np.abs(np.diff(component_table["normalized variance explained"]))
    lower_diff_varex_norm = diff_varex_norm[(len(diff_varex_norm) // 2) :]
    varex_norm_thr = np.mean([lower_diff_varex_norm.max(), diff_varex_norm.min()])
    varex_norm_min = component_table["normalized variance explained"][
        (len(diff_varex_norm) // 2)
        + np.arange(len(lower_diff_varex_norm))[lower_diff_varex_norm >= varex_norm_thr][0]
        + 1
    ]
    varex_norm_cum = np.cumsum(component_table["normalized variance explained"])
    n_independent_echos = n_independent_echos or n_echos
    fmin, fmid, fmax = getfbounds(n_independent_echos)

    if int(kdaw) == -1:
        lim_idx = (
            utils.andb([component_table["kappa"] < fmid, component_table["kappa"] > fmin]) == 2
        )
        kappa_lim = component_table.loc[lim_idx, "kappa"].values
        kappa_thr = kappa_lim[getelbow(kappa_lim)]

        lim_idx = utils.andb([component_table["rho"] < fmid, component_table["rho"] > fmin]) == 2
        rho_lim = component_table.loc[lim_idx, "rho"].values
        rho_thr = rho_lim[getelbow(rho_lim)]
        stabilize = True
        LGR.info("kdaw set to -1. Switching TEDPCA algorithm to kundu-stabilize")
    elif int(rdaw) == -1:
        lim_idx = utils.andb([component_table["rho"] < fmid, component_table["rho"] > fmin]) == 2
        rho_lim = component_table.loc[lim_idx, "rho"].values
        rho_thr = rho_lim[getelbow(rho_lim)]
    else:
        kappa_thr = np.average(
            sorted([fmin, (getelbow(component_table["kappa"], return_val=True) / 2), fmid]),
            weights=[kdaw, 1, 1],
        )
        rho_thr = np.average(
            sorted([fmin, (getelbow_cons(component_table["rho"], return_val=True) / 2), fmid]),
            weights=[rdaw, 1, 1],
        )

    # Reject if low Kappa, Rho, and variance explained
    is_lowk = component_table["kappa"] <= kappa_thr
    is_lowr = component_table["rho"] <= rho_thr
    is_lowe = component_table["normalized variance explained"] <= eigenvalue_elbow
    is_lowkre = is_lowk & is_lowr & is_lowe
    component_table.loc[is_lowkre, "classification"] = "rejected"
    component_table.loc[is_lowkre, "rationale"] += "P001;"

    # Reject if low variance explained
    is_lows = component_table["normalized variance explained"] <= varex_norm_min
    component_table.loc[is_lows, "classification"] = "rejected"
    component_table.loc[is_lows, "rationale"] += "P002;"

    # Reject if Kappa over limit
    is_fmax1 = component_table["kappa"] == F_MAX
    component_table.loc[is_fmax1, "classification"] = "rejected"
    component_table.loc[is_fmax1, "rationale"] += "P003;"

    # Reject if Rho over limit
    is_fmax2 = component_table["rho"] == F_MAX
    component_table.loc[is_fmax2, "classification"] = "rejected"
    component_table.loc[is_fmax2, "rationale"] += "P004;"

    if stabilize:
        temp7 = varex_norm_cum >= 0.95
        component_table.loc[temp7, "classification"] = "rejected"
        component_table.loc[temp7, "rationale"] += "P005;"
        under_fmin1 = component_table["kappa"] <= fmin
        component_table.loc[under_fmin1, "classification"] = "rejected"
        component_table.loc[under_fmin1, "rationale"] += "P006;"
        under_fmin2 = component_table["rho"] <= fmin
        component_table.loc[under_fmin2, "classification"] = "rejected"
        component_table.loc[under_fmin2, "rationale"] += "P007;"

    n_components = component_table.loc[component_table["classification"] == "accepted"].shape[0]
    LGR.info(
        f"Selected {n_components} components with Kappa threshold: {kappa_thr:.02f}, Rho "
        f"threshold: {rho_thr:.02f}"
    )

    # Move decision columns to end
    component_table = clean_dataframe(component_table)

    metric_metadata = collect.get_metadata(component_table)
    return component_table, metric_metadata
