"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging

import numpy as np
from scipy import stats

from tedana.metrics import collect
from tedana.selection._utils import clean_dataframe, getelbow
from tedana.stats import getfbounds

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
    if (
        "classification" in comptable.columns
        and "original_classification" not in comptable.columns
    ):
        comptable["original_classification"] = comptable["classification"]
        comptable["original_rationale"] = comptable["rationale"]

    comptable["classification"] = "accepted"
    comptable["rationale"] = ""

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
    comptable.loc[rej, "rationale"] += "I001;"
    comptable.loc[ign, "classification"] = "ignored"
    comptable.loc[ign, "rationale"] += "I001;"

    # Move decision columns to end
    comptable = clean_dataframe(comptable)
    metric_metadata = collect.get_metadata(comptable)
    return comptable, metric_metadata


def kundu_selection_v2(comptable, n_echos, n_vols):
    """
    Classify components as "accepted", "rejected", or "ignored" based on
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
    n_vols : :obj:`int`
        Number of volumes in dataset

    Returns
    -------
    comptable : :obj:`pandas.DataFrame`
        Updated component table with additional metrics and with
        classification (accepted, rejected, or ignored)
    metric_metadata : :obj:`dict`
        Dictionary with metadata about calculated metrics.
        Each entry corresponds to a column in ``comptable``.

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

    comptable["classification"] = "accepted"
    comptable["rationale"] = ""

    # Set knobs
    LOW_PERC = 25
    HIGH_PERC = 90
    if n_vols < 100:
        EXTEND_FACTOR = 3
    else:
        EXTEND_FACTOR = 2
    RESTRICT_FACTOR = 2

    # Lists of components
    all_comps = np.arange(comptable.shape[0])
    # unclf is a full list that is whittled down over criteria
    # since the default classification is "accepted", at the end of the tree
    # the remaining elements in unclf are classified as accepted
    unclf = all_comps.copy()

    """
    Step 1: Reject anything that's obviously an artifact
    a. Estimate a null variance
    """
    # Rho is higher than Kappa
    temp_rej0a = all_comps[(comptable["rho"] > comptable["kappa"])]
    comptable.loc[temp_rej0a, "classification"] = "rejected"
    comptable.loc[temp_rej0a, "rationale"] += "I002;"

    # Number of significant voxels for S0 model is higher than number for T2
    # model *and* number for T2 model is greater than zero.
    temp_rej0b = all_comps[
        ((comptable["countsigFS0"] > comptable["countsigFT2"]) & (comptable["countsigFT2"] > 0))
    ]
    comptable.loc[temp_rej0b, "classification"] = "rejected"
    comptable.loc[temp_rej0b, "rationale"] += "I003;"
    rej = np.union1d(temp_rej0a, temp_rej0b)

    # Dice score for S0 maps is higher than Dice score for T2 maps and variance
    # explained is higher than the median across components.
    temp_rej1 = all_comps[
        (comptable["dice_FS0"] > comptable["dice_FT2"])
        & (comptable["variance explained"] > np.median(comptable["variance explained"]))
    ]
    comptable.loc[temp_rej1, "classification"] = "rejected"
    comptable.loc[temp_rej1, "rationale"] += "I004;"
    rej = np.union1d(temp_rej1, rej)

    # T-value is less than zero (noise has higher F-statistics than signal in
    # map) and variance explained is higher than the median across components.
    temp_rej2 = unclf[
        (comptable.loc[unclf, "signal-noise_t"] < 0)
        & (comptable.loc[unclf, "variance explained"] > np.median(comptable["variance explained"]))
    ]
    comptable.loc[temp_rej2, "classification"] = "rejected"
    comptable.loc[temp_rej2, "rationale"] += "I005;"
    rej = np.union1d(temp_rej2, rej)
    unclf = np.setdiff1d(unclf, rej)

    # Quit early if no potentially accepted components remain
    if len(unclf) == 0:
        LGR.warning("No BOLD-like components detected. Ignoring all remaining components.")
        ign = sorted(np.setdiff1d(all_comps, rej))
        comptable.loc[ign, "classification"] = "ignored"
        comptable.loc[ign, "rationale"] += "I006;"

        # Move decision columns to end
        comptable = clean_dataframe(comptable)
        metric_metadata = collect.get_metadata(comptable)
        return comptable, metric_metadata

    """
    Step 2: Make a guess for what the good components are, in order to
    estimate good component properties
    a. Not outlier variance
    b. Kappa>kappa_elbow
    c. Rho<Rho_elbow
    d. High T2* dice compared to S0 dice
    e. Gain of F_T2 in clusters vs noise
    f. Estimate a low and high variance
    """
    # Step 2a
    # Upper limit for variance explained is median across components with high
    # Kappa values. High Kappa is defined as Kappa above Kappa elbow.
    varex_upper_p = np.median(
        comptable.loc[
            comptable["kappa"] > getelbow(comptable["kappa"], return_val=True),
            "variance explained",
        ]
    )

    # Sort component table by variance explained and find outlier components by
    # change in variance explained from one component to the next.
    # Remove variance-explained outliers from list of components to consider
    # for acceptance. These components will have another chance to be accepted
    # later on.
    # NOTE: We're not sure why this is done this way, nor why it's specifically
    # done three times.
    ncls = unclf.copy()
    for i_loop in range(3):
        temp_comptable = comptable.loc[ncls].sort_values(
            by=["variance explained"], ascending=False
        )
        diff_vals = temp_comptable["variance explained"].diff(-1)
        diff_vals = diff_vals.fillna(0)
        ncls = temp_comptable.loc[diff_vals < varex_upper_p].index.values

    # Compute elbows from other elbows
    f05, _, f01 = getfbounds(n_echos)
    kappas_nonsig = comptable.loc[comptable["kappa"] < f01, "kappa"]
    if not kappas_nonsig.size:
        LGR.warning(
            "No nonsignificant kappa values detected. "
            "Only using elbow calculated from all kappa values."
        )
        kappas_nonsig_elbow = np.nan
    else:
        kappas_nonsig_elbow = getelbow(kappas_nonsig, return_val=True)

    kappas_all_elbow = getelbow(comptable["kappa"], return_val=True)

    # NOTE: Would an elbow from all Kappa values *ever* be lower than one from
    # a subset of lower (i.e., nonsignificant) values?
    kappa_elbow = np.nanmin((kappas_all_elbow, kappas_nonsig_elbow))
    rhos_ncls_elbow = getelbow(comptable.loc[ncls, "rho"], return_val=True)
    rhos_all_elbow = getelbow(comptable["rho"], return_val=True)
    rho_elbow = np.mean((rhos_ncls_elbow, rhos_all_elbow, f05))

    # Provisionally accept components based on Kappa and Rho elbows
    acc_prov = ncls[
        (comptable.loc[ncls, "kappa"] >= kappa_elbow) & (comptable.loc[ncls, "rho"] < rho_elbow)
    ]

    # Quit early if no potentially accepted components remain
    if len(acc_prov) <= 1:
        LGR.warning("Too few BOLD-like components detected. Ignoring all remaining.")
        ign = sorted(np.setdiff1d(all_comps, rej))
        comptable.loc[ign, "classification"] = "ignored"
        comptable.loc[ign, "rationale"] += "I006;"

        # Move decision columns to end
        comptable = clean_dataframe(comptable)
        metric_metadata = collect.get_metadata(comptable)
        return comptable, metric_metadata

    # Calculate "rate" for kappa: kappa range divided by variance explained
    # range, for potentially accepted components
    # NOTE: What is the logic behind this?
    kappa_rate = (
        np.max(comptable.loc[acc_prov, "kappa"]) - np.min(comptable.loc[acc_prov, "kappa"])
    ) / (
        np.max(comptable.loc[acc_prov, "variance explained"])
        - np.min(comptable.loc[acc_prov, "variance explained"])
    )
    comptable["kappa ratio"] = kappa_rate * comptable["variance explained"] / comptable["kappa"]

    # Calculate bounds for variance explained
    varex_lower = stats.scoreatpercentile(comptable.loc[acc_prov, "variance explained"], LOW_PERC)
    varex_upper = stats.scoreatpercentile(comptable.loc[acc_prov, "variance explained"], HIGH_PERC)

    """
    Step 3: Get rid of midk components; i.e., those with higher than
    max decision score and high variance
    """
    max_good_d_score = EXTEND_FACTOR * len(acc_prov)
    midk = unclf[
        (comptable.loc[unclf, "d_table_score"] > max_good_d_score)
        & (comptable.loc[unclf, "variance explained"] > EXTEND_FACTOR * varex_upper)
    ]
    comptable.loc[midk, "classification"] = "rejected"
    comptable.loc[midk, "rationale"] += "I007;"
    unclf = np.setdiff1d(unclf, midk)
    acc_prov = np.setdiff1d(acc_prov, midk)

    """
    Step 4: Find components to ignore
    """
    # collect high variance unclassified components
    # and mix of high/low provisionally accepted
    high_varex = np.union1d(
        acc_prov, unclf[comptable.loc[unclf, "variance explained"] > varex_lower]
    )
    # ignore low variance components
    ign = np.setdiff1d(unclf, high_varex)
    # but only if they have bad decision scores
    ign = np.setdiff1d(ign, ign[comptable.loc[ign, "d_table_score"] < max_good_d_score])
    # and low kappa
    ign = np.setdiff1d(ign, ign[comptable.loc[ign, "kappa"] > kappa_elbow])
    comptable.loc[ign, "classification"] = "ignored"
    comptable.loc[ign, "rationale"] += "I008;"
    unclf = np.setdiff1d(unclf, ign)

    """
    Step 5: Scrub the set if there are components that haven't been rejected or
    ignored, but are still not listed in the provisionally accepted group.
    """
    if len(unclf) > len(acc_prov):
        comptable["d_table_score_scrub"] = np.nan
        # Recompute the midk steps on the limited set to clean up the tail
        d_table_rank = np.vstack(
            [
                len(unclf) - stats.rankdata(comptable.loc[unclf, "kappa"]),
                len(unclf) - stats.rankdata(comptable.loc[unclf, "dice_FT2"]),
                len(unclf) - stats.rankdata(comptable.loc[unclf, "signal-noise_t"]),
                stats.rankdata(comptable.loc[unclf, "countnoise"]),
                len(unclf) - stats.rankdata(comptable.loc[unclf, "countsigFT2"]),
            ]
        ).T
        comptable.loc[unclf, "d_table_score_scrub"] = d_table_rank.mean(1)
        num_acc_guess = int(
            np.mean(
                [
                    np.sum(
                        (comptable.loc[unclf, "kappa"] > kappa_elbow)
                        & (comptable.loc[unclf, "rho"] < rho_elbow)
                    ),
                    np.sum(comptable.loc[unclf, "kappa"] > kappa_elbow),
                ]
            )
        )

        # Rejection candidate based on artifact type A: candartA
        conservative_guess = num_acc_guess / RESTRICT_FACTOR
        candartA = np.intersect1d(
            unclf[comptable.loc[unclf, "d_table_score_scrub"] > conservative_guess],
            unclf[comptable.loc[unclf, "kappa ratio"] > EXTEND_FACTOR * 2],
        )
        candartA = candartA[
            comptable.loc[candartA, "variance explained"] > varex_upper * EXTEND_FACTOR
        ]
        comptable.loc[candartA, "classification"] = "rejected"
        comptable.loc[candartA, "rationale"] += "I009;"
        midk = np.union1d(midk, candartA)
        unclf = np.setdiff1d(unclf, midk)

        # Rejection candidate based on artifact type B: candartB
        conservative_guess2 = num_acc_guess * HIGH_PERC / 100.0
        candartB = unclf[comptable.loc[unclf, "d_table_score_scrub"] > conservative_guess2]
        candartB = candartB[
            comptable.loc[candartB, "variance explained"] > varex_lower * EXTEND_FACTOR
        ]
        comptable.loc[candartB, "classification"] = "rejected"
        comptable.loc[candartB, "rationale"] += "I010;"
        midk = np.union1d(midk, candartB)
        unclf = np.setdiff1d(unclf, midk)

        # Find components to ignore
        # Of the remaining components, ignore ones with higher variance even if their
        # decision tree scores are poor, instead of rejecting them
        sorted_varex = np.flip(np.sort(comptable.loc[unclf, "variance explained"].to_numpy()))
        new_varex_lower = stats.scoreatpercentile(sorted_varex[:num_acc_guess], LOW_PERC)
        candart = unclf[comptable.loc[unclf, "d_table_score_scrub"] > num_acc_guess]
        ign_add0 = candart[comptable.loc[candart, "variance explained"] > new_varex_lower]
        ign_add0 = np.setdiff1d(ign_add0, midk)
        comptable.loc[ign_add0, "classification"] = "ignored"
        comptable.loc[ign_add0, "rationale"] += "I011;"
        ign = np.union1d(ign, ign_add0)
        unclf = np.setdiff1d(unclf, ign)

        # Ignore low Kappa, high variance explained components
        ign_add1 = np.intersect1d(
            unclf[comptable.loc[unclf, "kappa"] <= kappa_elbow],
            unclf[comptable.loc[unclf, "variance explained"] > new_varex_lower],
        )
        ign_add1 = np.setdiff1d(ign_add1, midk)
        comptable.loc[ign_add1, "classification"] = "ignored"
        comptable.loc[ign_add1, "rationale"] += "I012;"

    # at this point, unclf is equivalent to accepted

    # Move decision columns to end
    comptable = clean_dataframe(comptable)
    metric_metadata = collect.get_metadata(comptable)
    return comptable, metric_metadata
