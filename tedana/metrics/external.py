"""Metrics based on fits of component time series to external time series."""

import logging
import re
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from tedana import utils
from tedana.stats import fit_model

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


class RegressError(Exception):
    """Passes errors that are raised when `validate_extern_regress` fails."""

    pass


def load_validate_external_regressors(
    external_regressors: str, external_regressor_config: Dict, n_vols: int
) -> Tuple[pd.DataFrame, Dict]:
    """Load and validate external regressors and descriptors in dictionary.

    Parameters
    ----------
    external_regressors: :obj:`str`
        Path and name of tsv file that includes external regressor time series
    external_regressor_config: :obj:`dict`
        A dictionary with info for fitting external regressors to component time series
    n_vols: :obj:`int`
        Number of timepoints in the fMRI time series

    Returns
    -------
    external_regressors: :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config: :obj:`dict`
        A validated dictionary with info for fitting external regressors to component time series.
        If regex patterns like '^mot_.*$' are used to define regressor names,
        this is replaced with a list of the match column names used in external_regressors
    """
    try:
        external_regressors = pd.read_table(external_regressors)
    except FileNotFoundError:
        raise ValueError(f"Cannot load tsv file with external regressors: {external_regressors}")

    external_regressor_config = validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )

    return external_regressors, external_regressor_config


def validate_extern_regress(
    external_regressors: Dict, external_regressor_config: Dict, n_vols: int
) -> Dict:
    """Confirm external regressor dictionary matches data and expands regular expressions.

    Most keys in external_regressor_config are valided in component_selector.validate_tree
    which is run when a component selector object is initialized.
    This function validates external_regressor_config against the dataset-specific
    external_regressors time series.
    If the config names specific column labels with the f_stats_partial_models key,
    then this confirms those column labels are used in external_regressors
    Also checks if the number of time points in the external regressors matches
    the number of time point in the fMRI data.

    Parameters
    ----------
    external_regressors : :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config : :obj:`dict`
        Information describing the external regressors and
        method to use for fitting and statistical tests
    n_vols : :obj:`int`
        The number of time points in the fMRI time series

    Returns
    -------
    external_regressor_config: :obj:`dict`
        A validated dictionary with info for fitting external regressors
        to component time series.
        If regex patterns like '^mot_.*$' are used to define regressor names,
        this is replaced with a list of the matching column names used in external_regressors

    Raises
    ------
    RegressorError if any validation test fails
    """
    # err_msg is used to collect all errors in validation rather than require this to be run
    # multiple times to see all validation errors. Will either collect errors and raise at the
    # end of the function or raise errors that prevent the rest of the function from completing
    err_msg = ""

    # Validating the information in external_regressor_config works
    # with the data in external_regressors

    # Currently column labels only need to be predefined for calc_stats==F
    # and there are f_stats_partial_models or if "task_keep" is used.
    # column_label_specifications will include the names of the column labels for
    # both f_stats_partial_models and task_keep if either or both are defined.
    column_label_specifications = set()
    if "f_stats_partial_models" in set(external_regressor_config.keys()):
        column_label_specifications.update(
            set(external_regressor_config["f_stats_partial_models"])
        )
    if "task_keep" in set(external_regressor_config.keys()):
        column_label_specifications.update(["task_keep"])

    if column_label_specifications:
        regressor_names = set(external_regressors.columns)
        expected_regressor_names = set()
        # for each column label, check if the label matches column names in external_regressors
        for partial_models in column_label_specifications:
            tmp_partial_model_names = set()
            for tmp_name in external_regressor_config[partial_models]:
                # If a label starts with ^ treat match the pattern to column names using regex
                if tmp_name.startswith("^"):
                    tmp_replacements = [
                        reg_name
                        for reg_name in regressor_names
                        if re.match(tmp_name, reg_name, re.IGNORECASE)
                    ]
                    if not tmp_replacements:
                        err_msg += (
                            f"No external regressor labels matching regex '{tmp_name}' found."
                        )
                    tmp_partial_model_names.update(set(tmp_replacements))
                else:
                    tmp_partial_model_names.add(tmp_name)
            external_regressor_config[partial_models] = list(tmp_partial_model_names)
            if expected_regressor_names.intersection(tmp_partial_model_names):
                LGR.warning(
                    "External regressors used in more than one partial model: "
                    f"{expected_regressor_names.intersection(tmp_partial_model_names)}"
                )
            expected_regressor_names.update(tmp_partial_model_names)
        if expected_regressor_names - regressor_names:
            err_msg += (
                "Inputed regressors in external_regressors do not include all expected "
                "regressors in partial models or task\n"
                "Expected regressors not in inputted regressors: "
                f"{expected_regressor_names - regressor_names}\n"
                f"Inputted Regressors: {regressor_names}"
            )
        if regressor_names - expected_regressor_names:
            LGR.warning(
                "Regressor labels in external_regressors are not all included in F "
                "statistic partial models or task models. Regressor not in a partial "
                "model or task_keep, it will be included in the full F statistic model "
                "and treated like nuisance regressors to remove. "
                "Regressors not included in any partial model: "
                f"{regressor_names - expected_regressor_names}"
            )

    if len(external_regressors.index) != n_vols:
        err_msg += (
            f"External Regressors have {len(external_regressors.index)} timepoints\n"
            f"while fMRI data have {n_vols} timepoints"
        )

    if err_msg:
        raise RegressError(err_msg)

    return external_regressor_config


def fit_regressors(
    comptable: pd.DataFrame,
    external_regressors: pd.DataFrame,
    external_regressor_config: Dict,
    mixing: npt.NDArray,
) -> pd.DataFrame:
    """Fit regressors to the mixing matrix.

    Uses correlation or F statistics in a linear model depending on the calc_stats
    value in external_regressor_config

    Parameters
    ----------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component,
        with a column for each metric. The index is the component number.
    external_regressors : :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config : :obj:`dict`
        Information describing the external regressors and
        method to use for fitting and statistical tests
    mixing : (T x C) array_like
        Mixing matrix for converting input data to component space,
        where `C` is components and `T` is the same as in `data_cat`

    Returns
    -------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table.
        Same as inputted, with added columns for metrics related to the external regressor fits
    """
    n_vols = mixing.shape[0]

    # If the order of detrending regressors is specified, then pass to
    # create_legendre_polynomial_basis_set
    # otherwise the function sets an order for the Legendre polynomials
    if external_regressor_config["detrend"] is True:
        legendre_arr = utils.create_legendre_polynomial_basis_set(n_vols, dtrank=None)
        LGR.info(
            "External regressors fit includes detrending with "
            f"{legendre_arr.shape[1]} Legendre Polynomial regressors."
        )

    elif (
        isinstance(external_regressor_config["detrend"], int)
        and external_regressor_config["detrend"] > 0
    ):
        legendre_arr = utils.create_legendre_polynomial_basis_set(
            n_vols, dtrank=external_regressor_config["detrend"]
        )
        LGR.info(
            "External regressors fit includes detrending with "
            f"{legendre_arr.shape[1]} Legendre Polynomial regressors."
        )
    else:
        LGR.warning(
            "External regressor fitted without detrending fMRI time series. Only removing mean"
        )
        legendre_arr = utils.create_legendre_polynomial_basis_set(n_vols, dtrank=1)

    detrend_labels = []
    for label_idx in range(legendre_arr.shape[1]):
        detrend_labels.append(f"baseline {label_idx}")
    detrend_regressors = pd.DataFrame(data=legendre_arr, columns=detrend_labels)

    if external_regressor_config["calc_stats"].lower() == "f":
        comptable = fit_mixing_to_regressors(
            comptable, external_regressors, external_regressor_config, mixing, detrend_regressors
        )
    else:
        # This should already be validated by this point, but keeping the catch clause here
        # since this would otherwise just return comptable with no changes, which would
        # make a hard-to-track error
        raise ValueError(
            "calc_stats for external regressors in decision tree is "
            f"{external_regressor_config['calc_stats'].lower()}, which is not valid."
        )

    return comptable


def fit_mixing_to_regressors(
    comptable: pd.DataFrame,
    external_regressors: pd.DataFrame,
    external_regressor_config: Dict,
    mixing: npt.NDArray,
    detrend_regressors: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Linear Model and calculate F statistics and P values for combinations of regressors.

    Equation: Y = XB + E
    - Y = each ICA component (mixing)
    - X = Design (Regressor) matrix (subsets of noise_regress_table)
    - B = Weighting Factors (solving for B)
    - E = errors (Y - Y_pred OR Y - XB)

    Parameters
    ----------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table.
        One row for each component, with a column for each metric.
        The index is the component number.
    external_regressors : :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config : :obj:`dict`
        Information describing the external regressors and
        method to use for fitting and statistical tests
    mixing : (T x C) array_like
        Mixing matrix for converting input data to component space,
        where `C` is components and `T` is the same as in `data_cat`
    detrend_regressors: (n_vols x polort) :obj:`pandas.DataFrame`
        Dataframe containing the detrending regressor time series

    Returns
    -------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table.
        Same as inputted, with added columns for metrics related to external regressor fits.
        New columns for F, R2, and p values for the full model and all partial models.
        Names are "Fstat Full Model", "pval Full Model", "R2stat Full Model",
        and "Full" is replaced by the partial model name for each partial model
    """
    LGR.info("Running fit_mixing_to_regressors")
    LGR.info(f"ICA matrix has {mixing.shape[0]} time points and {mixing.shape[1]} components")

    # regressor_models is a dictionary of all the models that will be fit to the mixing matrix
    #  It will always include 'base' which is just the polort detrending regressors and
    #  'full' which is all relevant regressors including the detrending regressors
    #  For F statistics, the other models need for tests are those that include everything
    #  EXCEPT the category of interest. For example, there will also be a field for "no Motion"
    #  which contains all regressors in the full model except those that model motion
    regressor_models = build_fstat_regressor_models(
        external_regressors, external_regressor_config, detrend_regressors
    )

    # This is the test for the fit of the full model vs the polort detrending baseline
    # The outputs will be what we use to decide which components to reject
    betas_full, f_vals_tmp, p_vals_tmp, r2_vals_tmp = fit_model_with_stats(
        y=mixing, regressor_models=regressor_models, base_label="base"
    )

    # TODO beta_full_model are the fits to all external regressors and might be useful to save
    # TODO Also consider saving regressor_models or the detrending regressors
    # betas_full_model = pd.DataFrame(
    #     data=betas_full.T,
    #     columns=np.concatenate(
    #         (np.array(detrend_regressors.columns), np.array(exte rnal_regressors.columns))
    #     ),
    # )
    f_vals = pd.DataFrame(data=f_vals_tmp, columns=["Fstat Full Model"])
    p_vals = pd.DataFrame(data=p_vals_tmp, columns=["pval Full Model"])
    r2_vals = pd.DataFrame(data=r2_vals_tmp, columns=["R2stat Full Model"])

    # Run a separate model if there are task regressors that might want to be kept
    if "task keep" in regressor_models.keys():
        betas_task_keep, f_vals_tmp, p_vals_tmp, r2_vals_tmp = fit_model_with_stats(
            y=mixing, regressor_models=regressor_models, base_label="base", full_label="task keep"
        )
        f_vals["Fstat Task Model"] = f_vals_tmp
        p_vals["pval Task Model"] = p_vals_tmp
        r2_vals["R2stat Task Model"] = r2_vals_tmp

    # Test the fits between the full model and the full model excluding one category of regressor
    if "f_stats_partial_models" in external_regressor_config.keys():
        for pmodel in external_regressor_config["f_stats_partial_models"]:
            _, f_vals_tmp, p_vals_tmp, r2_vals_tmp = fit_model_with_stats(
                mixing, regressor_models, f"no {pmodel}"
            )
            f_vals[f"Fstat {pmodel} Model"] = f_vals_tmp
            p_vals[f"pval {pmodel} Model"] = p_vals_tmp
            r2_vals[f"R2stat {pmodel} Model"] = r2_vals_tmp

    # Add all F p and R2 statistics to comptable
    comptable = pd.concat((comptable, f_vals, p_vals, r2_vals), axis=1)

    return comptable


def build_fstat_regressor_models(
    external_regressors: pd.DataFrame,
    external_regressor_config: Dict,
    detrend_regressors: pd.DataFrame,
) -> Dict:
    """Combine detrending all or subsets of external regressors to make models to fit and test.

    Parameters
    ----------
    external_regressors : :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config : :obj:`dict`
        Information describing the external regressors and
        method to use for fitting and statistical tests
    detrend_regressors: (n_vols x polort) :obj:`pandas.DataFrame`
        Dataframe containing the detrending regressor time series

    Returns
    -------
    regressor_models: :obj:`dict`
        Each element in the dictionary is a numpy array defining the regressors in a
        regressor model.
        The models that are always included are 'base' which is just the detrending regressors,
        and 'full' which is all user-provided external regressors and the detrending regressors.
        If partial models are named in external_regressor_config["f_stats_partial_models"],
        then each of those will have a dictionary element named "no" then model name and the
        regressors included will be everything except the specified regressors.
        That is "no motion" will include all regressors except the motion regressors.
        This is for the F test which compares the variance explained with the full model to the
        variance explained if the regressors-of-interest for the partial model are removed.
    """
    # The category titles to group each regressor
    if "f_stats_partial_models" in external_regressor_config:
        partial_models = external_regressor_config["f_stats_partial_models"]
    else:
        partial_models = []

    # All regressor labels from the data frame
    regressor_labels = external_regressors.columns
    detrend_regressors_arr = detrend_regressors.to_numpy()
    regressor_models = {"base": detrend_regressors_arr}
    LGR.info(f"Size for base regressor model: {regressor_models['base'].shape}")

    if "task_keep" in external_regressor_config:
        task_keep_model = external_regressor_config["task_keep"]
        # If there is a task_keep model, then the full model should exclude every
        # regressor in task_keep
        tmp_model_labels = {"full": set(regressor_labels) - set(task_keep_model)}
        tmp_model_labels["task keep"] = set(task_keep_model)
        for model_name in ["full", "task keep"]:
            regressor_models[model_name] = detrend_regressors_arr
            for keep_label in tmp_model_labels[model_name]:
                regressor_models[model_name] = np.concatenate(
                    (
                        regressor_models[model_name],
                        np.atleast_2d(
                            stats.zscore(external_regressors[keep_label].to_numpy(), axis=0)
                        ).T,
                    ),
                    axis=1,
                )
            tmp_model_labels[model_name].update(set(detrend_regressors.columns))
            LGR.info(
                f"Size for {model_name} regressor model: {regressor_models[model_name].shape}"
            )
            LGR.info(f"Regressors in {model_name} model: {sorted(tmp_model_labels[model_name])}")
        # Remove task_keep regressors from regressor_labels before calculating partial models
        regressor_labels = set(regressor_labels) - set(task_keep_model)
    else:
        regressor_models["full"] = np.concatenate(
            (detrend_regressors_arr, stats.zscore(external_regressors.to_numpy(), axis=0)), axis=1
        )
        LGR.info(f"Size for full regressor model: {regressor_models['full'].shape}")
        LGR.info(
            "Regressors in full model: "
            f"{sorted(set(regressor_labels).union(set(detrend_regressors.columns)))}"
        )

    for pmodel in partial_models:
        # For F statistics, the other models to test are those that include everything EXCEPT
        # the category of interest
        # That is "no motion" should contain the full model excluding motion regressors
        keep_labels = set(regressor_labels) - set(external_regressor_config[pmodel])
        no_pmodel = f"no {pmodel}"
        regressor_models[no_pmodel] = detrend_regressors_arr
        for keep_label in keep_labels:
            regressor_models[no_pmodel] = np.concatenate(
                (
                    regressor_models[no_pmodel],
                    np.atleast_2d(
                        stats.zscore(external_regressors[keep_label].to_numpy(), axis=0)
                    ).T,
                ),
                axis=1,
            )
        keep_labels.update(set(detrend_regressors.columns))
        LGR.info(
            f"Size for external regressor partial model '{no_pmodel}': "
            f"{regressor_models[no_pmodel].shape}"
        )
        LGR.info(
            "Regressors in partial model (everything but regressors of interest) "
            f"'{no_pmodel}': {sorted(keep_labels)}"
        )

    return regressor_models


def fit_model_with_stats(
    y: npt.NDArray, regressor_models: Dict, base_label: str, full_label: str = "full"
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Fit full and partial models and calculate F stats, R2, and p values.

    Math from page 11-14 of https://afni.nimh.nih.gov/pub/dist/doc/manual/3dDeconvolve.pdf
    Calculates Y=betas*X + error for the base and the full model
    F = ((SSE_base-SSE_full)/(DF_base-DF_full)) / (SSE_full/DF_full)
    DF = degrees of freedom
    SSE = sum of squares error

    Parameters
    ----------
    y : (T X C) :obj:`numpy.ndarray`
        Time by mixing matrix components for the time series for fitting
    regressor_models: :obj:`dict`
        Each element in the dictionary is a numpy array defining the regressors in a
        regressor model.
        Inclues 'full', 'base' and partial models.
    base_label : :obj:`str`
        The base model to compare the full model against.
        For F stat for the full model, this should be 'base'.
        For partial models, this should be the name of the partial model (i.e. "no motion").
    full_label : :obj:`str`
        The full model to use.
        Default="full".
        "full" is expected if the goal is to compare all nuissance regressors to a base model.
        "task_keep" for the special case of fitting pre-defined task-locked regressors.

    Returns
    -------
    betas_full : (C x R) :obj:`numpy.ndarray`
        The beta fits for the full model (components x regressors)
    f_vals : (C x M) :obj:`numpy.ndarray`
        The F statistics for the fits to the full and partial models
    p_vals : (C x M) :obj:`numpy.ndarray`
        The p values for the fits to the full and partial models
    r2_vals : (C x M) :obj:`numpy.ndarray`
        The R2 statistics for the fits to the full and partial models
    """
    betas_base, sse_base, df_base = fit_model(regressor_models[base_label], y)
    betas_full, sse_full, df_full = fit_model(regressor_models[full_label], y)

    # larger sample variance / smaller sample variance (F = (SSE1 – SSE2 / m) / SSE2 / n-k,
    # where SSE = residual sum of squares, m = number of restrictions and k = number of
    # independent variables) -> the 'm' restrictions in this case is the DOF range between
    # the base - full model, the 'n-k' is the number of DOF (independent variables/timepoints)
    # from the fully-fitted model
    f_vals = np.divide((sse_base - sse_full) / (df_base - df_full), (sse_full / df_full))
    # cumulative distribution (FX(x) = P(X<=x), X = real-valued variable, x=probable variable
    # within distribution) of the F-values + extra parameters to shape the range of the
    # distribution values (range: start = DOF_base (unfitted) -
    #       DOF_Full (fitted with full model), end = DOF_Full)
    p_vals = 1 - stats.f.cdf(f_vals, df_base - df_full, df_full)
    # estimate proportion of variance (R2-squared fit) by SSE full [fully fitted]
    # (sum of 1st errors) / SSE base [non-fitted] (sum of 2nd errors) ... and subtracting
    # the error ratio from 1: R² = SSR (fitted model)/SST(total or model base) =
    # Σ (Y_pred-Y_actual)**2 / Σ (Y_pred-Y_actual)**2
    r2_vals = 1 - np.divide(sse_full, sse_base)
    print(y.shape)

    return betas_full, f_vals, p_vals, r2_vals
