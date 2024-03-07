"""Metrics unrelated to TE-(in)dependence."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from tedana.stats import fit_model

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

DEFAULT_REGRESSOR_DICTS = ["Mot12_CSF", "Fmodel", "corr_detrend", "corr"]


class RegressError(Exception):
    """Passes errors that are raised when `validate_extern_regress` fails."""

    pass


def load_validate_external_regressors(external_regressors, external_regressor_config, n_time):
    """
    Load and validate external regressors and descriptors in dictionary.

    Parameters
    ----------
    external_regressors: :obj:`str`
        Path and name of tsv file that includes external regressor time series
    external_regressor_config: :obj:`dict`
        A validated dictionary with info for fitting external regressors
        to component time series
    n_time: :obj:`int`
        Number of timepoints in the fMRI time series

    Returns
    -------
    external_regressors: :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    """
    try:
        external_regressors = pd.read_table(external_regressors)
    except FileNotFoundError:
        raise ValueError(f"Cannot load tsv file with external regressors: {external_regressors}")

    validate_extern_regress(external_regressors, external_regressor_config, n_time)

    return external_regressors


def validate_extern_regress(external_regressors, external_regressor_config, n_time):
    """
    Confirm that provided external regressor dictionary is valid and matches data.

    Checks if expected keys are in external_regressor_config.
    Checks if any regressor labels in the dictionary are specified in the
    user-defined external_regressors
    Checks if the number of time points in the external regressors matches
    the number of time point in the fMRI data

    Parameters
    ----------
    external_regressors : :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config : :obj:`dict`
        Information describing the external regressors and
        method to use for fitting and statistical tests
    n_time : :obj:`int`
        The number of time point in the fMRI time series

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
    if "f_stats_partial_models" in set(external_regressor_config.keys()):
        regressor_names = set(external_regressors.columns)
        expected_regressor_names = set()
        for partial_models in external_regressor_config["f_stats_partial_models"]:
            tmp_names = set(external_regressor_config[partial_models])
            if expected_regressor_names - tmp_names:
                LGR.warning(
                    "External regressors used in more than one partial model: "
                    f"{expected_regressor_names - tmp_names}"
                )
            expected_regressor_names.update(tmp_names)
        if expected_regressor_names - regressor_names:
            err_msg += (
                "Inputed regressors in external_regressors do not include all expected "
                "regressors in partial models\n"
                "Expected regressors not in inputted regressors: "
                f"{expected_regressor_names - regressor_names}\n"
                f"Inputted Regressors: {regressor_names}"
            )
        if regressor_names - expected_regressor_names:
            LGR.warning(
                "Regressor labels in external_regressors are not all included in F "
                "statistic partial models. Whether or not a regressor is in a partial "
                "model, it will be included in the full F statistic model"
                "Regressors not incldued in any partial model: "
                f"{regressor_names - expected_regressor_names}"
            )

        if len(external_regressors.index) != n_time:
            err_msg += (
                f"External Regressors have len(external_regressors.index) timepoints\n"
                f"while fMRI data have {n_time} timepoints"
            )

        if err_msg:
            raise RegressError(err_msg)


def fit_regressors(comptable, external_regressors, external_regressor_config, mixing):
    """
    Fit regressors to the mixing matrix.

    Uses correlation or F statistics in a linear model depending on the calc_stats
    value in external_regressor_config

    Parameters
    ----------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index is the component number.
    external_regressors : :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config : :obj:`dict`
        Information describing the external regressors and
        method to use for fitting and statistical tests
    mixing : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data_cat`

    Returns
    -------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. Same as inputted, with additional columns
        for metrics related to fitting the external regressors
    """
    n_time = mixing.shape[0]

    # If the order of polynomial detrending is specified, then pass to make_detrend_regressors
    # otherwise the function sets a detrending polynomial order
    if external_regressor_config["detrend"] is True:
        detrend_regressors = make_detrend_regressors(n_time, polort=None)
    elif (
        isinstance(external_regressor_config["detrend"], int)
        and external_regressor_config["detrend"] > 0
    ):
        detrend_regressors = make_detrend_regressors(
            n_time, polort=external_regressor_config["detrend"]
        )
    else:
        LGR.warning("External regressor fitting applied without detrending fMRI time series")

    if external_regressor_config["calc_stats"].lower() == "f":
        # external_regressors = pd.concat([external_regressors, detrend_regressors])
        comptable = fit_mixing_to_regressors(
            comptable, external_regressors, external_regressor_config, mixing, detrend_regressors
        )
    else:
        # This should already be valided by this point, but keeping the catch clause here
        # since this would otherwise just return comptable with no changes, which would
        # make a hard-to-track error
        raise ValueError(
            "calc_stats for external regressors in decision tree is "
            f"{external_regressor_config['calc_stats'].lower()}, which is not valid."
        )

    return comptable


def make_detrend_regressors(n_time, polort=None):
    """
    Create polynomial detrending regressors to use for removing slow drifts from data.

    Parameters
    ----------
    n_time : :obj:`int`
        The number of time point in the fMRI time series
    polort : :obj:`int` or :obj:`NoneType`
        The number of polynomial regressors to create (i.e. 3 is x^0, x^1, x^2)
        If None, then this is set to 1+floor(n_time/150)

    Returns
    -------
    detrend_regressors: (n_time x polort) :obj:`pandas.DataFrame`
        Dataframe containing the detrending regressor time series
        x^0 = 1. All other regressors are zscored so that they have
        a mean of 0 and a stdev of 1.
        Dataframe column labels are polort0 - polort{polort-1}
    """
    if polort is None:
        polort = int(1 + np.floor(n_time / 150))

    # create polynomial detrending regressors -> each additive term leads
    # to more points of transformation [curves]
    detrend_regressors = np.zeros((n_time, polort))
    # create polynomial detrended to the power of 0 [1's],
    # **1 [linear trend -> f(x) = a + bx],
    # **2 [quadratic trend -> f(x) = a + bx + cx²],
    # **3 [cubic trend -> f(x) = f(x) = a + bx + cx² + dx³],
    # **4 [quintic trend -> f(x) = a + bx + cx² + dx³ + ex⁴]
    for idx in range(polort):
        # create a linear space with numbers in range [-1,1] because the mean = 0,
        # and include the number of timepoints for each regressor
        tmp = np.linspace(-1, 1, num=n_time) ** idx
        if idx == 0:
            detrend_regressors[:, idx] = tmp
            detrend_labels = ["polort0"]
        else:
            # detrend the regressors by z-scoring the data (zero-mean-centered & stdev of 1)
            detrend_regressors[:, idx] = stats.zscore(tmp)
            # concatenate the polynomial power-detrended regressors within a matrix
            detrend_labels.append(f"polort{idx}")

    # Vestigial code that was used to test whether outputs looked correct.
    # if show_plot:
    #     plt.plot(detrend_regressors)  # display the polynomial power-detrended regressors
    #     plt.show()
    detrend_regressors = pd.DataFrame(data=detrend_regressors, columns=detrend_labels)

    return detrend_regressors


def fit_mixing_to_regressors(
    comptable,
    external_regressors,
    external_regressor_config,
    mixing,
    detrend_regressors,
):
    """
    Compute Linear Model and calculate F statistics and P values for combinations of regressors.

    Equation: Y = XB + E
    - Y = each ICA component (mixing)
    - X = Design (Regressor) matrix (subsets of noise_regress_table)
    - B = Weighting Factors (solving for B)
    - E = errors (Y - Y_pred OR Y - XB)

    Parameters
    ----------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index is the component number.
    external_regressors : :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config : :obj:`dict`
        Information describing the external regressors and
        method to use for fitting and statistical tests
    mixing : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data_cat`
    detrend_regressors: (n_time x polort) :obj:`pandas.DataFrame`
        Dataframe containing the detrending regressor time series

    Returns
    -------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. Same as inputted, with additional columns
        for metrics related to fitting the external regressors.
        There are new columns for F, R2, and p values for the full model
        and all partial models. Names are "Fstat Full Model", "pval Full Model",
        "R2stat Full Model" and "Full" is replaced by the partial model name
        for each partial model
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
    # TODO Need to edit this function so regressor_models can use something besides the full model
    if "task keep" in regressor_models.keys():
        betas_task_keep, f_vals_tmp, p_vals_tmp, r2_vals_tmp = fit_model_with_stats(
            y=mixing, regressor_models=regressor_models, base_label="base", full_label="task keep"
        )

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
    external_regressors, external_regressor_config, detrend_regressors
):
    """
    Combine detrending all or subsets of external regressors to make models to fit and test.

    Parameters
    ----------
    external_regressors : :obj:`pandas.DataFrame`
        Each column is a labelled regressor and the number of rows should
        match the number of timepoints in the fMRI time series
    external_regressor_config : :obj:`dict`
        Information describing the external regressors and
        method to use for fitting and statistical tests
    detrend_regressors: (n_time x polort) :obj:`pandas.DataFrame`
        Dataframe containing the detrending regressor time series

    Returns
    -------
    regressor_models: :obj:`dict`
        Each element in the dictionary is a numpy array defining the regressors in a
        regressor model. The models that are always included are 'base' which is just the
        detrending regressors, and 'full' which is all user-provided external regressors and
        the detrending regressors. If there are partial models that are named in
        external_regressor_config["f_stats_partial_models"] then each of those will have a
        dictionary element named "no" then model name and the regressors included will be
        everything except the specified regressors. That is "no motion" will include all
        regressors except the motion regressors. This is for the F test which compares
        the variance explained with the full model to the variance explained if the
        regressors-of-interest for the partial model are removed.
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
    LGR.info(f"Size for base Regressor Model: {regressor_models['base'].shape}")

    if "task_keep_model" in external_regressor_config:
        task_keep_model = external_regressor_config["task_keep_model"]
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
            LGR.info(f"Size for {model_name} Model: {regressor_models[model_name].shape}")
            LGR.info(
                f"Regressors In {model_name} Model:'{model_name}': {tmp_model_labels[model_name]}"
            )
    else:
        regressor_models["full"] = np.concatenate(
            (detrend_regressors_arr, stats.zscore(external_regressors.to_numpy(), axis=0)), axis=1
        )
        LGR.info(f"Size for full Regressor Model: {regressor_models['full'].shape}")

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
            f"Size for External Regressor Partial Model '{no_pmodel}': "
            f"{regressor_models[no_pmodel].shape}"
        )
        LGR.info(
            "Regressors In Partial Model (everything but regressors of interest) "
            f"'{no_pmodel}': {keep_labels}"
        )

    # vestigial codethat was used to check outputs and might be worth reviving
    # if show_plot:
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(3, 2, 1)
    #     ax.plot(detrend_regressors)
    #     plt.title("detrend")
    #     for idx, reg_cat in enumerate(regress_categories):
    #         if idx < 5:
    #             ax = fig.add_subplot(3, 2, idx + 2)
    #             ax.plot(stats.zscore(categorized_regressors[reg_cat].to_numpy(), axis=0))
    #             plt.title(reg_cat)
    #     plt.savefig(
    #         f"{prefix}_ModelRegressors.jpeg", pil_kwargs={"quality": 20}, dpi="figure"
    #     )  # could also be saves as .eps

    return regressor_models


def fit_model_with_stats(y, regressor_models, base_label, full_label="full"):
    """
    Fit full and partial models and calculate F stats, R2, and p values.

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
        regressor model. Inclues 'full', 'base' and partial models.
    base_label : :obj:`str`
        The base model to compare the full model against. For F stat for the full
        model, this should be 'base'. For partial models, this should be the name
        of the partial model (i.e. "no motion").
    full_label : :obj:`str`
        The full model to use. Default="full" but can also be "task_keep" if
        tasks were inputted and want to compare results against task.

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

    # Vestigial code for testing that might be worth reviving
    # Plots the fits for the first 20 components
    # if show_plot:
    #     plt.clf()
    #     fig = plt.figure(figsize=(20, 24))
    #     for idx in range(30):
    #         # print('Outer bound index: ', idx)

    #         if idx < Y.shape[1]:  # num of components
    #             # print('Actual axis index: ', idx)
    #             ax = fig.add_subplot(5, 6, idx + 1)  # this axis index starts from 1
    #             plot_fit(
    #                 ax,
    #                 Y[:, idx],
    #                 betas_full[:, idx],
    #                 regressor_models["full"],
    #                 betas_base=betas_base[:, idx],
    #                 X_base=regressor_models[base_label],
    #                 F_val=F_vals[idx],
    #                 p_val=p_vals[idx],
    #                 R2_val=R2_vals[idx],
    #                 SSE_base=SSE_base[idx],
    #                 SSE_full=SSE_full[idx],
    #                 base_legend=base_label,
    #             )
    #     base_save_label = base_label.replace(" ", "_")
    #     plt.savefig(
    #         f"{prefix}_ModelFits_{base_save_label}.jpeg",
    #         pil_kwargs={"quality": 20},
    #         dpi="figure",
    #     )  # could also be saved as eps

    return betas_full, f_vals, p_vals, r2_vals


# Vestigial code that was used for testing accuracy of some variables
# Might be worth reviving
# def plot_fit(
#     ax,
#     Y,
#     betas_full,
#     X_full,
#     betas_base=None,
#     X_base=None,
#     F_val=None,
#     p_val=None,
#     R2_val=None,
#     SSE_base=None,
#     SSE_full=None,
#     base_legend="base fit",
# ):
#     """
#     plot_fit: Plot the component time series and the fits to the full and base models

#     INPUTS:
#     ax: axis handle for the figure subplot
#     Y: The ICA component time series to fit to
#     betas_full: The full model fitting parameters
#     X_full: The time series for the full model

#     Optional:
#     betas_base, X_base=None: Model parameters and time series for base model
#     (not plotted if absent)
#     F_val, p_val, R2_val, SSE_base, SSE_full: Fit statistics to include with each plot
#     base_legend: A description of what the base model is to include in the legent
#     """

#     ax.plot(Y, color="black")
#     ax.plot(
#         np.matmul(X_full, betas_full.T), color="red"
#     )  # the 'red' plot is the matrix-multiplication product of the time series *
#     if (type(betas_base) != "NoneType") and (type(X_base) != "NoneType"):
#         ax.plot(np.matmul(X_base, betas_base.T), color="green")
#         ax.text(
#             250,
#             2,
#             f"F={np.around(F_val, decimals=4)}\np={np.around(p_val, decimals=4)}
#               \nR2={np.around(R2_val, decimals=4)}\nSSE_base={np.around(SSE_base,
#               decimals=4)}\nSSE_full={np.around(SSE_full, decimals=4)}",
#         )
#         ax.legend(["ICA Component", "Full fit", f"{base_legend} fit"], loc="best")
#     else:
#         ax.text(
#             250,
#             2,
#             f"F={np.around(F_val, decimals=4)}\np={np.around(p_val, decimals=4)}\nR2=
#                   {np.around(R2_val, decimals=4)}\nSSE_full={np.around(SSE_full, decimals=4)}",
#         )
#         ax.legend(["ICA Component", "Full fit"], loc="best")
