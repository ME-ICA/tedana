"""Tests for tedana.metrics.external."""

import logging
import os.path as op
import re

import pandas as pd
import pytest

from tedana import utils
from tedana.io import load_json
from tedana.metrics import external
from tedana.tests.utils import data_for_testing_info, download_test_data

THIS_DIR = op.dirname(op.abspath(__file__))
LGR = logging.getLogger("GENERAL")

# ----------------------------------------------------------------------
# Functions Used For Tests
# ----------------------------------------------------------------------


def sample_external_regressors(regress_choice="valid"):
    """
    Retrieves a sample external regressor dataframe.

    Parameters
    ----------
    regress_choice : :obj:`str` How to keep or alter the external regressor data
        Options are:
        "valid": Column labels expected in demo_external_regressors_motion_task_models
        The labels in the config file are lowercase and this file is capitalized, but it should
        still be valid.
        "no_mot_y_column": The column labeled "Mot_Pitch" is removed.

    Returns
    -------
    external_regressors : :obj:`pandas.DataFrame` External regressor table
    n_vols : :obj:`int` Number of time points (rows) in external_regressors

    Notes
    -----
    The loaded external regressors are in ./tests/data/external_regress_Ftest_3echo.tsv
    These are based on tedana being run with default parameters on the 3 echo data using
    the mixing matrix downloaded with the three echo data
    .testing_data_cache/three-echo/TED.three-echo/desc_ICA_mixing_static.tsv
    For the external regressors:
    Column 0 (Mot_X) is the time series for ICA component 8 + Gaussian noise
    Column 1 (Mot_Y) is 0.6 * comp 18 + 0.4 * comp 29 + Gaussian Noise
    Column 2 (Mot_Z) is 0.8 * comp 18 + 1.2 * Gaussian Noise
    Column 3 (Mot_Pitch) is 0.9 * comp 30 + 0.1 * comp 61 + Gaussian Noise
    Columns 4-5 are Gaussian noise
    Columns 6-11 are the first derivatives of columns 0-5
    Column 12 (CSF) is comp 11 + Gaussian Noise
    Column 13 (Signal) is comp 30 + Gaussian Noise

    The base gaussian noise is mean=0, stdev=1.
    The scaling weights for components and noise are set so that, with an R^2>0.5 threshold:
    ICA Comp 8 rejected solely based on the fit to Mot_X
    ICA Comp 31 looks strongly inversely correlated to Comp 8 and is also rejected
    ICA Comp 18 rejected based on the combined fit to Mot_X and Mot_Y (signif Motion partial model)
    ICA Comp 29 NOT rejected based only on the fit to Mot_Y
    ICA Comp 11 rejected based on a fit to CSF (signif CSF partial model)
    ICA Comp 30 accepted based on fit to task model even through also fits to Mot_Pitch
    """
    sample_fname = op.join(THIS_DIR, "data", "external_regress_Ftest_3echo.tsv")

    external_regressors = pd.read_csv(sample_fname, delimiter="\t")

    if regress_choice == "no_mot_y_column":
        external_regressors = external_regressors.drop(columns="Mot_Y")
    elif regress_choice != "valid":
        raise ValueError(f"regress_choice is {regress_choice}, which is not a listed option")

    n_vols = len(external_regressors)

    return external_regressors, n_vols


def sample_external_regressor_config(config_choice="valid"):
    """
    Retrieves a sample external regressor configuration dictionary.

    Parameters
    ----------
    config_choice : :obj:`str` How to keep or alter the config file
        Options are:
        "valid": Config dictionary stored in demo_external_regressors_motion_task_models
        "no_task_partial": Removes "task_keep" and everything with partial F stats
        "csf_in_mot": Adds "CSF" to the list of motion regressor partial models
        "signal_in_mot": Adds "Signal" to the list of motion regressor partial models

    Returns
    -------
    external_regressor_config : :obj:`dict` External Regressor Dictionary
    """

    sample_fname = op.join(
        THIS_DIR,
        "../resources/decision_trees",
        "demo_external_regressors_motion_task_models.json",
    )
    tree = load_json(sample_fname)
    external_regressor_config = tree["external_regressor_config"]

    if config_choice == "no_task_partial":
        external_regressor_config = [external_regressor_config[0]]
        external_regressor_config[0].pop("partial_models")
    elif config_choice == "csf_in_mot":
        external_regressor_config[0]["partial_models"]["Motion"].append("^csf.*$")
    elif config_choice == "signal_in_mot":
        external_regressor_config[0]["partial_models"]["Motion"].append("Signal")
    elif config_choice == "unmatched_regex":
        external_regressor_config[0]["partial_models"]["Motion"] = ["^translation_.*$"]
    elif config_choice != "valid":
        raise ValueError(f"config_choice is {config_choice}, which is not a listed option")

    return external_regressor_config


def sample_mixing_matrix():
    """Load and return the three-echo mixing matrix."""

    test_data_path, osf_id = data_for_testing_info("three-echo")
    download_test_data(osf_id, test_data_path)

    return pd.read_csv(
        op.join(
            data_for_testing_info("path"),
            "three-echo/TED.three-echo/desc_ICA_mixing_static.tsv",
        ),
        delimiter="\t",
    ).to_numpy()


def sample_comptable(n_components):
    """Create an empty component table.

    Parameters
    ----------
    n_components : :obj:`int`
        The number of components (rows) in the compponent table DataFrame

    Returns
    -------
    component_table : :obj:`pd.DataFrame`
        A component table with a single "Component" column with
        "ICA_" number for each row
    """

    row_vals = []
    for ridx in range(n_components):
        row_vals.append(f"ICA_{str(ridx).zfill(2)}")

    return pd.DataFrame(data={"Component": row_vals})


def sample_detrend_regressors(n_vols, dtrank=None):
    """
    Creates Legendre polynomial detrending regressors.

    Parameters
    ----------
    n_vols: :obj:`int`
        The number of volumes or time points for the regressors
    dtrank : :obj:`int` or None
        The rank (number) of detrending regressors to create
        Automatically calculate if None (default)

    Returns
    -------
    detrend_regressors : :obj:`pd.DataFrame` The specified detrending regressors
    """

    legendre_arr = utils.create_legendre_polynomial_basis_set(n_vols, dtrank)
    detrend_labels = []
    for label_idx in range(legendre_arr.shape[1]):
        detrend_labels.append(f"baseline {label_idx}")
    return pd.DataFrame(data=legendre_arr, columns=detrend_labels)


# validate_extern_regress
# -----------------------
def test_validate_extern_regress_succeeds(caplog):
    """Test validate_extern_regress works as expected."""

    external_regressors, n_vols = sample_external_regressors()
    external_regressor_config = sample_external_regressor_config()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )

    # The regex patterns should have been replaced with the full names of the regressors
    assert set(external_regressor_config_expanded[0]["partial_models"]["Motion"]) == set(
        [
            "Mot_X",
            "Mot_d1_Yaw",
            "Mot_d1_Y",
            "Mot_d1_Pitch",
            "Mot_Z",
            "Mot_d1_Z",
            "Mot_d1_Roll",
            "Mot_d1_X",
            "Mot_Pitch",
            "Mot_Y",
            "Mot_Yaw",
            "Mot_Roll",
        ]
    )
    assert external_regressor_config_expanded[0]["partial_models"]["CSF"] == ["CSF"]
    assert external_regressor_config_expanded[1]["regressors"] == ["Signal"]
    assert "WARNING" not in caplog.text

    # Rerunning with explicit names for the above three categories instead of regex patterns
    # Shouldn't change anything, but making sure it runs
    caplog.clear()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config_expanded, n_vols
    )
    assert "WARNING" not in caplog.text

    # Removing all partial model and task_keep stuff to confirm it still runs, but with a warning
    caplog.clear()
    external_regressor_config = sample_external_regressor_config("no_task_partial")
    external.validate_extern_regress(external_regressors, external_regressor_config, n_vols)
    assert (
        "User-provided external_regressors include columns not used "
        "in any external regressor model: ['Signal']"
    ) in caplog.text

    # Add "CSF" to "Motion" partial model (also in "CSF" partial model) to test if warning appears
    caplog.clear()
    external_regressor_config = sample_external_regressor_config("csf_in_mot")
    external.validate_extern_regress(external_regressors, external_regressor_config, n_vols)
    assert "['CSF'] used in more than one partial regressor model for nuisance" in caplog.text


def test_validate_extern_regress_fails():
    """Test validate_extern_regress fails when expected."""

    external_regressors, n_vols = sample_external_regressors()
    external_regressor_config = sample_external_regressor_config()

    # If there are a different number of time points in the fMRI data and external regressors
    with pytest.raises(
        external.RegressError, match=f"while fMRI data have {n_vols - 1} timepoints"
    ):
        external.validate_extern_regress(
            external_regressors, external_regressor_config, n_vols - 1
        )

    # If no external regressor labels match the regex label in config
    external_regressor_config = sample_external_regressor_config("unmatched_regex")
    with pytest.raises(
        external.RegressError,
        match=(
            re.escape(
                "No external regressor labels matching regular expression '^translation_.*$' found"
            )
        ),
    ):
        external.validate_extern_regress(external_regressors, external_regressor_config, n_vols)

    # If Signal is in a partial model, but not "regressors" for the full model
    external_regressor_config = sample_external_regressor_config("signal_in_mot")
    with pytest.raises(
        external.RegressError,
        match=(
            re.escape(
                "Partial models in nuisance include regressors that "
                "are excluded from its full model: ['Signal']"
            )
        ),
    ):
        external.validate_extern_regress(external_regressors, external_regressor_config, n_vols)

    # If a regressor expected in the config is not in external_regressors
    # Run successfully to expand Motion labels in config and then create error
    # when "Mot_Y" is in the config, but removed from external_regressors
    external_regressor_config = sample_external_regressor_config()
    external_regressors, n_vols = sample_external_regressors()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )
    external_regressors, n_vols = sample_external_regressors("no_mot_y_column")
    # The same error message will appear twice.
    # One for "regressor" and once for motion partial model
    with pytest.raises(
        external.RegressError,
        match=re.escape(
            "No external regressor matching 'Mot_Y' was found.\n"
            "No external regressor matching 'Mot_Y' was found."
        ),
    ):
        external.validate_extern_regress(
            external_regressors, external_regressor_config_expanded, n_vols
        )


# load_validate_external_regressors
# ---------------------------------


def test_load_validate_external_regressors_fails():
    """Test load_validate_external_regressors fails when not given a  tsv file."""

    external_regressors = "NotATSVFile.tsv"
    external_regressor_config = sample_external_regressor_config("valid")
    with pytest.raises(
        ValueError, match=f"Cannot load tsv file with external regressors: {external_regressors}"
    ):
        external.load_validate_external_regressors(
            external_regressors, external_regressor_config, 200
        )


def test_load_validate_external_regressors_smoke():
    """Test load_validate_external_regressors succeeds."""

    external_regressors = op.join(THIS_DIR, "data", "external_regress_Ftest_3echo.tsv")
    n_vols = 75
    external_regressor_config = sample_external_regressor_config()

    # Not testing outputs because this is just calling validate_extern_regress and
    # outputs are checked in those tests
    external.load_validate_external_regressors(
        external_regressors, external_regressor_config, n_vols
    )


# fit_regressors
# --------------


def test_fit_regressors(caplog):
    """Test conditions fit_regressors succeeds and fails."""

    caplog.set_level(logging.INFO)
    external_regressors, n_vols = sample_external_regressors()
    external_regressor_config = sample_external_regressor_config()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )
    mixing = sample_mixing_matrix()

    # Running with external_regressor_config["detrend"] is True,
    #  which results in 1 detrending regressor
    component_table = sample_comptable(mixing.shape[1])
    component_table = external.fit_regressors(
        component_table, external_regressors, external_regressor_config_expanded, mixing
    )

    # Contents will be valided in fit_mixing_to_regressors so just checking column labels here
    assert set(component_table.keys()) == {
        "Component",
        "Fstat nuisance model",
        "Fstat task model",
        "Fstat nuisance Motion partial model",
        "Fstat nuisance CSF partial model",
        "pval nuisance model",
        "pval task model",
        "pval nuisance Motion partial model",
        "pval nuisance CSF partial model",
        "R2stat nuisance model",
        "R2stat task model",
        "R2stat nuisance Motion partial model",
        "R2stat nuisance CSF partial model",
    }

    assert (
        "External regressors fit for nuisance includes detrending "
        "with 1 Legendre Polynomial regressors" in caplog.text
    )

    caplog.clear()
    # Running with external_regressor_config["detrend"]=3, which results in 3 detrending regressors
    external_regressor_config[1]["detrend"] = 3
    component_table = sample_comptable(mixing.shape[1])
    component_table = external.fit_regressors(
        component_table, external_regressors, external_regressor_config_expanded, mixing
    )
    assert (
        "External regressors fit for task includes detrending "
        "with 3 Legendre Polynomial regressors" in caplog.text
    )

    caplog.clear()
    # Running with external_regressor_config["detrend"]=0,
    #  which results in 1 detrend regressors (demeaning)
    external_regressor_config[0]["detrend"] = 0
    component_table = sample_comptable(mixing.shape[1])
    component_table = external.fit_regressors(
        component_table, external_regressors, external_regressor_config_expanded, mixing
    )
    assert (
        "External regressor for nuisance fitted without detrending fMRI time series. "
        "Only removing mean" in caplog.text
    )

    caplog.clear()
    external_regressor_config[1]["statistic"] = "Corr"
    component_table = sample_comptable(mixing.shape[1])
    with pytest.raises(
        ValueError,
        match=(
            "statistic for task external regressors in decision tree is corr, "
            "which is not valid."
        ),
    ):
        component_table = external.fit_regressors(
            component_table, external_regressors, external_regressor_config_expanded, mixing
        )


# fit_mixing_to_regressors
# --------------


def test_fit_mixing_to_regressors(caplog):
    """Test conditions fit_mixing_to_regressors succeeds and fails."""

    # Note: Outputs from fit_model_with_stats are also tested within this function

    caplog.set_level(logging.INFO)
    external_regressors, n_vols = sample_external_regressors()
    external_regressor_config = sample_external_regressor_config()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )
    mixing = sample_mixing_matrix()

    detrend_regressors = sample_detrend_regressors(n_vols, dtrank=None)

    # Running with external_regressor_config["detrend"] is True,
    #  which results in 1 detrending regressor
    component_table = sample_comptable(mixing.shape[1])

    for config_idx in range(2):
        component_table = external.fit_mixing_to_regressors(
            component_table,
            external_regressors,
            external_regressor_config_expanded[config_idx],
            mixing,
            detrend_regressors,
        )

    # Since a fixed mixing matrix is used, the values should always be consistent
    # Comparing just 3 rows and rounding to 6 decimal places to avoid testing failures
    # due to differences in floating point precision between systems
    output_rows_to_validate = component_table.iloc[[0, 11, 30]].round(decimals=6)
    expected_results = pd.DataFrame(
        columns=[
            "Component",
            "Fstat nuisance model",
            "Fstat task model",
            "Fstat nuisance Motion partial model",
            "Fstat nuisance CSF partial model",
            "pval nuisance model",
            "pval task model",
            "pval nuisance Motion partial model",
            "pval nuisance CSF partial model",
            "R2stat nuisance model",
            "R2stat task model",
            "R2stat nuisance Motion partial model",
            "R2stat nuisance CSF partial model",
        ]
    )
    expected_results.loc[0] = [
        "ICA_00",
        0.5898043794795538,
        0.040242260292383224,
        0.6359651437299336,
        0.5882298391006501,
        0.8529159565446598,
        0.8415655022508225,
        0.8033066601119929,
        0.4460627598486151,
        0.1116607090996441,
        0.0005509601152330346,
        0.11119635498661495,
        0.009551010649882286,
    ]
    expected_results.loc[11] = [
        "ICA_11",
        5.050391950932562,
        0.3101483992796387,
        0.6191428219572478,
        37.021610927761515,
        5.897391126885587e-06,
        0.5792925973677331,
        0.8177727274388485,
        8.422777264538439e-08,
        0.518377055217612,
        0.004230633903377634,
        0.10857438156630017,
        0.377688252390028,
    ]
    expected_results.loc[30] = [
        "ICA_30",
        5.869398664558788,
        109.32951177196031,
        6.215675922255525,
        1.524970189426933,
        7.193855290354989e-07,
        3.3306690738754696e-16,
        5.303071232143353e-07,
        0.22160450819684074,
        0.5557244697248107,
        0.5996259777665551,
        0.5501080476751579,
        0.024389778752502478,
    ]

    assert (
        (output_rows_to_validate.sort_index(axis=1))
        .compare(expected_results.sort_index(axis=1).round(decimals=6))
        .empty
    )


# build_fstat_regressor_models
# --------------


def test_build_fstat_regressor_models(caplog):
    """Test conditions build_fstat_regressor_models succeeds and fails."""

    caplog.set_level(logging.INFO)
    external_regressors, n_vols = sample_external_regressors()
    external_regressor_config = sample_external_regressor_config()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )

    detrend_regressors = sample_detrend_regressors(n_vols, dtrank=3)

    # Running nuisance with partial_models
    regressor_models = external.build_fstat_regressor_models(
        external_regressors, external_regressor_config_expanded[0], detrend_regressors
    )

    assert regressor_models["full"].shape == (n_vols, 16)
    assert (
        "Regressors in full model for nuisance: "
        "['CSF', 'Mot_Pitch', 'Mot_Roll', 'Mot_X', 'Mot_Y', 'Mot_Yaw', "
        "'Mot_Z', 'Mot_d1_Pitch', 'Mot_d1_Roll', 'Mot_d1_X', 'Mot_d1_Y', 'Mot_d1_Yaw', "
        "'Mot_d1_Z', 'baseline 0', 'baseline 1', 'baseline 2']"
    ) in caplog.text
    assert regressor_models["no CSF"].shape == (n_vols, 15)
    assert (
        "Regressors in partial model (everything but regressors of interest) 'no CSF': "
        "['Mot_Pitch', 'Mot_Roll', 'Mot_X', 'Mot_Y', 'Mot_Yaw', 'Mot_Z', 'Mot_d1_Pitch', "
        "'Mot_d1_Roll', 'Mot_d1_X', 'Mot_d1_Y', 'Mot_d1_Yaw', 'Mot_d1_Z', "
        "'baseline 0', 'baseline 1', 'baseline 2']"
    ) in caplog.text
    assert regressor_models["no Motion"].shape == (n_vols, 4)
    assert (
        "Regressors in partial model (everything but regressors of interest) 'no Motion': "
        "['CSF', 'baseline 0', 'baseline 1', 'baseline 2']" in caplog.text
    )

    # Running task regressor
    caplog.clear()
    regressor_models = external.build_fstat_regressor_models(
        external_regressors, external_regressor_config_expanded[1], detrend_regressors
    )

    assert regressor_models["full"].shape == (n_vols, 4)
    assert (
        "Regressors in full model for task: ['Signal', 'baseline 0', 'baseline 1', 'baseline 2']"
        in caplog.text
    )
