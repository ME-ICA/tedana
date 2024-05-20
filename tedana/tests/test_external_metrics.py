"""Tests for tedana.metrics.external."""

import logging
import os.path as op

import pandas as pd
import pytest

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
        "valid": Column labels expected in demo_minimal_external_regressors_motion_task_models
        The labels in the config file are lowercase and this file is capitalized, but it should
        still be valid.
        "no_mot_y_column": The column labeled "Mot_Pitch" is removed.

    Returns
    -------
    external_regressors : :obj:`pandas.DataFrame` External regressor table
    n_vols : :obj:`int` Number of time points (rows) in external_regressors
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
        "valid": Config dictionary stored in demo_minimal_external_regressors_motion_task_models
        "no_task": Removes "task_keep" info from config
        "no_task_partial": Removes "task_keep" and everything with partial F stats
        "csf_in_mot": Adds "CSF" to the list of motion regressor partial models

    Returns
    -------
    external_regressor_config : :obj:`dict` External Regressor Dictionary
    """

    sample_fname = op.join(
        THIS_DIR,
        "../resources/decision_trees",
        "demo_minimal_external_regressors_motion_task_models.json",
    )
    tree = load_json(sample_fname)
    external_regressor_config = tree["external_regressor_config"]

    if config_choice == "no_task":
        external_regressor_config.pop("task_keep")
    elif config_choice == "no_task_partial":
        external_regressor_config.pop("task_keep")
        external_regressor_config.pop("f_stats_partial_models")
        external_regressor_config.pop("Motion")
        external_regressor_config.pop("CSF")
    elif config_choice == "csf_in_mot":
        external_regressor_config["Motion"].append("CSF")
    elif config_choice == "unmatched_regex":
        external_regressor_config["Motion"] = ["^translation_.*$"]
    elif config_choice != "valid":
        raise ValueError(f"config_choice is {config_choice}, which is not a listed option")

    return external_regressor_config


def sample_mixing_matrix():
    """Load and return the three-echo mixing matrix."""

    test_data_path, osf_id = data_for_testing_info("three-echo")
    download_test_data(osf_id, test_data_path)

    return pd.read_csv(
        op.join(
            THIS_DIR,
            "../../.testing_data_cache/three-echo/TED.three-echo/desc_ICA_mixing_static.tsv",
        ),
        delimiter="\t",
    ).to_numpy()


def sample_comptable(n_components):
    """Create an empty component table."""

    row_vals = []
    for ridx in range(n_components):
        row_vals.append(f"ICA_{str(ridx).zfill(2)}")

    return pd.DataFrame(data={"Component": row_vals})


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
    assert set(external_regressor_config_expanded["Motion"]) == set(
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
    assert external_regressor_config_expanded["CSF"] == ["CSF"]
    assert external_regressor_config_expanded["task_keep"] == ["Signal"]
    assert "WARNING" not in caplog.text

    # Rerunning with explicit names for the above three categories instead of regex patterns
    # Shouldn't change anything, but making sure it runs
    caplog.clear()
    external_regressor_config = external.validate_extern_regress(
        external_regressors, external_regressor_config_expanded, n_vols
    )
    assert "WARNING" not in caplog.text

    # Removing all partial model and task_keep stuff to confirm it still runs
    caplog.clear()
    external_regressor_config = sample_external_regressor_config("no_task_partial")
    external.validate_extern_regress(external_regressors, external_regressor_config, n_vols)
    assert caplog.text == ""

    # Removing "task_keep" from config to test if warning appears
    caplog.clear()
    external_regressor_config = sample_external_regressor_config("no_task")
    external.validate_extern_regress(external_regressors, external_regressor_config, n_vols)
    assert "Regressor labels in external_regressors are not all included in F" in caplog.text

    # Add "CSF" to "Motion" partial model (also in "CSF" partial model) to test if warning appears
    caplog.clear()
    external_regressor_config = sample_external_regressor_config("csf_in_mot")
    external.validate_extern_regress(external_regressors, external_regressor_config, n_vols)
    assert "External regressors used in more than one partial model" in caplog.text


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
    with pytest.raises(external.RegressError, match="No external regressor labels matching regex"):
        external.validate_extern_regress(external_regressors, external_regressor_config, n_vols)

    # If a regressor expected in the config is not in external_regressors
    # Run successfully to expand Motion labels in config and then create error
    # when "Mot_Y" is in the config, but removed from external_regressros
    external_regressor_config = sample_external_regressor_config()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )
    external_regressors, n_vols = sample_external_regressors("no_mot_y_column")
    with pytest.raises(
        external.RegressError,
        match="Inputed regressors in external_regressors do not include all expected",
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


def test_fit_regressors_succeeds():
    """Test conditions fit_regressors should succeed."""

    external_regressors, n_vols = sample_external_regressors()
    external_regressor_config = sample_external_regressor_config()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )
    mixing = sample_mixing_matrix()

    comptable = sample_comptable(mixing.shape[1])
    comptable = external.fit_regressors(
        comptable, external_regressors, external_regressor_config_expanded, mixing
    )

    # TODO Add validation of output and tests of conditional statements
