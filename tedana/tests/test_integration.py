"""Integration tests for "real" data."""

import glob
import logging
import os
import os.path as op
import re
import shutil
import subprocess

import pandas as pd
import pytest
from pkg_resources import resource_filename

from tedana.io import InputHarvester
from tedana.tests.utils import data_for_testing_info, download_test_data
from tedana.workflows import t2smap as t2smap_cli
from tedana.workflows import tedana as tedana_cli
from tedana.workflows.ica_reclassify import ica_reclassify_workflow

# Need to see if a no BOLD warning occurred
LOGGER = logging.getLogger(__name__)
# Added a testing logger to output whether or not testing data were downlaoded
TestLGR = logging.getLogger("TESTING")


def check_integration_outputs(fname, outpath, n_logs=1):
    """
    Checks outputs of integration tests.

    Parameters
    ----------
    fname : str
        Path to file with expected outputs
    outpath : str
        Path to output directory generated from integration tests
    """

    # Gets filepaths generated by integration test
    found_files = [
        os.path.relpath(f, outpath)
        for f in glob.glob(os.path.join(outpath, "**"), recursive=True)[1:]
    ]

    # Checks for log file
    log_regex = "^tedana_[12][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}[0-9]{2}[0-9]{2}.tsv$"
    logfiles = [out for out in found_files if re.match(log_regex, out)]
    assert len(logfiles) == n_logs

    # Removes logfiles from list of existing files
    for log in logfiles:
        found_files.remove(log)

    # Compares remaining files with those expected
    with open(fname) as f:
        expected_files = f.read().splitlines()
    expected_files = [os.path.normpath(path) for path in expected_files]

    if sorted(found_files) != sorted(expected_files):
        expected_not_found = sorted(list(set(expected_files) - set(found_files)))
        found_not_expected = sorted(list(set(found_files) - set(expected_files)))

        msg = ""
        if expected_not_found:
            msg += "\nExpected but not found:\n\t"
            msg += "\n\t".join(expected_not_found)

        if found_not_expected:
            msg += "\nFound but not expected:\n\t"
            msg += "\n\t".join(found_not_expected)
        raise ValueError(msg)


def reclassify_raw() -> str:
    test_data_path, _ = data_for_testing_info("three-echo-reclassify")
    return os.path.join(test_data_path, "TED.three-echo")


def reclassify_raw_registry() -> str:
    return os.path.join(reclassify_raw(), "desc-tedana_registry.json")


def guarantee_reclassify_data() -> None:
    """Ensures that the reclassify data exists at the expected path and return path."""

    test_data_path, osf_id = data_for_testing_info("three-echo-reclassify")

    # Should now be checking and not downloading for each test so don't see if statement here
    # if not os.path.exists(reclassify_raw_registry()):
    download_test_data(osf_id, test_data_path)
    # else:
    # Path exists, be sure that everything in registry exists
    ioh = InputHarvester(reclassify_raw_registry())
    all_present = True
    for _, v in ioh.registry.items():
        if not isinstance(v, list):
            if not os.path.exists(os.path.join(reclassify_raw(), v)):
                all_present = False
                break
    if not all_present:
        # Something was removed, need to re-download
        shutil.rmtree(reclassify_raw())
        guarantee_reclassify_data()
    return test_data_path


def test_integration_five_echo(skip_integration):
    """Integration test of the full tedana workflow using five-echo test data."""

    if skip_integration:
        pytest.skip("Skipping five-echo integration test")

    test_data_path, osf_id = data_for_testing_info("five-echo")
    out_dir = os.path.abspath(os.path.join(test_data_path, "../../outputs/five-echo"))
    # out_dir_manual = f"{out_dir}-manual"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # if os.path.exists(out_dir_manual):
    #     shutil.rmtree(out_dir_manual)

    # download data and run the test
    download_test_data(osf_id, test_data_path)
    prepend = f"{test_data_path}/p06.SBJ01_S09_Task11_e"
    suffix = ".sm.nii.gz"
    datalist = [prepend + str(i + 1) + suffix for i in range(5)]
    echo_times = [15.4, 29.7, 44.0, 58.3, 72.6]
    # adding n_independent_echos=4 to test workflow code using n_independent_echos is executed
    tedana_cli.tedana_workflow(
        data=datalist,
        tes=echo_times,
        ica_method="robustica",
        n_robust_runs=4,
        n_independent_echos=4,
        out_dir=out_dir,
        tedpca=0.95,
        fittype="curvefit",
        tree="minimal",
        fixed_seed=49,
        tedort=True,
        verbose=True,
        prefix="sub-01",
    )

    # Just a check on the component table pending a unit test of load_comptable
    component_table = os.path.join(out_dir, "sub-01_desc-tedana_metrics.tsv")
    df = pd.read_table(component_table)
    assert isinstance(df, pd.DataFrame)

    # compare the generated output files
    fn = resource_filename("tedana", "tests/data/nih_five_echo_outputs_verbose.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_four_echo(skip_integration):
    """Integration test of the full tedana workflow using four-echo test data."""

    if skip_integration:
        pytest.skip("Skipping four-echo integration test")

    test_data_path, osf_id = data_for_testing_info("four-echo")
    out_dir = os.path.abspath(os.path.join(test_data_path, "../../outputs/four-echo"))
    out_dir_manual = f"{out_dir}-manual"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if os.path.exists(out_dir_manual):
        shutil.rmtree(out_dir_manual)

    # download data and run the test
    download_test_data(osf_id, test_data_path)
    prepend = f"{test_data_path}/sub-PILOT_ses-01_task-localizerDetection_run-01_echo-"
    suffix = "_space-sbref_desc-preproc_bold+orig.HEAD"
    datalist = [prepend + str(i + 1) + suffix for i in range(4)]
    tedana_cli.tedana_workflow(
        data=datalist,
        mixing_file=op.join(op.dirname(datalist[0]), "desc-ICA_mixing_static.tsv"),
        tes=[11.8, 28.04, 44.28, 60.52],
        ica_method="fastica",
        out_dir=out_dir,
        tedpca="kundu-stabilize",
        gscontrol=["gsr", "mir"],
        png_cmap="bone",
        prefix="sub-01",
        debug=True,
        verbose=True,
    )

    # compare the generated output files
    fn = resource_filename("tedana", "tests/data/fiu_four_echo_outputs.txt")

    check_integration_outputs(fn, out_dir)

    ica_reclassify_workflow(
        op.join(out_dir, "sub-01_desc-tedana_registry.json"),
        accept=[1, 2, 3],
        reject=[4, 5, 6],
        tag_accept=["manual tag 1", "manual tag 2"],
        out_dir=out_dir_manual,
        mir=True,
        verbose=True,
    )

    component_table = pd.read_csv(op.join(out_dir_manual, "desc-tedana_metrics.tsv"), sep="\t")
    assert set(component_table.loc[1]["classification_tags"].split(",")) == {
        "Likely BOLD",
        "manual tag 1",
        "manual tag 2",
    }


def test_integration_three_echo(skip_integration):
    """Integration test of the full tedana workflow using three-echo test data."""

    if skip_integration:
        pytest.skip("Skipping three-echo integration test")

    test_data_path, osf_id = data_for_testing_info("three-echo")
    out_dir = os.path.abspath(os.path.join(test_data_path, "../../outputs/three-echo"))
    out_dir_manual = f"{out_dir}-rerun"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if os.path.exists(out_dir_manual):
        shutil.rmtree(out_dir_manual)

    # download data and run the test
    download_test_data(osf_id, test_data_path)
    tedana_cli.tedana_workflow(
        data=f"{test_data_path}/three_echo_Cornell_zcat.nii.gz",
        tes=[14.5, 38.5, 62.5],
        out_dir=out_dir,
        low_mem=True,
        tedpca="aic",
    )

    # Test re-running, but use the CLI
    # TODO Move this to a separate integration test, use the fixed desc_ICA_mixing_static.tsv that
    #      is distributed with the testing data, and test specific outputs for consistent values
    args = [
        "-d",
        f"{test_data_path}/three_echo_Cornell_zcat.nii.gz",
        "-e",
        "14.5",
        "38.5",
        "62.5",
        "--out-dir",
        out_dir_manual,
        "--debug",
        "--verbose",
        "-f",
        "--mix",
        os.path.join(out_dir, "desc-ICA_mixing.tsv"),
    ]
    tedana_cli._main(args)

    # compare the generated output files
    fn = resource_filename("tedana", "tests/data/cornell_three_echo_outputs.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_three_echo_external_regressors_single_model(skip_integration, caplog):
    """Integration test of tedana workflow with extern regress and F stat."""

    if skip_integration:
        pytest.skip("Skipping three-echo with external regressors integration test")

    test_data_path, osf_id = data_for_testing_info("three-echo")
    out_dir = os.path.abspath(
        os.path.join(test_data_path, "../../outputs/three-echo-externalreg-Ftest")
    )

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # download data and run the test
    # external_regress_Ftest_3echo.tsv has 13 rows. Based on a local run on the 3 echo data:
    #  Col 1 (trans_x_correlation) is the TS for ICA comp 59 + similar stdev Gaussian Noise
    #  Col 2 (trans_y_correlation) is 0.4*comp29+0.5+comp20+Gaussian Noise
    #  Col 3 (trans_z_correlation) is comp20+Gaussian Noise
    #  Col 4-6 are Gaussian noise representing pitch/roll/yaw
    #  Col 7-12 are the first derivative of col 1-6
    # With the currently set up decision tree,
    # Component 59 should be rejected because it is correlated to trans_x and,
    # comp 20 should be rejected because of a signif fit to a combination of trans_y and trans_z.
    # Component 29 is not rejected because the fit does not cross a r>0.8 threshold
    # Note that the above is in comparision to the minimal decision tree
    # but the integration test for 3 echoes uses the kundu tree
    download_test_data(osf_id, test_data_path)
    tree_name = "resources/decision_trees/demo_external_regressors_single_model.json"
    tedana_cli.tedana_workflow(
        data=f"{test_data_path}/three_echo_Cornell_zcat.nii.gz",
        tes=[14.5, 38.5, 62.5],
        out_dir=out_dir,
        tree=resource_filename("tedana", tree_name),
        external_regressors=resource_filename(
            "tedana", "tests/data/external_regress_Ftest_3echo.tsv"
        ),
        low_mem=True,
        tedpca="aic",
    )

    # compare the generated output files
    fn = resource_filename("tedana", "tests/data/cornell_three_echo_outputs.txt")
    check_integration_outputs(fn, out_dir)

    assert "It is strongly recommended to provide an external mask," in caplog.text


def test_integration_three_echo_external_regressors_motion_task_models(skip_integration):
    """Integration test of tedana workflow with extern regress and F stat."""

    if skip_integration:
        pytest.skip("Skipping three-echo with external regressors integration test")

    test_data_path, osf_id = data_for_testing_info("three-echo")
    out_dir = os.path.abspath(
        os.path.join(test_data_path, "../../outputs/three-echo-externalreg-Ftest-multimodels")
    )

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # download data and run the test
    # external_regress_Ftest_3echo.tsv has 12 columns for motion, 1 for CSF, and 1 for task signal
    # The regressor values and expected fits with the data are detailed in:
    # tests.test_external_metrics.sample_external_regressors
    download_test_data(osf_id, test_data_path)
    tree_name = "resources/decision_trees/demo_external_regressors_motion_task_models.json"
    tedana_cli.tedana_workflow(
        data=f"{test_data_path}/three_echo_Cornell_zcat.nii.gz",
        tes=[14.5, 38.5, 62.5],
        out_dir=out_dir,
        tree=resource_filename("tedana", tree_name),
        external_regressors=resource_filename(
            "tedana", "tests/data/external_regress_Ftest_3echo.tsv"
        ),
        mixing_file=f"{test_data_path}/desc_ICA_mixing_static.tsv",
        low_mem=True,
        tedpca="aic",
    )

    # compare the generated output files
    fn = resource_filename("tedana", "tests/data/cornell_three_echo_preset_mixing_outputs.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_insufficient_args(skip_integration):
    if skip_integration:
        pytest.skip("Skipping reclassify insufficient args")

    guarantee_reclassify_data()

    test_data_path, osf_id = data_for_testing_info("three-echo")
    out_dir = os.path.abspath(
        os.path.join(test_data_path, "../../outputs/reclassify/insufficient")
    )

    args = ["ica_reclassify", reclassify_raw_registry(), "--out-dir", out_dir]

    result = subprocess.run(args, capture_output=True)
    assert b"ValueError: Must manually accept or reject" in result.stderr
    assert result.returncode != 0


def test_integration_reclassify_quiet_csv(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify quiet csv")

    test_data_path = guarantee_reclassify_data()
    out_dir = os.path.abspath(os.path.join(test_data_path, "../outputs/reclassify/quiet"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Make some files that have components to manually accept and reject
    to_accept = [i for i in range(3)]
    to_reject = [i for i in range(7, 4)]
    acc_df = pd.DataFrame(data=to_accept, columns=["Components"])
    rej_df = pd.DataFrame(data=to_reject, columns=["Components"])
    acc_csv_fname = os.path.join(reclassify_raw(), "accept.csv")
    rej_csv_fname = os.path.join(reclassify_raw(), "reject.csv")
    acc_df.to_csv(acc_csv_fname)
    rej_df.to_csv(rej_csv_fname)

    # also adding parameters for --tagacc and --tagrej
    args = [
        "ica_reclassify",
        "--manacc",
        acc_csv_fname,
        "--manrej",
        rej_csv_fname,
        "--tagacc",
        "manual accept",
        "--tagrej",
        "manual reject, manual reject2",
        "--out-dir",
        out_dir,
        reclassify_raw_registry(),
    ]

    results = subprocess.run(args, capture_output=True)
    assert results.returncode == 0
    fn = resource_filename("tedana", "tests/data/reclassify_quiet_out.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_quiet_spaces(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify quiet space-delimited integers")

    test_data_path = guarantee_reclassify_data()
    out_dir = os.path.abspath(os.path.join(test_data_path, "../outputs/reclassify/quiet"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    args = [
        "ica_reclassify",
        "--manacc",
        "1",
        "--manrej",
        "4",
        "5",
        "6",
        "--out-dir",
        out_dir,
        reclassify_raw_registry(),
    ]

    results = subprocess.run(args, capture_output=True)
    assert results.returncode == 0
    fn = resource_filename("tedana", "tests/data/reclassify_quiet_out.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_quiet_string(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify quiet string of integers")

    test_data_path = guarantee_reclassify_data()
    out_dir = os.path.abspath(os.path.join(test_data_path, "../outputs/reclassify/quiet"))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    args = [
        "ica_reclassify",
        "--manacc",
        "1,2,3",
        "--manrej",
        "4,5,6",
        "--out-dir",
        out_dir,
        reclassify_raw_registry(),
    ]

    results = subprocess.run(args, capture_output=True)
    assert results.returncode == 0
    fn = resource_filename("tedana", "tests/data/reclassify_quiet_out.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_debug(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify debug")

    test_data_path = guarantee_reclassify_data()
    out_dir = os.path.abspath(os.path.join(test_data_path, "../outputs/reclassify/debug"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    args = [
        "ica_reclassify",
        "--manacc",
        "1",
        "2",
        "3",
        "--prefix",
        "sub-testymctestface",
        "--convention",
        "orig",
        "--tedort",
        "--mir",
        "--no-reports",
        "--out-dir",
        out_dir,
        "--debug",
        reclassify_raw_registry(),
    ]

    results = subprocess.run(args, capture_output=True)
    assert results.returncode == 0
    fn = resource_filename("tedana", "tests/data/reclassify_debug_out.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_both_rej_acc(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify both rejected and accepted")

    test_data_path = guarantee_reclassify_data()
    out_dir = os.path.abspath(os.path.join(test_data_path, "../outputs/reclassify/both_rej_acc"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    with pytest.raises(
        ValueError,
        match=r"The following components were both accepted and",
    ):
        ica_reclassify_workflow(
            reclassify_raw_registry(),
            accept=[1],
            reject=[1, 2, 3],
            out_dir=out_dir,
        )


def test_integration_reclassify_run_twice(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify both rejected and accepted")

    test_data_path = guarantee_reclassify_data()
    out_dir = os.path.abspath(os.path.join(test_data_path, "../outputs/reclassify/run_twice"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Also testing if a manually specified tag is added to classification_tags the first time,
    # and when it is run again with overwrite, two different tags are added the second time.
    ica_reclassify_workflow(
        reclassify_raw_registry(),
        accept=[1, 2, 3],
        tag_accept="manual tag",
        out_dir=out_dir,
        no_reports=True,
    )
    component_table = pd.read_csv(op.join(out_dir, "desc-tedana_metrics.tsv"), sep="\t")
    assert set(component_table.loc[1]["classification_tags"].split(",")) == {
        "Likely BOLD",
        "manual tag",
    }
    ica_reclassify_workflow(
        reclassify_raw_registry(),
        accept=[1, 2, 3],
        tag_accept="manual tag 2, manual tag 3",
        out_dir=out_dir,
        overwrite=True,
        no_reports=True,
    )
    fn = resource_filename("tedana", "tests/data/reclassify_run_twice.txt")
    check_integration_outputs(fn, out_dir, n_logs=2)
    component_table = pd.read_csv(op.join(out_dir, "desc-tedana_metrics.tsv"), sep="\t")
    assert set(component_table.loc[1]["classification_tags"].split(",")) == {
        "Likely BOLD",
        "manual tag 2",
        "manual tag 3",
    }


def test_integration_reclassify_no_bold(skip_integration, caplog):
    if skip_integration:
        pytest.skip("Skip reclassify both rejected and accepted")

    test_data_path = guarantee_reclassify_data()
    out_dir = os.path.abspath(os.path.join(test_data_path, "../outputs/reclassify/no_bold"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    ioh = InputHarvester(reclassify_raw_registry())
    component_table = ioh.get_file_contents("ICA metrics tsv")
    to_accept = [i for i in range(len(component_table))]

    ica_reclassify_workflow(
        reclassify_raw_registry(),
        reject=to_accept,
        out_dir=out_dir,
        no_reports=True,
    )
    assert "No accepted components remaining after manual classification!" in caplog.text

    fn = resource_filename("tedana", "tests/data/reclassify_no_bold.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_accrej_files(skip_integration, caplog):
    if skip_integration:
        pytest.skip("Skip reclassify both rejected and accepted")

    test_data_path = guarantee_reclassify_data()
    out_dir = os.path.abspath(os.path.join(test_data_path, "../outputs/reclassify/no_bold"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    ioh = InputHarvester(reclassify_raw_registry())
    component_table = ioh.get_file_contents("ICA metrics tsv")
    to_accept = [i for i in range(len(component_table))]

    ica_reclassify_workflow(
        reclassify_raw_registry(),
        reject=to_accept,
        out_dir=out_dir,
        no_reports=True,
    )
    assert "No accepted components remaining after manual classification!" in caplog.text

    fn = resource_filename("tedana", "tests/data/reclassify_no_bold.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_t2smap(skip_integration):
    """Integration test of the full t2smap workflow using five-echo test data."""
    if skip_integration:
        pytest.skip("Skipping t2smap integration test")
    test_data_path, osf_id = data_for_testing_info("five-echo")
    out_dir = os.path.abspath(os.path.join(test_data_path, "../../outputs/t2smap_five-echo"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # download data and run the test
    download_test_data(osf_id, test_data_path)
    prepend = f"{test_data_path}/p06.SBJ01_S09_Task11_e"
    suffix = ".sm.nii.gz"
    datalist = [prepend + str(i + 1) + suffix for i in range(5)]
    echo_times = [15.4, 29.7, 44.0, 58.3, 72.6]
    args = (
        ["-d"]
        + datalist
        + ["-e"]
        + [str(te) for te in echo_times]
        + ["--out-dir", out_dir, "--fittype", "curvefit"]
        + ["--masktype", "dropout", "decay"]
        + ["--n-independent-echos", "4"]
    )
    t2smap_cli._main(args)

    # compare the generated output files
    fname = resource_filename("tedana", "tests/data/nih_five_echo_outputs_t2smap.txt")
    # Gets filepaths generated by integration test
    found_files = [
        os.path.relpath(f, out_dir)
        for f in glob.glob(os.path.join(out_dir, "**"), recursive=True)[1:]
    ]

    # Compares remaining files with those expected
    with open(fname) as f:
        expected_files = f.read().splitlines()
    assert sorted(expected_files) == sorted(found_files)
