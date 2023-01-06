"""
Integration tests for "real" data
"""

import glob
import logging
import os
import os.path as op
import re
import shutil
import subprocess
import tarfile
from gzip import GzipFile
from io import BytesIO

import pandas as pd
import pytest
import requests
from pkg_resources import resource_filename

from tedana.io import InputHarvester
from tedana.workflows import t2smap as t2smap_cli
from tedana.workflows import tedana as tedana_cli
from tedana.workflows.tedana_reclassify import post_tedana

# Need to see if a no BOLD warning occurred
LOGGER = logging.getLogger(__name__)


def check_integration_outputs(fname, outpath, n_logs=1):
    """
    Checks outputs of integration tests

    Parameters
    ----------
    fname : str
        Path to file with expected outputs
    outpath : str
        Path to output directory generated from integration tests
    """

    # Gets filepaths generated by integration test
    existing = [
        os.path.relpath(f, outpath)
        for f in glob.glob(os.path.join(outpath, "**"), recursive=True)[1:]
    ]

    # Checks for log file
    log_regex = "^tedana_[12][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}[0-9]{2}[0-9]{2}.tsv$"
    logfiles = [out for out in existing if re.match(log_regex, out)]
    assert len(logfiles) == n_logs

    # Removes logfiles from list of existing files
    for log in logfiles:
        existing.remove(log)

    # Compares remaining files with those expected
    with open(fname, "r") as f:
        tocheck = f.read().splitlines()
    tocheck = [os.path.normpath(path) for path in tocheck]
    assert sorted(tocheck) == sorted(existing)


def download_test_data(osf, outpath):
    """
    Downloads tar.gz data stored at `osf` and unpacks into `outpath`

    Parameters
    ----------
    osf : str
        URL to OSF file that contains data to be downloaded
    outpath : str
        Path to directory where OSF data should be extracted
    """

    req = requests.get(osf)
    req.raise_for_status()
    t = tarfile.open(fileobj=GzipFile(fileobj=BytesIO(req.content)))
    os.makedirs(outpath, exist_ok=True)
    t.extractall(outpath)


def reclassify_path() -> str:
    """Get the path to the reclassify test data."""
    return "/tmp/data/reclassify/"


def reclassify_raw() -> str:
    return os.path.join(reclassify_path(), "TED.three-echo")


def reclassify_raw_registry() -> str:
    return os.path.join(reclassify_raw(), "desc-tedana_registry.json")


def reclassify_url() -> str:
    """Get the URL to reclassify test data."""
    return "https://osf.io/f6g45/download"


def guarantee_reclassify_data() -> None:
    """Ensures that the reclassify data exists at the expected path."""
    if not os.path.exists(reclassify_raw_registry()):
        download_test_data(reclassify_url(), reclassify_path())
    else:
        # Path exists, be sure that everything in registry exists
        ioh = InputHarvester(os.path.join(reclassify_raw(), "desc-tedana_registry.json"))
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


def test_integration_five_echo(skip_integration):
    """Integration test of the full tedana workflow using five-echo test data."""

    if skip_integration:
        pytest.skip("Skipping five-echo integration test")

    out_dir = "/tmp/data/five-echo/TED.five-echo"
    out_dir_manual = "/tmp/data/five-echo/TED.five-echo-manual"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if os.path.exists(out_dir_manual):
        shutil.rmtree(out_dir_manual)

    # download data and run the test
    download_test_data("https://osf.io/9c42e/download", os.path.dirname(out_dir))
    prepend = "/tmp/data/five-echo/p06.SBJ01_S09_Task11_e"
    suffix = ".sm.nii.gz"
    datalist = [prepend + str(i + 1) + suffix for i in range(5)]
    echo_times = [15.4, 29.7, 44.0, 58.3, 72.6]
    tedana_cli.tedana_workflow(
        data=datalist,
        tes=echo_times,
        out_dir=out_dir,
        tedpca=0.95,
        fittype="curvefit",
        fixed_seed=49,
        tedort=True,
        verbose=True,
    )

    # Just a check on the component table pending a unit test of load_comptable
    comptable = os.path.join(out_dir, "desc-tedana_metrics.tsv")
    df = pd.read_table(comptable)
    assert isinstance(df, pd.DataFrame)

    # compare the generated output files
    fn = resource_filename("tedana", "tests/data/nih_five_echo_outputs_verbose.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_four_echo(skip_integration):
    """Integration test of the full tedana workflow using four-echo test data"""

    if skip_integration:
        pytest.skip("Skipping four-echo integration test")

    out_dir = "/tmp/data/four-echo/TED.four-echo"
    out_dir_manual = "/tmp/data/four-echo/TED.four-echo-manual"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if os.path.exists(out_dir_manual):
        shutil.rmtree(out_dir_manual)

    # download data and run the test
    download_test_data("https://osf.io/gnj73/download", os.path.dirname(out_dir))
    prepend = "/tmp/data/four-echo/"
    prepend += "sub-PILOT_ses-01_task-localizerDetection_run-01_echo-"
    suffix = "_space-sbref_desc-preproc_bold+orig.HEAD"
    datalist = [prepend + str(i + 1) + suffix for i in range(4)]
    tedana_cli.tedana_workflow(
        data=datalist,
        tes=[11.8, 28.04, 44.28, 60.52],
        out_dir=out_dir,
        tedpca="kundu-stabilize",
        gscontrol=["gsr", "mir"],
        png_cmap="bone",
        debug=True,
        verbose=True,
    )

    post_tedana(
        op.join(out_dir, "desc-tedana_registry.json"),
        accept=[1, 2, 3],
        reject=[4, 5, 6],
        out_dir=out_dir_manual,
        mir=True,
    )

    # compare the generated output files
    fn = resource_filename("tedana", "tests/data/fiu_four_echo_outputs.txt")

    check_integration_outputs(fn, out_dir)


def test_integration_three_echo(skip_integration):
    """Integration test of the full tedana workflow using three-echo test data"""

    if skip_integration:
        pytest.skip("Skipping three-echo integration test")

    out_dir = "/tmp/data/three-echo/TED.three-echo"
    out_dir_manual = "/tmp/data/three-echo/TED.three-echo-rerun"

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if os.path.exists(out_dir_manual):
        shutil.rmtree(out_dir_manual)

    # download data and run the test
    download_test_data("https://osf.io/rqhfc/download", os.path.dirname(out_dir))
    tedana_cli.tedana_workflow(
        data="/tmp/data/three-echo/three_echo_Cornell_zcat.nii.gz",
        tes=[14.5, 38.5, 62.5],
        out_dir=out_dir,
        low_mem=True,
        tedpca="aic",
    )

    # Test re-running, but use the CLI
    args = [
        "-d",
        "/tmp/data/three-echo/three_echo_Cornell_zcat.nii.gz",
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


def test_integration_reclassify_insufficient_args(skip_integration):
    if skip_integration:
        pytest.skip("Skipping reclassify insufficient args")

    guarantee_reclassify_data()

    args = [
        "tedana_reclassify",
        os.path.join(reclassify_raw(), "desc-tedana_registry.json"),
    ]

    result = subprocess.run(args, capture_output=True)
    assert b"ValueError: Must manually accept or reject" in result.stderr
    assert result.returncode != 0


def test_integration_reclassify_quiet_csv(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify quiet csv")

    guarantee_reclassify_data()
    out_dir = os.path.join(reclassify_path(), "quiet")
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

    args = [
        "tedana_reclassify",
        "--manacc",
        acc_csv_fname,
        "--manrej",
        rej_csv_fname,
        "--out-dir",
        out_dir,
        os.path.join(reclassify_raw(), "desc-tedana_registry.json"),
    ]

    results = subprocess.run(args, capture_output=True)
    assert results.returncode == 0
    fn = resource_filename("tedana", "tests/data/reclassify_quiet_out.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_quiet_spaces(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify quiet space-delimited integers")

    guarantee_reclassify_data()
    out_dir = os.path.join(reclassify_path(), "quiet")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    args = [
        "tedana_reclassify",
        "--manacc",
        "1",
        "2",
        "3",
        "--manrej",
        "4",
        "5",
        "6",
        "--out-dir",
        out_dir,
        os.path.join(reclassify_raw(), "desc-tedana_registry.json"),
    ]

    results = subprocess.run(args, capture_output=True)
    assert results.returncode == 0
    fn = resource_filename("tedana", "tests/data/reclassify_quiet_out.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_quiet_string(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify quiet string of integers")

    guarantee_reclassify_data()
    out_dir = os.path.join(reclassify_path(), "quiet")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    args = [
        "tedana_reclassify",
        "--manacc",
        "1,2,3",
        "--manrej",
        "4,5,6,",
        "--out-dir",
        out_dir,
        os.path.join(reclassify_raw(), "desc-tedana_registry.json"),
    ]

    results = subprocess.run(args, capture_output=True)
    assert results.returncode == 0
    fn = resource_filename("tedana", "tests/data/reclassify_quiet_out.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_debug(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify debug")

    guarantee_reclassify_data()
    out_dir = os.path.join(reclassify_path(), "debug")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    args = [
        "tedana_reclassify",
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
        os.path.join(reclassify_raw(), "desc-tedana_registry.json"),
    ]

    results = subprocess.run(args, capture_output=True)
    assert results.returncode == 0
    fn = resource_filename("tedana", "tests/data/reclassify_debug_out.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_reclassify_both_rej_acc(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify both rejected and accepted")

    guarantee_reclassify_data()
    out_dir = os.path.join(reclassify_path(), "both_rej_acc")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    with pytest.raises(
        ValueError,
        match=r"The following components were both accepted and",
    ):
        post_tedana(
            reclassify_raw_registry(),
            accept=[1, 2, 3],
            reject=[1, 2, 3],
            out_dir=out_dir,
        )


def test_integration_reclassify_run_twice(skip_integration):
    if skip_integration:
        pytest.skip("Skip reclassify both rejected and accepted")

    guarantee_reclassify_data()
    out_dir = os.path.join(reclassify_path(), "run_twice")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    post_tedana(
        reclassify_raw_registry(),
        accept=[1, 2, 3],
        out_dir=out_dir,
        no_reports=True,
    )
    post_tedana(
        reclassify_raw_registry(),
        accept=[1, 2, 3],
        out_dir=out_dir,
        force=True,
        no_reports=True,
    )
    fn = resource_filename("tedana", "tests/data/reclassify_run_twice.txt")
    check_integration_outputs(fn, out_dir, n_logs=2)


def test_integration_reclassify_no_bold(skip_integration, caplog):
    if skip_integration:
        pytest.skip("Skip reclassify both rejected and accepted")

    guarantee_reclassify_data()
    out_dir = os.path.join(reclassify_path(), "no_bold")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    ioh = InputHarvester(reclassify_raw_registry())
    comptable = ioh.get_file_contents("ICA metrics tsv")
    to_accept = [i for i in range(len(comptable))]

    post_tedana(
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

    guarantee_reclassify_data()
    out_dir = os.path.join(reclassify_path(), "no_bold")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    ioh = InputHarvester(reclassify_raw_registry())
    comptable = ioh.get_file_contents("ICA metrics tsv")
    to_accept = [i for i in range(len(comptable))]

    post_tedana(
        reclassify_raw_registry(),
        reject=to_accept,
        out_dir=out_dir,
        no_reports=True,
    )
    assert "No accepted components remaining after manual classification!" in caplog.text

    fn = resource_filename("tedana", "tests/data/reclassify_no_bold.txt")
    check_integration_outputs(fn, out_dir)


def test_integration_t2smap(skip_integration):
    """Integration test of the full t2smap workflow using five-echo test data"""
    if skip_integration:
        pytest.skip("Skipping t2smap integration test")
    out_dir = "/tmp/data/five-echo/t2smap_five-echo"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # download data and run the test
    download_test_data("https://osf.io/9c42e/download", os.path.dirname(out_dir))
    prepend = "/tmp/data/five-echo/p06.SBJ01_S09_Task11_e"
    suffix = ".sm.nii.gz"
    datalist = [prepend + str(i + 1) + suffix for i in range(5)]
    echo_times = [15.4, 29.7, 44.0, 58.3, 72.6]
    args = (
        ["-d"]
        + datalist
        + ["-e"]
        + [str(te) for te in echo_times]
        + ["--out-dir", out_dir, "--fittype", "curvefit"]
    )
    t2smap_cli._main(args)

    # compare the generated output files
    fname = resource_filename("tedana", "tests/data/nih_five_echo_outputs_t2smap.txt")
    # Gets filepaths generated by integration test
    existing = [
        os.path.relpath(f, out_dir)
        for f in glob.glob(os.path.join(out_dir, "**"), recursive=True)[1:]
    ]

    # Compares remaining files with those expected
    with open(fname, "r") as f:
        tocheck = f.read().splitlines()
    assert sorted(tocheck) == sorted(existing)
