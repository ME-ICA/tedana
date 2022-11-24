
import os

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
    # TODO push changes to my branch
    # TODO script downloading data...
    # Test re-running, but use the CLI
    args = [
        "-d",
        "/tmp/data/three-echo/three_echo_Cornell_zcat.nii.gz", # TODO this should be a list of NIFTIs, see_five_echo
        "-e", # TODO read from.json in the dataset *1000?
        "14.5",
        "38.5",
        "62.5",
        "--out-dir",
        out_dir_manual,
        "--debug",
        "--verbose",
        "--ctab",
        os.path.join(out_dir, "desc-tedana_metrics.tsv"),
        "--mix",
        os.path.join(out_dir, "desc-ICA_mixing.tsv"),
    ]
    tedana_cli._main(args) # FIXME ficheros no coinciden
