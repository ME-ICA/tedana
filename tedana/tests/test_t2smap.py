"""Tests for t2smap."""

import json
import os.path as op
from shutil import rmtree

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from tedana import workflows
from tedana.tests.utils import get_test_data_path
from tedana.workflows import t2smap as t2smap_cli
from tedana.workflows import tedana as tedana_cli


def test_backend_cli_option():
    """Both workflow CLIs expose the shared backend selector."""
    for parser in (t2smap_cli._get_parser(), tedana_cli._get_parser()):
        backend = parser._option_string_actions["--backend"]
        assert backend.default == "cpu"
        assert backend.choices == ["cpu", "cuda", "auto"]


def _write_single_volume_echoes(data_dir, tmp_path):
    """Write the first volume of each test echo as a 3D NIfTI image."""
    data = []
    for echo in range(1, 4):
        img = nb.load(op.join(data_dir, f"echo{echo}.nii.gz"))
        echo_data = np.asanyarray(img.dataobj)[..., 0]
        echo_path = tmp_path / f"echo{echo}_single.nii.gz"
        nb.Nifti1Image(echo_data, img.affine).to_filename(echo_path)
        data.append(str(echo_path))
    return data


class TestT2smap:
    def test_basic_t2smap1(self):
        """
        A very simple test, to confirm that t2smap creates output.

        files.
        """
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data, [14.5, 38.5, 62.5], combmode="t2s", fitmode="all", out_dir=out_dir
        )

        # Check outputs
        assert op.isfile(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def test_basic_t2smap2(self):
        """
        A very simple test, to confirm that t2smap creates output.

        files when fitmode is set to ts.
        """
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data, [14.5, 38.5, 62.5], combmode="t2s", fitmode="ts", out_dir=out_dir
        )

        # Check outputs
        assert op.isfile(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def test_basic_t2smap3(self):
        """
        A very simple test, to confirm that t2smap creates output.

        files when combmode is set to 'paid'.
        """
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data, [14.5, 38.5, 62.5], combmode="paid", fitmode="all", out_dir=out_dir
        )

        # Check outputs
        assert op.isfile(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def test_basic_t2smap4(self):
        """
        A very simple test, to confirm that t2smap creates output.

        files when combmode is set to 'paid' and fitmode is set to 'ts'.
        """
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        workflows.t2smap_workflow(
            data, [14.5, 38.5, 62.5], combmode="paid", fitmode="ts", out_dir=out_dir
        )

        # Check outputs
        assert op.isfile(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def test_single_volume_t2smap(self, tmp_path):
        """A simple test to confirm that t2smap supports 3D echo inputs."""
        data_dir = get_test_data_path()
        data = _write_single_volume_echoes(data_dir, tmp_path)
        mask = op.join(data_dir, "mask.nii.gz")
        out_dir = tmp_path / "TED.echo1.single-volume.t2smap"
        workflows.t2smap_workflow(
            data,
            [14.5, 38.5, 62.5],
            combmode="t2s",
            fitmode="all",
            mask=mask,
            out_dir=str(out_dir),
            verbose=True,
        )

        target_shape = nb.load(data[0]).shape
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert img.shape == target_shape
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert img.shape == target_shape
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert img.shape == target_shape
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert img.shape == target_shape
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert img.shape == target_shape + (1,)

    def test_single_volume_t2smap_paid_fails(self, tmp_path):
        """PAID combination needs temporal variance, so one-volume inputs should fail clearly."""
        data_dir = get_test_data_path()
        data = _write_single_volume_echoes(data_dir, tmp_path)
        mask = op.join(data_dir, "mask.nii.gz")
        out_dir = tmp_path / "TED.echo1.single-volume-paid.t2smap"

        with pytest.raises(ValueError, match="PAID combination requires more than one volume"):
            workflows.t2smap_workflow(
                data,
                [14.5, 38.5, 62.5],
                combmode="paid",
                fitmode="all",
                mask=mask,
                out_dir=str(out_dir),
            )

    def test_t2smap_cli(self):
        """Run test_basic_t2smap1, but use the CLI method."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        args = (
            ["-d"]
            + data
            + [
                "-e",
                "14.5",
                "38.5",
                "62.5",
                "--dummy-scans",
                "1",
                "--exclude",
                "0:2",  # exclude one volume beyond the dummy scan
                "--combmode",
                "t2s",
                "--fitmode",
                "all",
                "--out-dir",
                out_dir,
            ]
        )
        workflows.t2smap._main(args)

        # Check outputs
        img = nb.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4
        in_img = nb.load(data[0])
        target_shape = list(in_img.shape)
        target_shape[3] = target_shape[3] - 1  # account for dummy scans, but not exclude; #1401
        output_shape = list(img.shape)
        assert output_shape == target_shape

    def test_failing_t2smap_01(self):
        """A simple failing configuration for t2smap."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        with pytest.raises(ValueError, match="Excluding volumes is not supported for fitmode"):
            workflows.t2smap_workflow(
                data,
                [14.5, 38.5, 62.5],
                combmode="t2s",
                fitmode="ts",
                out_dir=out_dir,
                exclude="0,1,2,3",
            )

    def test_failing_t2smap_02(self):
        """A simple failing configuration for t2smap."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = "TED.echo1.t2smap"
        with pytest.raises(ValueError, match="out of bounds"):
            workflows.t2smap_workflow(
                data,
                [14.5, 38.5, 62.5],
                combmode="t2s",
                fitmode="all",
                out_dir=out_dir,
                exclude="1000",
            )

    def test_interpolate_failing_voxels_curvefit(self, tmp_path):
        """Smoke test: t2smap_workflow with interpolate_failing_voxels=True completes."""
        data_dir = get_test_data_path()
        data = [
            op.join(data_dir, "echo1.nii.gz"),
            op.join(data_dir, "echo2.nii.gz"),
            op.join(data_dir, "echo3.nii.gz"),
        ]
        out_dir = str(tmp_path / "output")
        workflows.t2smap_workflow(
            data,
            [14.5, 38.5, 62.5],
            combmode="t2s",
            fitmode="all",
            fittype="curvefit",
            interpolate_failing_voxels=True,
            out_dir=out_dir,
        )
        assert op.isfile(op.join(out_dir, "T2starmap.nii.gz"))
        assert op.isfile(op.join(out_dir, "S0map.nii.gz"))

        # The failures map encodes 0 (never failed), 1 (failed the first
        # curvefit pass), and 2 (failed both passes).
        failures_file = op.join(out_dir, "desc-decayFitFailures_mask.nii.gz")
        assert op.isfile(failures_file)
        failures_data = nb.load(failures_file).get_fdata()
        assert set(np.unique(failures_data)).issubset({0, 1, 2})
        # This dataset has a single first-pass failure that converges on the
        # second pass, so the map should contain a 1 but no 2.
        assert (failures_data == 1).any()
        assert not (failures_data == 2).any()

    def test_t2smap_exclude_last_index(self, tmp_path):
        """The last input volume can be excluded; one past it is out of bounds."""
        data_dir = get_test_data_path()
        data = [op.join(data_dir, f"echo{echo}.nii.gz") for echo in range(1, 4)]
        n_input_vols = nb.load(data[0]).shape[3]

        out_dir = str(tmp_path / "last")
        workflows.t2smap_workflow(
            data,
            [14.5, 38.5, 62.5],
            combmode="t2s",
            fitmode="all",
            out_dir=out_dir,
            dummy_scans=1,
            exclude=str(n_input_vols - 1),
        )
        table = pd.read_table(op.join(out_dir, "desc-censoring_timeseries.tsv"))
        assert table.shape[0] == n_input_vols - 1
        assert table["sample_mask"].iloc[-1] == 0
        assert table["sample_mask"].iloc[:-1].all()

        with pytest.raises(ValueError, match=r"between 0 and"):
            workflows.t2smap_workflow(
                data,
                [14.5, 38.5, 62.5],
                combmode="t2s",
                fitmode="all",
                out_dir=str(tmp_path / "past_end"),
                exclude=str(n_input_vols),
            )

    def test_t2smap_censoring_table(self, tmp_path):
        """Censoring writes a table and sidecar; no censoring writes neither."""
        data_dir = get_test_data_path()
        data = [op.join(data_dir, f"echo{echo}.nii.gz") for echo in range(1, 4)]
        in_img = nb.load(data[0])
        n_input_vols = in_img.shape[3]
        tr = float(in_img.header.get_zooms()[-1])

        out_dir = str(tmp_path / "censored")
        workflows.t2smap_workflow(
            data,
            [14.5, 38.5, 62.5],
            combmode="t2s",
            fitmode="all",
            out_dir=out_dir,
            dummy_scans=1,
            exclude="0,2:4",  # input volume 0 is a dummy scan and is ignored
        )

        table = pd.read_table(op.join(out_dir, "desc-censoring_timeseries.tsv"))
        n_vols = n_input_vols - 1
        np.testing.assert_array_equal(table["volume_index"], np.arange(n_vols))
        np.testing.assert_array_equal(table["original_volume_index"], np.arange(1, n_input_vols))
        np.testing.assert_allclose(table["time"], np.arange(n_vols) * tr)
        expected_mask = np.ones(n_vols, dtype=int)
        expected_mask[[1, 2]] = 0
        np.testing.assert_array_equal(table["sample_mask"], expected_mask)

        with open(op.join(out_dir, "desc-censoring_timeseries.json")) as fobj:
            metadata = json.load(fobj)
        assert metadata["DummyScans"] == 1
        assert metadata["NumberOfCensoredVolumes"] == 2
        assert metadata["LongestCensoredRun"] == 2

        with open(op.join(out_dir, "desc-tedana_registry.json")) as fobj:
            registry = json.load(fobj)
        assert registry["censoring tsv"] == "desc-censoring_timeseries.tsv"
        assert registry["censoring json"] == "desc-censoring_timeseries.json"

        # The optimally combined data keep every post-dummy volume
        assert nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz")).shape[3] == n_vols

        # Excluding only dummy scans, or nothing, writes no censoring files
        for i_run, exclude in enumerate((None, "0")):
            out_dir = str(tmp_path / f"uncensored_{i_run}")
            workflows.t2smap_workflow(
                data,
                [14.5, 38.5, 62.5],
                combmode="t2s",
                fitmode="all",
                out_dir=out_dir,
                dummy_scans=1,
                exclude=exclude,
            )
            assert not op.isfile(op.join(out_dir, "desc-censoring_timeseries.tsv"))
            assert not op.isfile(op.join(out_dir, "desc-censoring_timeseries.json"))

    @pytest.mark.parametrize("combmode", ["t2s", "paid"])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_t2smap_censored_spike_invariance(self, tmp_path, use_mask, combmode):
        """Values in censored volumes cannot affect any estimate.

        This covers the masks, the T2*/S0 maps, the RMSE map and decay metrics, and the
        retained volumes of the optimally combined data (including PAID weights).
        Without a user mask, this also checks that the initial EPI mask is computed
        from the retained volumes only.
        """
        data_dir = get_test_data_path()
        data = [op.join(data_dir, f"echo{echo}.nii.gz") for echo in range(1, 4)]
        mask = op.join(data_dir, "mask.nii.gz") if use_mask else None
        dummy_scans = 1
        # The test data have 5 volumes: 1 dummy scan, 2 censored, 2 retained
        exclude = "2,4"
        censored_input_vols = [2, 4]

        # Write float32 copies of the data, with and without huge spikes in the censored
        # volumes. Both copies go through the same writer, so the retained volumes are
        # bit-identical (an integer header would rescale the whole image around the spikes).
        in_files = {"orig": [], "spiked": []}
        for echo, fname in enumerate(data, start=1):
            img = nb.load(fname)
            header = img.header.copy()
            header.set_data_dtype(np.float32)
            arr = img.get_fdata(dtype=np.float32)
            spiked_arr = arr.copy()
            spiked_arr[..., censored_input_vols] = spiked_arr[..., censored_input_vols] * 50 + 1e5
            for label, out_arr in (("orig", arr), ("spiked", spiked_arr)):
                out_fname = str(tmp_path / f"{label}_echo{echo}.nii.gz")
                nb.Nifti1Image(out_arr, img.affine, header).to_filename(out_fname)
                in_files[label].append(out_fname)

        out_dirs = {}
        for label in ("orig", "spiked"):
            out_dirs[label] = str(tmp_path / label)
            workflows.t2smap_workflow(
                in_files[label],
                [14.5, 38.5, 62.5],
                combmode=combmode,
                fitmode="all",
                mask=mask,
                out_dir=out_dirs[label],
                dummy_scans=dummy_scans,
                exclude=exclude,
            )

        for fname in (
            "desc-adaptiveGoodSignal_mask.nii.gz",
            "T2starmap.nii.gz",
            "S0map.nii.gz",
            "desc-rmse_statmap.nii.gz",
        ):
            orig = nb.load(op.join(out_dirs["orig"], fname)).get_fdata()
            spiked = nb.load(op.join(out_dirs["spiked"], fname)).get_fdata()
            np.testing.assert_allclose(spiked, orig, err_msg=fname)

        decay_metrics = {}
        for label in ("orig", "spiked"):
            with open(op.join(out_dirs[label], "desc-decay_metrics.json")) as fobj:
                decay_metrics[label] = json.load(fobj)

        np.testing.assert_equal(decay_metrics["spiked"], decay_metrics["orig"])

        # The per-volume RMSE confounds keep every volume; only retained rows must match
        confounds = {
            label: pd.read_table(op.join(out_dirs[label], "desc-confounds_timeseries.tsv"))
            for label in ("orig", "spiked")
        }
        assert confounds["orig"].shape[0] == nb.load(data[0]).shape[3] - dummy_scans

        # Retained volumes of the optimally combined data are unchanged as well
        n_vols = nb.load(data[0]).shape[3] - dummy_scans
        retained = np.setdiff1d(
            np.arange(n_vols),
            np.array(censored_input_vols) - dummy_scans,
        )
        orig = nb.load(op.join(out_dirs["orig"], "desc-optcom_bold.nii.gz")).get_fdata()
        spiked = nb.load(op.join(out_dirs["spiked"], "desc-optcom_bold.nii.gz")).get_fdata()
        np.testing.assert_allclose(spiked[..., retained], orig[..., retained], rtol=1e-5)
        pd.testing.assert_frame_equal(
            confounds["spiked"].iloc[retained].reset_index(drop=True),
            confounds["orig"].iloc[retained].reset_index(drop=True),
        )

    def test_t2smap_ts_fitmode_dummy_only_exclude(self, tmp_path):
        """fitmode='ts' runs when every exclude index falls within the dummy scans."""
        data_dir = get_test_data_path()
        data = [op.join(data_dir, f"echo{echo}.nii.gz") for echo in range(1, 4)]
        out_dir = str(tmp_path / "ts")
        workflows.t2smap_workflow(
            data,
            [14.5, 38.5, 62.5],
            combmode="t2s",
            fitmode="ts",
            out_dir=out_dir,
            dummy_scans=1,
            exclude="0",
        )
        assert op.isfile(op.join(out_dir, "T2starmap.nii.gz"))
        assert not op.isfile(op.join(out_dir, "desc-censoring_timeseries.tsv"))

        # A real (post-dummy) exclusion is still rejected
        with pytest.raises(ValueError, match="not supported for fitmode='ts'"):
            workflows.t2smap_workflow(
                data,
                [14.5, 38.5, 62.5],
                combmode="t2s",
                fitmode="ts",
                out_dir=str(tmp_path / "ts_censored"),
                dummy_scans=1,
                exclude="0:2",
            )

    def test_t2smap_censoring_table_msec_tr(self, tmp_path):
        """The censoring table's times are in seconds even if the header uses msec."""
        data_dir = get_test_data_path()
        data = []
        for echo in range(1, 4):
            img = nb.load(op.join(data_dir, f"echo{echo}.nii.gz"))
            header = img.header.copy()
            zooms = list(header.get_zooms())
            zooms[3] = zooms[3] * 1000
            header.set_zooms(zooms)
            header.set_xyzt_units(xyz="mm", t="msec")
            fname = str(tmp_path / f"msec_echo{echo}.nii.gz")
            nb.Nifti1Image(np.asanyarray(img.dataobj), img.affine, header).to_filename(fname)
            data.append(fname)

        out_dir = str(tmp_path / "out")
        workflows.t2smap_workflow(
            data,
            [14.5, 38.5, 62.5],
            combmode="t2s",
            fitmode="all",
            out_dir=out_dir,
            exclude="2",
        )
        table = pd.read_table(op.join(out_dir, "desc-censoring_timeseries.tsv"))
        tr = float(nb.load(op.join(data_dir, "echo1.nii.gz")).header.get_zooms()[3])
        np.testing.assert_allclose(table["time"], np.arange(table.shape[0]) * tr)

    def teardown_method(self):
        # Clean up folders (may not exist if a test used tmp_path)
        if op.isdir("TED.echo1.t2smap"):
            rmtree("TED.echo1.t2smap")
