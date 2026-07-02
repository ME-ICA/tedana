"""Tests for t2smap."""

import os.path as op
from shutil import rmtree

import nibabel as nb
import numpy as np
import pytest

from tedana import workflows
from tedana.tests.utils import get_test_data_path


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
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
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
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
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
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
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
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nb.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

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
        img = nb.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nb.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
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
        with pytest.raises(ValueError, match="The maximum exclude index"):
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

    def teardown_method(self):
        # Clean up folders (may not exist if a test used tmp_path)
        if op.isdir("TED.echo1.t2smap"):
            rmtree("TED.echo1.t2smap")


def _make_zero_phase_files(echo_files, out_dir):
    """Write zero-valued phase NIfTIs matching the given magnitude echoes."""
    phase_files = []
    for i, f in enumerate(echo_files):
        img = nb.load(f)
        phase = nb.Nifti1Image(np.zeros(img.shape, dtype=np.float32), img.affine, img.header)
        path = op.join(out_dir, f"phase{i + 1}.nii.gz")
        phase.to_filename(path)
        phase_files.append(path)
    return phase_files


@pytest.mark.parametrize("fitmode", ["all", "ts", "varys0"])
def test_t2smap_nlls(tmp_path, fitmode):
    """t2smap runs with fittype='nlls' across fitmodes and writes expected maps."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    mask = op.join(data_dir, "mask.nii.gz")
    phase_files = _make_zero_phase_files(echo_files, str(tmp_path))
    workflows.t2smap_workflow(
        data=echo_files,
        tes=[14.5, 38.5, 62.5],
        phase=phase_files,
        mask=mask,
        fittype="nlls",
        fitmode=fitmode,
        out_dir=str(tmp_path),
    )
    assert op.exists(op.join(tmp_path, "T2starmap.nii.gz"))
    assert op.exists(op.join(tmp_path, "S0map.nii.gz"))
    assert op.exists(op.join(tmp_path, "frequencyHzmap.nii.gz"))
    assert op.exists(op.join(tmp_path, "phase0map.nii.gz"))
    assert op.exists(op.join(tmp_path, "desc-optcom_bold.nii.gz"))
    # varys0 should produce a 4D S0 timeseries; all/ts produce 3D/4D maps respectively
    s0 = nb.load(op.join(tmp_path, "S0map.nii.gz"))
    if fitmode == "varys0":
        assert s0.ndim == 4


def test_t2smap_nlls_requires_phase(tmp_path):
    """fittype='nlls' without --phase raises."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    with pytest.raises(ValueError, match="requires phase"):
        workflows.t2smap_workflow(
            data=echo_files,
            tes=[14.5, 38.5, 62.5],
            fittype="nlls",
            out_dir=str(tmp_path),
        )


def test_t2smap_varys0_requires_nlls(tmp_path):
    """fitmode='varys0' with a non-nlls fittype raises."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    with pytest.raises(ValueError, match="varys0"):
        workflows.t2smap_workflow(
            data=echo_files,
            tes=[14.5, 38.5, 62.5],
            fittype="loglin",
            fitmode="varys0",
            out_dir=str(tmp_path),
        )
