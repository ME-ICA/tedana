"""Tests for t2smap."""

import os.path as op
from shutil import rmtree

import nibabel as nib

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
        img = nib.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
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
        img = nib.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nib.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nib.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nib.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nib.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
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
        img = nib.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
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
        img = nib.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nib.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nib.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 4
        img = nib.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 4
        img = nib.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
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
        img = nib.load(op.join(out_dir, "T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-limited_T2starmap.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-limited_S0map.nii.gz"))
        assert len(img.shape) == 3
        img = nib.load(op.join(out_dir, "desc-optcom_bold.nii.gz"))
        assert len(img.shape) == 4

    def teardown_method(self):
        # Clean up folders
        rmtree("TED.echo1.t2smap")
