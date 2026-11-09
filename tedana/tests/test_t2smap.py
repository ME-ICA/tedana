"""Tests for t2smap."""

import json
import os.path as op
from shutil import rmtree

import nibabel as nb
import numpy as np
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


def _make_small_mask(mask_file, out_dir):
    """Restrict a mask to a small central box so per-voxel NLLS tests stay fast."""
    img = nb.load(mask_file)
    box = np.zeros(img.shape, dtype=bool)
    center = [dim // 2 for dim in img.shape]
    box[
        center[0] - 5 : center[0] + 5,
        center[1] - 5 : center[1] + 5,
        center[2] - 2 : center[2] + 2,
    ] = True
    small = nb.Nifti1Image((np.asanyarray(img.dataobj) > 0) & box, img.affine, img.header)
    small.set_data_dtype(np.uint8)
    path = op.join(out_dir, "small_mask.nii.gz")
    small.to_filename(path)
    return path


@pytest.mark.parametrize("fitmode", ["all", "ts", "varys0"])
def test_t2smap_nlls(tmp_path, fitmode):
    """t2smap runs with fittype='nlls' across fitmodes and writes expected maps."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    mask = _make_small_mask(op.join(data_dir, "mask.nii.gz"), str(tmp_path))
    phase_files = _make_zero_phase_files(echo_files, str(tmp_path))
    workflows.t2smap_workflow(
        data=echo_files,
        tes=[14.5, 38.5, 62.5],
        phase=phase_files,
        mask=mask,
        fittype="nlls",
        fitmode=fitmode,
        out_dir=str(tmp_path),
        verbose=True,
    )
    assert op.exists(op.join(tmp_path, "T2starmap.nii.gz"))
    # S0 is complex, so its magnitude and phase are labeled with the BIDS part entity.
    assert op.exists(op.join(tmp_path, "part-mag_S0map.nii.gz"))
    assert op.exists(op.join(tmp_path, "part-mag_desc-limited_S0map.nii.gz"))
    assert not op.exists(op.join(tmp_path, "S0map.nii.gz"))
    assert not op.exists(op.join(tmp_path, "desc-limited_S0map.nii.gz"))
    assert op.exists(op.join(tmp_path, "desc-nlls_fieldmap.nii.gz"))
    assert op.exists(op.join(tmp_path, "part-phase_S0map.nii.gz"))
    assert op.exists(op.join(tmp_path, "desc-optcom_bold.nii.gz"))
    # BIDS requires Units for fieldmap and phase images.
    with open(op.join(tmp_path, "desc-nlls_fieldmap.json")) as fo:
        assert json.load(fo)["Units"] == "Hz"
    with open(op.join(tmp_path, "part-phase_S0map.json")) as fo:
        assert json.load(fo)["Units"] == "rad"
    # varys0 should produce a 4D S0 timeseries; all/ts produce 3D/4D maps respectively
    s0 = nb.load(op.join(tmp_path, "part-mag_S0map.nii.gz"))
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


def _write_phase_files(echo_files, out_dir, n_vols=None):
    """Write zero-valued phase NIfTIs, optionally truncated to ``n_vols`` volumes."""
    phase_files = _make_zero_phase_files(echo_files, out_dir)
    if n_vols is not None:
        for path in phase_files:
            img = nb.load(path)
            nb.Nifti1Image(img.get_fdata()[..., :n_vols], img.affine, img.header).to_filename(path)
    return phase_files


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fittype": "loglin", "phase": "PHASE"}, "only used with fittype='nlls'"),
        ({"fittype": "nlls", "phase": "PHASE0"}, "same number of files"),
        ({"fittype": "nlls", "phase": "SHORT"}, "does not match data shape"),
    ],
)
def test_t2smap_phase_validation(tmp_path, kwargs, match):
    """Phase inputs are validated against the fit type and the magnitude data."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    phase = {
        "PHASE": lambda: _make_zero_phase_files(echo_files, str(tmp_path)),
        "PHASE0": lambda: _make_zero_phase_files(echo_files, str(tmp_path))[0],
        "SHORT": lambda: _write_phase_files(echo_files, str(tmp_path), n_vols=3),
    }[kwargs["phase"]]()
    with pytest.raises(ValueError, match=match):
        workflows.t2smap_workflow(
            data=echo_files,
            tes=[14.5, 38.5, 62.5],
            mask=_make_small_mask(op.join(data_dir, "mask.nii.gz"), str(tmp_path)),
            out_dir=str(tmp_path / "out"),
            **{**kwargs, "phase": phase},
        )


def test_t2smap_nlls_dummy_scans(tmp_path):
    """Dummy scans are removed from the phase data as well as the magnitude data."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    workflows.t2smap_workflow(
        data=echo_files,
        tes=[14.5, 38.5, 62.5],
        phase=_make_zero_phase_files(echo_files, str(tmp_path)),
        mask=_make_small_mask(op.join(data_dir, "mask.nii.gz"), str(tmp_path)),
        fittype="nlls",
        dummy_scans=2,
        out_dir=str(tmp_path / "out"),
    )
    optcom = nb.load(op.join(tmp_path, "out", "desc-optcom_bold.nii.gz"))
    assert optcom.shape[-1] == nb.load(echo_files[0]).shape[-1] - 2


@pytest.mark.parametrize("fitmode, n_failures", [("all", 3), ("varys0", 3), ("all", 0)])
def test_t2smap_nlls_interpolates_failures(tmp_path, monkeypatch, caplog, fitmode, n_failures):
    """Failing nlls fits are interpolated, including volume-wise S0 for varys0."""
    from tedana import decay

    fit_complex_decay = decay.fit_complex_decay

    def fit_with_failures(*args, **kwargs):
        result = fit_complex_decay(*args, **kwargs)
        result["failures"][:] = False
        result["failures"][:n_failures] = True
        result["t2s"][:n_failures] = np.nan
        result["s0"][:n_failures] = np.nan
        return result

    monkeypatch.setattr(decay, "fit_complex_decay", fit_with_failures)
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    mask = _make_small_mask(op.join(data_dir, "mask.nii.gz"), str(tmp_path))
    out_dir = tmp_path / "out"
    workflows.t2smap_workflow(
        data=echo_files,
        tes=[14.5, 38.5, 62.5],
        phase=_make_zero_phase_files(echo_files, str(tmp_path)),
        mask=mask,
        fittype="nlls",
        fitmode=fitmode,
        interpolate_failing_voxels=True,
        out_dir=str(out_dir),
    )
    in_mask = nb.load(mask).get_fdata() > 0
    t2s = nb.load(op.join(out_dir, "T2starmap.nii.gz")).get_fdata()[in_mask]
    s0 = nb.load(op.join(out_dir, "part-mag_S0map.nii.gz")).get_fdata()[in_mask]
    assert np.isfinite(t2s).all() and np.isfinite(s0).all()
    failures = nb.load(op.join(out_dir, "desc-decayFitFailures_mask.nii.gz")).get_fdata()
    assert failures.sum() == n_failures
    if n_failures:
        assert "Interpolating T2*/S0 estimates" in caplog.text
    else:
        assert "No nlls failures found" in caplog.text


def test_t2smap_interpolation_warns_for_loglin(tmp_path, caplog):
    """Interpolation is only available for nonlinear fits."""
    data_dir = get_test_data_path()
    echo_files = [op.join(data_dir, f"echo{i + 1}.nii.gz") for i in range(3)]
    workflows.t2smap_workflow(
        data=echo_files,
        tes=[14.5, 38.5, 62.5],
        mask=_make_small_mask(op.join(data_dir, "mask.nii.gz"), str(tmp_path)),
        fittype="loglin",
        interpolate_failing_voxels=True,
        out_dir=str(tmp_path / "out"),
    )
    assert "interpolation will be skipped" in caplog.text
