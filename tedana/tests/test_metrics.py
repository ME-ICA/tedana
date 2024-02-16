"""Tests for tedana.metrics."""

import os.path as op

import numpy as np
import pandas as pd
import pytest

from tedana import io, utils
from tedana.metrics import collect, dependence
from tedana.tests.utils import get_test_data_path


@pytest.fixture(scope="module")
def testdata1():
    """Data used for tests of the metrics module."""
    tes = np.array([14.5, 38.5, 62.5])
    in_files = [op.join(get_test_data_path(), f"echo{i + 1}.nii.gz") for i in range(3)]
    data_cat, ref_img = io.load_data(in_files, n_echos=len(tes))
    _, adaptive_mask = utils.make_adaptive_mask(data_cat, getsum=True)
    data_optcom = np.mean(data_cat, axis=1)
    mixing = np.random.random((data_optcom.shape[1], 50))
    io_generator = io.OutputGenerator(ref_img)
    data_dict = {
        "data_cat": data_cat,
        "tes": tes,
        "data_optcom": data_optcom,
        "adaptive_mask": adaptive_mask,
        "generator": io_generator,
        "mixing": mixing,
    }
    return data_dict


def test_smoke_generate_metrics(testdata1):
    """Smoke test for tedana.metrics.collect.generate_metrics."""
    metrics = [
        "kappa",
        "rho",
        "countnoise",
        "countsigFT2",
        "countsigFS0",
        "dice_FT2",
        "dice_FS0",
        "signal-noise_t",
        "variance explained",
        "normalized variance explained",
        "d_table_score",
    ]
    comptable = collect.generate_metrics(
        testdata1["data_cat"],
        testdata1["data_optcom"],
        testdata1["mixing"],
        testdata1["adaptive_mask"],
        testdata1["tes"],
        testdata1["generator"],
        "ICA",
        metrics=metrics,
    )
    assert isinstance(comptable, pd.DataFrame)


def test_smoke_calculate_weights():
    """Smoke test for tedana.metrics.dependence.calculate_weights."""
    n_voxels, n_volumes, n_components = 1000, 100, 50
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    weights = dependence.calculate_weights(data_optcom, mixing)
    assert weights.shape == (n_voxels, n_components)


def test_smoke_calculate_betas():
    """Smoke test for tedana.metrics.dependence.calculate_betas."""
    n_voxels, n_volumes, n_components = 1000, 100, 50
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    betas = dependence.calculate_betas(data_optcom, mixing)
    assert betas.shape == (n_voxels, n_components)


def test_smoke_calculate_psc():
    """Smoke test for tedana.metrics.dependence.calculate_psc."""
    n_voxels, n_volumes, n_components = 1000, 100, 50
    data_optcom = np.random.random((n_voxels, n_volumes))
    optcom_betas = np.random.random((n_voxels, n_components))
    psc = dependence.calculate_psc(data_optcom, optcom_betas)
    assert psc.shape == (n_voxels, n_components)


def test_smoke_calculate_z_maps():
    """Smoke test for tedana.metrics.dependence.calculate_z_maps."""
    n_voxels, n_components = 1000, 50
    weights = np.random.random((n_voxels, n_components))
    z_maps = dependence.calculate_z_maps(weights, z_max=4)
    assert z_maps.shape == (n_voxels, n_components)


def test_smoke_calculate_f_maps():
    """Smoke test for tedana.metrics.dependence.calculate_f_maps."""
    n_voxels, n_echos, n_volumes, n_components = 1000, 5, 100, 50
    data_cat = np.random.random((n_voxels, n_echos, n_volumes))
    z_maps = np.random.normal(size=(n_voxels, n_components))
    mixing = np.random.random((n_volumes, n_components))
    adaptive_mask = np.random.randint(1, n_echos + 1, size=n_voxels)
    tes = np.array([15, 25, 35, 45, 55])
    f_t2_maps, f_s0_maps, _, _ = dependence.calculate_f_maps(
        data_cat, z_maps, mixing, adaptive_mask, tes, f_max=500
    )
    assert f_t2_maps.shape == f_s0_maps.shape == (n_voxels, n_components)


def test_smoke_calculate_varex():
    """Smoke test for tedana.metrics.dependence.calculate_varex."""
    n_voxels, n_components = 1000, 50
    optcom_betas = np.random.random((n_voxels, n_components))
    varex = dependence.calculate_varex(optcom_betas)
    assert varex.shape == (n_components,)


def test_smoke_calculate_varex_norm():
    """Smoke test for tedana.metrics.dependence.calculate_varex_norm."""
    n_voxels, n_components = 1000, 50
    weights = np.random.random((n_voxels, n_components))
    varex_norm = dependence.calculate_varex_norm(weights)
    assert varex_norm.shape == (n_components,)


def test_smoke_compute_dice():
    """Smoke test for tedana.metrics.dependence.compute_dice."""
    n_voxels, n_components = 1000, 50
    clmaps1 = np.random.randint(0, 2, size=(n_voxels, n_components))
    clmaps2 = np.random.randint(0, 2, size=(n_voxels, n_components))
    dice = dependence.compute_dice(clmaps1, clmaps2, axis=0)
    assert dice.shape == (n_components,)
    dice = dependence.compute_dice(clmaps1, clmaps2, axis=1)
    assert dice.shape == (n_voxels,)


def test_smoke_compute_signal_minus_noise_z():
    """Smoke test for tedana.metrics.dependence.compute_signal_minus_noise_z."""
    n_voxels, n_components = 1000, 50
    z_maps = np.random.normal(size=(n_voxels, n_components))
    z_clmaps = np.random.randint(0, 2, size=(n_voxels, n_components))
    f_t2_maps = np.random.random((n_voxels, n_components))
    (
        signal_minus_noise_z,
        signal_minus_noise_p,
    ) = dependence.compute_signal_minus_noise_z(z_maps, z_clmaps, f_t2_maps, z_thresh=1.95)
    assert signal_minus_noise_z.shape == signal_minus_noise_p.shape == (n_components,)


def test_smoke_compute_signal_minus_noise_t():
    """Smoke test for tedana.metrics.dependence.compute_signal_minus_noise_t."""
    n_voxels, n_components = 1000, 50
    z_maps = np.random.normal(size=(n_voxels, n_components))
    z_clmaps = np.random.randint(0, 2, size=(n_voxels, n_components))
    f_t2_maps = np.random.random((n_voxels, n_components))
    (
        signal_minus_noise_t,
        signal_minus_noise_p,
    ) = dependence.compute_signal_minus_noise_t(z_maps, z_clmaps, f_t2_maps, z_thresh=1.95)
    assert signal_minus_noise_t.shape == signal_minus_noise_p.shape == (n_components,)


def test_smoke_compute_countsignal():
    """Smoke test for tedana.metrics.dependence.compute_countsignal."""
    n_voxels, n_components = 1000, 50
    stat_cl_maps = np.random.randint(0, 2, size=(n_voxels, n_components))
    countsignal = dependence.compute_countsignal(stat_cl_maps)
    assert countsignal.shape == (n_components,)


def test_smoke_compute_countnoise():
    """Smoke test for tedana.metrics.dependence.compute_countnoise."""
    n_voxels, n_components = 1000, 50
    stat_maps = np.random.normal(size=(n_voxels, n_components))
    stat_cl_maps = np.random.randint(0, 2, size=(n_voxels, n_components))
    countnoise = dependence.compute_countnoise(stat_maps, stat_cl_maps, stat_thresh=1.95)
    assert countnoise.shape == (n_components,)


def test_smoke_generate_decision_table_score():
    """Smoke test for tedana.metrics.dependence.generate_decision_table_score."""
    n_voxels, n_components = 1000, 50
    kappa = np.random.random(n_components)
    dice_ft2 = np.random.random(n_components)
    signal_minus_noise_t = np.random.normal(size=n_components)
    countnoise = np.random.randint(0, n_voxels, size=n_components)
    countsigft2 = np.random.randint(0, n_voxels, size=n_components)
    decision_table_score = dependence.generate_decision_table_score(
        kappa, dice_ft2, signal_minus_noise_t, countnoise, countsigft2
    )
    assert decision_table_score.shape == (n_components,)


def test_smoke_calculate_dependence_metrics():
    """Smoke test for tedana.metrics.dependence.calculate_dependence_metrics."""
    n_voxels, n_components = 1000, 50
    f_t2_maps = np.random.random((n_voxels, n_components))
    f_s0_maps = np.random.random((n_voxels, n_components))
    z_maps = np.random.random((n_voxels, n_components))
    kappas, rhos = dependence.calculate_dependence_metrics(f_t2_maps, f_s0_maps, z_maps)
    assert kappas.shape == rhos.shape == (n_components,)
