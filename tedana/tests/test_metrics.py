"""Tests for tedana.metrics."""

import os.path as op

import numpy as np
import pandas as pd
import pytest

from tedana import io, utils
from tedana.metrics import collect, dependence, external
from tedana.metrics._utils import flip_components
from tedana.tests.test_external_metrics import (
    sample_external_regressor_config,
    sample_external_regressors,
)
from tedana.tests.utils import get_test_data_path


@pytest.fixture(scope="module")
def testdata1():
    """Data used for tests of the metrics module."""
    tes = np.array([14.5, 38.5, 62.5])
    in_files = [op.join(get_test_data_path(), f"echo{i + 1}.nii.gz") for i in range(3)]
    mask_file = op.join(get_test_data_path(), "mask.nii.gz")
    data_cat, ref_img = io.load_data(in_files, n_echos=len(tes))
    _, adaptive_mask = utils.make_adaptive_mask(
        data_cat,
        mask=mask_file,
        methods=["dropout", "decay"],
    )
    data_optcom = np.mean(data_cat, axis=1)
    mixing = np.random.random((data_optcom.shape[1], 3))
    io_generator = io.OutputGenerator(ref_img)

    # includes adaptive_mask_cut and mixing_cut which are used for ValueError tests
    #  for when dimensions do not align
    data_dict = {
        "data_cat": data_cat,
        "tes": tes,
        "data_optcom": data_optcom,
        "adaptive_mask": adaptive_mask,
        "adaptive_mask_cut": np.delete(adaptive_mask, (0), axis=0),
        "generator": io_generator,
        "mixing": mixing,
        "mixing_cut": np.delete(mixing, (0), axis=0),
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

    external_regressors, _ = sample_external_regressors()
    # these data have 50 volumes so cut external_regressors to 50 vols for these tests
    # This is just testing execution. Accuracy of values for external regressors are
    # tested in test_external_metrics
    n_vols = 5
    external_regressors = external_regressors.drop(labels=range(5, 75), axis=0)

    external_regressor_config = sample_external_regressor_config()
    external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors, external_regressor_config, n_vols
    )

    component_table, new_mixing = collect.generate_metrics(
        data_cat=testdata1["data_cat"],
        data_optcom=testdata1["data_optcom"],
        mixing=testdata1["mixing"],
        adaptive_mask=testdata1["adaptive_mask"],
        tes=testdata1["tes"],
        io_generator=testdata1["generator"],
        label="ICA",
        external_regressors=external_regressors,
        external_regressor_config=external_regressor_config_expanded,
        metrics=metrics,
    )
    assert isinstance(component_table, pd.DataFrame)
    # new_mixing should have flipped signs compared to mixing.
    # multiplying by "optimal sign" will flip the signs back so it should match
    assert (
        np.round(flip_components(new_mixing, signs=component_table["optimal sign"].to_numpy()), 4)
        == np.round(testdata1["mixing"], 4)
    ).all()


def test_generate_metrics_fails(testdata1):
    """Testing error conditions for tedana.metrics.collect.generate_metrics."""

    metrics = [
        "kappa",
        "rho",
    ]

    # missing external regressors
    external_regress = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    with pytest.raises(
        ValueError,
        match=(
            "If external_regressors is defined, then "
            "external_regressor_config also needs to be defined."
        ),
    ):
        component_table, _ = collect.generate_metrics(
            data_cat=testdata1["data_cat"],
            data_optcom=testdata1["data_optcom"],
            mixing=testdata1["mixing"],
            adaptive_mask=testdata1["adaptive_mask"],
            tes=testdata1["tes"],
            io_generator=testdata1["generator"],
            label="ICA",
            external_regressors=external_regress,
            metrics=metrics,
        )

    with pytest.raises(
        ValueError,
        match=(r"First dimensions \(number of samples\) of data_cat"),
    ):
        component_table, _ = collect.generate_metrics(
            data_cat=testdata1["data_cat"],
            data_optcom=testdata1["data_optcom"],
            mixing=testdata1["mixing"],
            adaptive_mask=testdata1["adaptive_mask_cut"],
            tes=testdata1["tes"],
            io_generator=testdata1["generator"],
            label="ICA",
            metrics=metrics,
        )

    with pytest.raises(
        ValueError,
        match=("does not match number of echoes provided"),
    ):
        component_table, _ = collect.generate_metrics(
            data_cat=testdata1["data_cat"],
            data_optcom=testdata1["data_optcom"],
            mixing=testdata1["mixing"],
            adaptive_mask=testdata1["adaptive_mask"],
            tes=testdata1["tes"][0:2],
            io_generator=testdata1["generator"],
            label="ICA",
            metrics=metrics,
        )

    with pytest.raises(
        ValueError,
        match=("Number of volumes in data_cat"),
    ):
        component_table, _ = collect.generate_metrics(
            data_cat=testdata1["data_cat"],
            data_optcom=testdata1["data_optcom"],
            mixing=testdata1["mixing_cut"],
            adaptive_mask=testdata1["adaptive_mask"],
            tes=testdata1["tes"],
            io_generator=testdata1["generator"],
            label="ICA",
            metrics=metrics,
        )


def test_smoke_calculate_weights():
    """Smoke test for tedana.metrics.dependence.calculate_weights."""
    n_voxels, n_volumes, n_components = 1000, 100, 50
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    weights = dependence.calculate_weights(data_optcom=data_optcom, mixing=mixing)
    assert weights.shape == (n_voxels, n_components)


def test_smoke_calculate_betas():
    """Smoke test for tedana.metrics.dependence.calculate_betas."""
    n_voxels, n_volumes, n_components = 1000, 100, 50
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    betas = dependence.calculate_betas(data=data_optcom, mixing=mixing)
    assert betas.shape == (n_voxels, n_components)


def test_smoke_calculate_psc():
    """Smoke test for tedana.metrics.dependence.calculate_psc."""
    n_voxels, n_volumes, n_components = 1000, 100, 50
    data_optcom = np.random.random((n_voxels, n_volumes))
    optcom_betas = np.random.random((n_voxels, n_components))
    psc = dependence.calculate_psc(data_optcom=data_optcom, optcom_betas=optcom_betas)
    assert psc.shape == (n_voxels, n_components)


def test_smoke_calculate_z_maps():
    """Smoke test for tedana.metrics.dependence.calculate_z_maps."""
    n_voxels, n_components = 1000, 50
    weights = np.random.random((n_voxels, n_components))
    z_maps = dependence.calculate_z_maps(weights=weights, z_max=4)
    assert z_maps.shape == (n_voxels, n_components)


def test_smoke_calculate_f_maps():
    """Smoke test for tedana.metrics.dependence.calculate_f_maps."""
    n_voxels, n_echos, n_volumes, n_components = 1000, 5, 100, 50
    data_cat = np.random.random((n_voxels, n_echos, n_volumes))
    z_maps = np.random.normal(size=(n_voxels, n_components))
    mixing = np.random.random((n_volumes, n_components))
    # The ordering is random, but make sure the adaptive mask always includes values of 1-5
    adaptive_mask = np.random.permutation(
        np.concatenate(
            (
                np.tile(1, int(np.round(n_voxels * 0.05))),
                np.tile(2, int(np.round(n_voxels * 0.1))),
                np.tile(3, int(np.round(n_voxels * 0.4))),
                np.tile(4, int(np.round(n_voxels * 0.2))),
                np.tile(5, int(np.round(n_voxels * 0.25))),
            )
        )
    )
    tes = np.array([15, 25, 35, 45, 55])
    f_t2_maps_orig, f_s0_maps_orig, _, _ = dependence.calculate_f_maps(
        data_cat=data_cat,
        z_maps=z_maps,
        mixing=mixing,
        adaptive_mask=adaptive_mask,
        tes=tes,
        f_max=500,
    )
    assert f_t2_maps_orig.shape == f_s0_maps_orig.shape == (n_voxels, n_components)

    # rerunning with n_independent_echos=3
    f_t2_maps, f_s0_maps, _, _ = dependence.calculate_f_maps(
        data_cat=data_cat,
        z_maps=z_maps,
        mixing=mixing,
        adaptive_mask=adaptive_mask,
        tes=tes,
        n_independent_echos=3,
        f_max=500,
    )
    assert f_t2_maps.shape == f_s0_maps.shape == (n_voxels, n_components)
    # exclude voxels f==0 and f==f_max since the >0 clause for 5 echoes wouldn't be true
    noextreme_f_mask = np.logical_and(
        np.logical_and(
            np.logical_and(f_t2_maps_orig > 0.0, f_s0_maps_orig > 0.0), f_t2_maps_orig < 500
        ),
        f_s0_maps_orig < 500,
    )
    # When n_independent_echos == the number of echoes (3),
    # then f_maps_orig should equal f_maps
    echo3_mask = np.logical_and(np.tile(adaptive_mask == 3, (50, 1)).T, noextreme_f_mask)
    assert np.round(
        np.min(f_t2_maps_orig[echo3_mask] - f_t2_maps[echo3_mask]), decimals=3
    ) == np.round(0.0, decimals=3)
    assert np.round(
        np.min(f_s0_maps_orig[echo3_mask] - f_s0_maps[echo3_mask]), decimals=3
    ) == np.round(0.0, decimals=3)
    assert np.round(
        np.max(f_t2_maps_orig[echo3_mask] - f_t2_maps[echo3_mask]), decimals=3
    ) == np.round(0.0, decimals=3)
    assert np.round(
        np.max(f_s0_maps_orig[echo3_mask] - f_s0_maps[echo3_mask]), decimals=3
    ) == np.round(0.0, decimals=3)
    # When n_independent_echos==3, there are 5 good echoes,
    # then f_maps_orig should always be larger than f_maps with fewer DOF
    echo5_mask = np.logical_and(np.tile(adaptive_mask == 5, (50, 1)).T, noextreme_f_mask)
    assert np.min(f_t2_maps_orig[echo5_mask] - f_t2_maps[echo5_mask]) > 0.0
    assert np.min(f_s0_maps_orig[echo5_mask] - f_s0_maps[echo5_mask]) > 0.0


def test_smoke_calculate_varex():
    """Smoke test for tedana.metrics.dependence.calculate_varex."""
    n_voxels, n_components = 1000, 50
    optcom_betas = np.random.random((n_voxels, n_components))
    varex = dependence.calculate_varex(optcom_betas=optcom_betas)
    assert varex.shape == (n_components,)


def test_smoke_calculate_varex_norm():
    """Smoke test for tedana.metrics.dependence.calculate_varex_norm."""
    n_voxels, n_components = 1000, 50
    weights = np.random.random((n_voxels, n_components))
    varex_norm = dependence.calculate_varex_norm(weights=weights)
    assert varex_norm.shape == (n_components,)


def test_smoke_compute_dice():
    """Smoke test for tedana.metrics.dependence.compute_dice."""
    n_voxels, n_components = 1000, 50
    clmaps1 = np.random.randint(0, 2, size=(n_voxels, n_components))
    clmaps2 = np.random.randint(0, 2, size=(n_voxels, n_components))
    dice = dependence.compute_dice(
        clmaps1=clmaps1,
        clmaps2=clmaps2,
        axis=0,
    )
    assert dice.shape == (n_components,)
    dice = dependence.compute_dice(
        clmaps1=clmaps1,
        clmaps2=clmaps2,
        axis=1,
    )
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
    ) = dependence.compute_signal_minus_noise_z(
        z_maps=z_maps,
        z_clmaps=z_clmaps,
        f_t2_maps=f_t2_maps,
        z_thresh=1.95,
    )
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
    ) = dependence.compute_signal_minus_noise_t(
        z_maps=z_maps,
        z_clmaps=z_clmaps,
        f_t2_maps=f_t2_maps,
        z_thresh=1.95,
    )
    assert signal_minus_noise_t.shape == signal_minus_noise_p.shape == (n_components,)


def test_smoke_compute_countsignal():
    """Smoke test for tedana.metrics.dependence.compute_countsignal."""
    n_voxels, n_components = 1000, 50
    stat_cl_maps = np.random.randint(0, 2, size=(n_voxels, n_components))
    countsignal = dependence.compute_countsignal(stat_cl_maps=stat_cl_maps)
    assert countsignal.shape == (n_components,)


def test_smoke_compute_countnoise():
    """Smoke test for tedana.metrics.dependence.compute_countnoise."""
    n_voxels, n_components = 1000, 50
    stat_maps = np.random.normal(size=(n_voxels, n_components))
    stat_cl_maps = np.random.randint(0, 2, size=(n_voxels, n_components))
    countnoise = dependence.compute_countnoise(
        stat_maps=stat_maps,
        stat_cl_maps=stat_cl_maps,
        stat_thresh=1.95,
    )
    assert countnoise.shape == (n_components,)


def test_smoke_generate_decision_table_score():
    """Smoke test for tedana.metrics.dependence.generate_decision_table_score."""
    n_voxels, n_components = 1000, 50
    kappa = np.random.random(n_components)
    dice_ft2 = np.random.random(n_components)
    signal_minus_noise_t = np.random.normal(size=n_components)
    countnoise = np.random.randint(0, n_voxels, size=n_components)
    countsig_ft2 = np.random.randint(0, n_voxels, size=n_components)
    decision_table_score = dependence.generate_decision_table_score(
        kappa=kappa,
        dice_ft2=dice_ft2,
        signal_minus_noise_t=signal_minus_noise_t,
        countnoise=countnoise,
        countsig_ft2=countsig_ft2,
    )
    assert decision_table_score.shape == (n_components,)


def test_smoke_calculate_dependence_metrics():
    """Smoke test for tedana.metrics.dependence.calculate_dependence_metrics."""
    n_voxels, n_components = 1000, 50
    f_t2_maps = np.random.random((n_voxels, n_components))
    f_s0_maps = np.random.random((n_voxels, n_components))
    z_maps = np.random.random((n_voxels, n_components))
    kappas, rhos = dependence.calculate_dependence_metrics(
        f_t2_maps=f_t2_maps,
        f_s0_maps=f_s0_maps,
        z_maps=z_maps,
    )
    assert kappas.shape == rhos.shape == (n_components,)
