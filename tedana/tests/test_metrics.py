"""Tests for tedana.metrics."""

import os.path as op

import numpy as np
import pandas as pd
import pytest

from tedana import io, utils
from tedana.metrics import collect, dependence, external
from tedana.metrics._utils import (
    add_external_dependencies,
    check_mask,
    dependency_resolver,
    determine_signs,
    flip_components,
)
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
        "kappa_rho_difference",
    ]

    external_regressors, _ = sample_external_regressors()
    # these data have 50 volumes so cut external_regressors to 50 vols for these tests
    # This is just testing execution. Accuracy of values for external regressors are
    # tested in test_external_metrics
    n_vols = 5
    external_regressors = external_regressors.drop(labels=range(5, 75), axis=0)

    external_regressor_config = sample_external_regressor_config()
    external_regressors, external_regressor_config_expanded = external.validate_extern_regress(
        external_regressors=external_regressors,
        external_regressor_config=external_regressor_config,
        n_vols=n_vols,
        dummy_scans=0,
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


def test_smoke_compute_kappa_rho_difference():
    """Smoke test for tedana.metrics.dependence.compute_kappa_rho_difference."""
    n_components = 50
    kappa = np.random.random(n_components)
    rho = np.random.random(n_components)
    kappa_rho_difference = dependence.compute_kappa_rho_difference(
        kappa=kappa,
        rho=rho,
    )
    assert kappa_rho_difference.shape == (n_components,)


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


# Comprehensive tests for _utils.py functions


def test_determine_signs():
    """Test determine_signs function for component sign determination."""
    # Test with right-skewed data (should return positive sign)
    weights_right_skew = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [10, 15]])
    signs = determine_signs(weights_right_skew, axis=0)
    assert signs.shape == (2,)
    assert np.all(signs == 1)  # Right-skewed data should have positive signs

    # Test with left-skewed data
    weights_left_skew = np.array([[-10, -15], [-4, -5], [-3, -4], [-2, -3], [-1, -2]])
    signs = determine_signs(weights_left_skew, axis=0)
    assert signs.shape == (2,)
    assert np.all(signs == -1)  # Left-skewed data should have negative signs

    # Test with zero skew (should default to 1)
    weights_symmetric = np.array([[1, 2], [-1, -2], [0, 0]])
    signs = determine_signs(weights_symmetric, axis=0)
    assert signs.shape == (2,)
    assert np.all(np.isin(signs, [1, -1]))

    # Test single component
    weights_single = np.array([[1], [2], [3], [4], [10]])
    signs = determine_signs(weights_single, axis=0)
    assert signs.shape == (1,)
    assert signs[0] in [1, -1]


def test_flip_components():
    """Test flip_components function for sign flipping."""
    # Test basic flipping
    arr1 = np.array([[1, 2], [3, 4], [5, 6]])
    signs = np.array([1, -1])
    flipped = flip_components(arr1, signs=signs)
    assert len(flipped) == 1
    expected = np.array([[1, -2], [3, -4], [5, -6]])
    assert np.allclose(flipped[0], expected)

    # Test multiple arrays with dimensions matching sign length
    arr2 = np.array([[10, 20], [30, 40], [50, 60]])
    flipped = flip_components(arr1, arr2, signs=signs)
    assert len(flipped) == 2
    assert np.allclose(flipped[0], expected)
    assert np.allclose(flipped[1], np.array([[10, -20], [30, -40], [50, -60]]))

    # Test all positive signs (no change)
    signs_positive = np.array([1, 1])
    flipped = flip_components(arr1, signs=signs_positive)
    assert np.allclose(flipped[0], arr1)

    # Test all negative signs (flip all)
    signs_negative = np.array([-1, -1])
    flipped = flip_components(arr1, signs=signs_negative)
    assert np.allclose(flipped[0], -arr1)

    # Test with assertion error for wrong sign dimensions
    with pytest.raises(AssertionError):
        signs_2d = np.array([[1, -1]])
        flip_components(arr1, signs=signs_2d)


def test_check_mask():
    """Test check_mask function for zero-variance detection."""
    # Test with valid data (no zero variance)
    data = np.random.randn(100, 5, 10)
    mask = np.ones(100, dtype=bool)
    # Should not raise an error
    check_mask(data, mask)

    # Test with zero variance in time dimension
    data_zero_var = np.zeros((100, 5, 10))
    data_zero_var[0, :, :] = 1.0  # One voxel with constant values across time
    with pytest.raises(ValueError, match="voxels in masked data have zero variance"):
        check_mask(data_zero_var, mask)

    # Test with 2D data
    data_2d = np.random.randn(100, 10)
    check_mask(data_2d, mask)

    # Test with 2D data with zero variance
    data_2d_zero = np.ones((100, 10))
    with pytest.raises(ValueError, match="voxels in masked data have zero variance"):
        check_mask(data_2d_zero, mask)

    # Test with subset mask (should only check masked voxels)
    data_mixed = np.random.randn(100, 5, 10)
    data_mixed[0, :, :] = 0  # First voxel has zero variance
    mask_subset = np.ones(100, dtype=bool)
    mask_subset[0] = False  # Exclude the problematic voxel
    # Should not raise an error since problematic voxel is masked out
    check_mask(data_mixed, mask_subset)


def test_dependency_resolver():
    """Test dependency_resolver function."""
    # Create a simple dependency dictionary
    dependencies = {
        "metric_a": ["input1"],
        "metric_b": ["metric_a", "input2"],
        "metric_c": ["metric_b"],
    }
    base_inputs = ["input1", "input2"]

    # Test resolving single metric
    required = dependency_resolver(dependencies, ["metric_a"], base_inputs)
    assert "metric_a" in required
    assert "input1" in required

    # Test resolving metric with dependencies
    required = dependency_resolver(dependencies, ["metric_b"], base_inputs)
    assert "metric_b" in required
    assert "metric_a" in required
    assert "input1" in required
    assert "input2" in required

    # Test resolving nested dependencies
    required = dependency_resolver(dependencies, ["metric_c"], base_inputs)
    assert "metric_c" in required
    assert "metric_b" in required
    assert "metric_a" in required

    # Test with unknown metric
    with pytest.raises(ValueError, match="Unknown metric"):
        dependency_resolver(dependencies, ["unknown_metric"], base_inputs)

    # Test with multiple requested metrics
    required = dependency_resolver(dependencies, ["metric_a", "metric_c"], base_inputs)
    assert "metric_a" in required
    assert "metric_c" in required
    assert "metric_b" in required


def test_add_external_dependencies():
    """Test add_external_dependencies function."""
    # Create a basic dependency config
    dependency_config = {
        "inputs": ["data", "mixing"],
        "dependencies": {"kappa": ["data"], "rho": ["mixing"]},
    }

    # Create external regressor config with F-statistic
    external_regressor_config = [
        {
            "regress_ID": "motion",
            "statistic": "f",
            "info": "Motion parameters",
            "report": "Motion correlation",
        }
    ]

    # Add external dependencies
    updated_config = add_external_dependencies(dependency_config, external_regressor_config)

    # Check that external regressors input was added
    assert "external regressors" in updated_config["inputs"]

    # Check that F-stat dependencies were added
    assert "Fstat motion model" in updated_config["dependencies"]
    assert "R2stat motion model" in updated_config["dependencies"]
    assert "pval motion model" in updated_config["dependencies"]

    # Check dependency values
    assert updated_config["dependencies"]["Fstat motion model"] == ["external regressors"]

    # Test with partial models
    external_regressor_config_partial = [
        {
            "regress_ID": "physio",
            "statistic": "f",
            "partial_models": {"cardiac": {}, "respiratory": {}},
            "info": "Physiological",
            "report": "Physio correlation",
        }
    ]

    updated_config_partial = add_external_dependencies(
        dependency_config.copy(), external_regressor_config_partial
    )

    # Check partial model dependencies
    assert "Fstat physio cardiac partial model" in updated_config_partial["dependencies"]
    assert "Fstat physio respiratory partial model" in updated_config_partial["dependencies"]


# Comprehensive tests for dependence.py functions


def test_calculate_varex_correctness():
    """Test numerical correctness of calculate_varex."""
    # Create simple test case with known values
    optcom_betas = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

    # Calculate expected values manually
    # compvar = sum of squared betas for each component
    # compvar = [1^2+2^2+3^2, 2^2+3^2+4^2, 3^2+4^2+5^2]
    # compvar = [14, 29, 50]
    # total = 93
    # varex = 100 * [14/93, 29/93, 50/93]

    varex = dependence.calculate_varex(optcom_betas=optcom_betas)

    expected = 100 * np.array([14.0, 29.0, 50.0]) / 93.0
    assert np.allclose(varex, expected)

    # Test that variance explained sums to 100
    assert np.isclose(varex.sum(), 100.0)

    # Test with single component
    optcom_betas_single = np.array([[1.0], [2.0], [3.0]])
    varex_single = dependence.calculate_varex(optcom_betas=optcom_betas_single)
    assert varex_single.shape == (1,)
    assert np.isclose(varex_single[0], 100.0)


def test_calculate_varex_norm_correctness():
    """Test numerical correctness of calculate_varex_norm."""
    # Create simple test case
    weights = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    # Calculate expected values
    # compvar = [1^2+2^2+3^2, 2^2+3^2+4^2] = [14, 29]
    # total = 43
    # varex_norm = [14/43, 29/43]

    varex_norm = dependence.calculate_varex_norm(weights=weights)

    expected = np.array([14.0, 29.0]) / 43.0
    assert np.allclose(varex_norm, expected)

    # Test that normalized variance explained sums to 1
    assert np.isclose(varex_norm.sum(), 1.0)

    # Test with single component
    weights_single = np.array([[1.0], [2.0]])
    varex_norm_single = dependence.calculate_varex_norm(weights=weights_single)
    assert varex_norm_single.shape == (1,)
    assert np.isclose(varex_norm_single[0], 1.0)


def test_calculate_z_maps_correctness():
    """Test numerical correctness of calculate_z_maps."""
    # Create weights with known mean and std
    weights = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])

    z_maps = dependence.calculate_z_maps(weights=weights, z_max=8)

    # Check that z-scoring worked (mean should be ~0, std should be ~1)
    assert np.allclose(z_maps.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(z_maps.std(axis=0), 1.0, atol=1e-10)

    # Test z_max clamping with more extreme data to ensure clamping actually occurs
    weights_extreme = np.array(
        [[1.0], [1.1], [1.2], [1.3], [1.4], [1.5], [1.6], [1.7], [1.8], [100.0]]
    )
    z_maps_clamped = dependence.calculate_z_maps(weights=weights_extreme, z_max=2.0)
    assert np.all(np.abs(z_maps_clamped) <= 2.0)

    # Test that the known extreme value (100.0) is clamped to z_max with correct magnitude
    extreme_clamped_value = z_maps_clamped[-1, 0]
    # The extreme value should be clamped to +z_max or -z_max
    assert np.isclose(np.abs(extreme_clamped_value), 2.0)


def test_calculate_dependence_metrics_correctness():
    """Test numerical correctness of calculate_dependence_metrics."""
    # Create simple test case
    n_voxels, n_components = 10, 2
    f_t2_maps = np.ones((n_voxels, n_components)) * 5.0
    f_s0_maps = np.ones((n_voxels, n_components)) * 3.0
    z_maps = np.ones((n_voxels, n_components)) * 2.0

    kappas, rhos = dependence.calculate_dependence_metrics(
        f_t2_maps=f_t2_maps, f_s0_maps=f_s0_maps, z_maps=z_maps
    )

    # With uniform weights and uniform f-maps, kappa and rho should equal the f-map values
    # weighted average with uniform weights = uniform values
    assert np.allclose(kappas, 5.0)
    assert np.allclose(rhos, 3.0)

    # Test with varying weights
    z_maps_vary = (
        np.arange(n_voxels * n_components).reshape((n_voxels, n_components)).astype(float)
    )
    f_t2_vary = np.ones((n_voxels, n_components)) * np.arange(n_voxels)[:, np.newaxis]

    kappas_vary, _ = dependence.calculate_dependence_metrics(
        f_t2_maps=f_t2_vary, f_s0_maps=f_s0_maps, z_maps=z_maps_vary
    )

    # Kappa should be weighted average
    weight_maps = z_maps_vary**2
    expected_kappa = np.average(f_t2_vary, weights=weight_maps, axis=0)
    assert np.allclose(kappas_vary, expected_kappa)


def test_compute_dice_correctness():
    """Test numerical correctness of compute_dice."""
    # Test perfect overlap (avoid all-zero columns)
    clmaps1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    clmaps2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    dice = dependence.compute_dice(clmaps1=clmaps1, clmaps2=clmaps2, axis=0)
    assert dice.shape == (3,)
    assert np.allclose(dice, 1.0)  # Perfect overlap should give Dice = 1

    # Test no overlap
    clmaps1 = np.array([[1, 0], [1, 0]])
    clmaps2 = np.array([[0, 1], [0, 1]])
    dice = dependence.compute_dice(clmaps1=clmaps1, clmaps2=clmaps2, axis=0)
    assert np.allclose(dice, 0.0)  # No overlap should give Dice = 0

    # Test partial overlap
    clmaps1 = np.array([[1, 1], [1, 0]])
    clmaps2 = np.array([[1, 1], [0, 1]])
    dice = dependence.compute_dice(clmaps1=clmaps1, clmaps2=clmaps2, axis=0)
    # Component 0: clmaps1 has [1,1], clmaps2 has [1,0], intersection = 1
    # Dice = 2*1/(2+1) = 2/3
    # Component 1: clmaps1 has [1,0], clmaps2 has [1,1], intersection = 1
    # Dice = 2*1/(1+2) = 2/3
    assert np.isclose(dice[0], 2.0 / 3.0)
    assert np.isclose(dice[1], 2.0 / 3.0)

    # Test another partial overlap case with different values
    clmaps1 = np.array([[1, 1], [1, 1], [0, 0]])
    clmaps2 = np.array([[1, 0], [1, 0], [0, 1]])
    dice = dependence.compute_dice(clmaps1=clmaps1, clmaps2=clmaps2, axis=0)
    # Component 0: clmaps1 has [1,1,0], clmaps2 has [1,1,0], intersection = 2
    # Dice = 2*2/(2+2) = 1.0 (perfect match)
    # Component 1: clmaps1 has [1,1,0], clmaps2 has [0,0,1], intersection = 0
    # Dice = 0.0 (no overlap)
    assert np.isclose(dice[0], 1.0)
    assert np.isclose(dice[1], 0.0)


def test_compute_kappa_rho_difference_correctness():
    """Test numerical correctness of compute_kappa_rho_difference."""
    # Test with equal kappa and rho
    kappa = np.array([10.0, 20.0, 30.0])
    rho = np.array([10.0, 20.0, 30.0])
    diff = dependence.compute_kappa_rho_difference(kappa=kappa, rho=rho)
    assert np.allclose(diff, 0.0)  # Should be 0 when equal

    # Test with kappa > rho
    kappa = np.array([10.0, 20.0])
    rho = np.array([5.0, 10.0])
    diff = dependence.compute_kappa_rho_difference(kappa=kappa, rho=rho)
    # Expected: |10-5|/(10+5) = 5/15 = 1/3, |20-10|/(20+10) = 10/30 = 1/3
    expected = np.array([1.0 / 3.0, 1.0 / 3.0])
    assert np.allclose(diff, expected)

    # Test with rho > kappa
    kappa = np.array([5.0])
    rho = np.array([15.0])
    diff = dependence.compute_kappa_rho_difference(kappa=kappa, rho=rho)
    # Expected: |5-15|/(5+15) = 10/20 = 0.5
    assert np.isclose(diff[0], 0.5)

    # Test extreme case where one dominates
    kappa = np.array([100.0])
    rho = np.array([1.0])
    diff = dependence.compute_kappa_rho_difference(kappa=kappa, rho=rho)
    # Expected: |100-1|/(100+1) = 99/101 ≈ 0.98
    assert np.isclose(diff[0], 99.0 / 101.0)


def test_generate_decision_table_score_correctness():
    """Test numerical correctness of generate_decision_table_score."""
    # Create simple test case with known rankings (5 components)
    kappa = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Higher is better
    dice_ft2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Higher is better
    signal_minus_noise_t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Higher is better
    countnoise = np.array([50, 40, 30, 20, 10])  # Lower is better
    countsig_ft2 = np.array([10, 20, 30, 40, 50])  # Higher is better

    d_table_score = dependence.generate_decision_table_score(
        kappa=kappa,
        dice_ft2=dice_ft2,
        signal_minus_noise_t=signal_minus_noise_t,
        countnoise=countnoise,
        countsig_ft2=countsig_ft2,
    )

    # Component with index 4 should have the best score (lowest value)
    # because it has highest kappa, dice, signal-noise, countsig and lowest countnoise
    assert np.argmin(d_table_score) == 4

    # Component with index 0 should have worst score (highest value)
    assert np.argmax(d_table_score) == 0

    # Test with single component
    d_table_single = dependence.generate_decision_table_score(
        kappa=np.array([1.0]),
        dice_ft2=np.array([0.5]),
        signal_minus_noise_t=np.array([2.0]),
        countnoise=np.array([10]),
        countsig_ft2=np.array([20]),
    )
    assert d_table_single.shape == (1,)
    # Single component: all metrics rank as 1, inverted ranks are 0
    # countnoise rank is 1 (not inverted)
    # Mean of [0, 0, 0, 1, 0] = 1/5 = 0.2
    assert np.isclose(d_table_single[0], 0.2)


def test_compute_countsignal_correctness():
    """Test numerical correctness of compute_countsignal."""
    # Create binary maps
    stat_cl_maps = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]])

    countsignal = dependence.compute_countsignal(stat_cl_maps=stat_cl_maps)

    # Expected counts: [3, 2, 2]
    expected = np.array([3, 2, 2])
    assert np.array_equal(countsignal, expected)

    # Test with all zeros
    stat_cl_maps_zeros = np.zeros((5, 3))
    countsignal_zeros = dependence.compute_countsignal(stat_cl_maps=stat_cl_maps_zeros)
    assert np.array_equal(countsignal_zeros, np.array([0, 0, 0]))

    # Test with all ones
    stat_cl_maps_ones = np.ones((5, 3))
    countsignal_ones = dependence.compute_countsignal(stat_cl_maps=stat_cl_maps_ones)
    assert np.array_equal(countsignal_ones, np.array([5, 5, 5]))


def test_compute_countnoise_correctness():
    """Test numerical correctness of compute_countnoise."""
    # Create test data
    stat_maps = np.array([[3.0, 1.0], [2.5, 0.5], [1.0, 2.5], [-2.5, -1.5]])
    stat_cl_maps = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
    stat_thresh = 1.95

    countnoise = dependence.compute_countnoise(
        stat_maps=stat_maps, stat_cl_maps=stat_cl_maps, stat_thresh=stat_thresh
    )

    # For component 0: abs(stat_maps[:, 0]) = [3.0, 2.5, 1.0, 2.5]
    # Values > 1.95: [3.0, 2.5, 2.5] at indices [0, 1, 3]
    # stat_cl_maps at these indices: [1, 1, 0]
    # Noise voxels (>thresh and not in cluster): only index 3
    # Count: 1

    # For component 1: abs(stat_maps[:, 1]) = [1.0, 0.5, 2.5, 1.5]
    # Values > 1.95: [2.5] at index [2]
    # stat_cl_maps at index 2: 1 (in cluster)
    # Noise voxels: 0

    expected = np.array([1, 0])
    assert np.array_equal(countnoise, expected)


# Tests for collect.py functions


def test_get_metadata():
    """Test get_metadata function."""
    # Create a sample component table with various columns
    component_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01"],
            "kappa": [10.0, 20.0],
            "rho": [5.0, 3.0],
            "variance explained": [30.0, 25.0],
            "normalized variance explained": [0.3, 0.25],
            "countsigFT2": [100, 150],
            "countsigFS0": [50, 75],
            "dice_FT2": [0.8, 0.7],
            "dice_FS0": [0.6, 0.5],
            "countnoise": [10, 5],
            "signal-noise_t": [3.0, 4.0],
            "signal-noise_p": [0.01, 0.001],
            "d_table_score": [5.0, 3.0],
            "classification": ["accepted", "rejected"],
            "rationale": ["BOLD-like", "High noise"],
            "optimal sign": [1, -1],
        }
    )

    metadata = collect.get_metadata(component_table)

    # Check that metadata exists for key metrics
    assert "Component" in metadata
    assert "kappa" in metadata
    assert "rho" in metadata
    assert "variance explained" in metadata
    assert "normalized variance explained" in metadata
    assert "countsigFT2" in metadata
    assert "countsigFS0" in metadata
    assert "dice_FT2" in metadata
    assert "dice_FS0" in metadata
    assert "countnoise" in metadata
    assert "signal-noise_t" in metadata
    assert "signal-noise_p" in metadata
    assert "d_table_score" in metadata
    assert "classification" in metadata
    assert "rationale" in metadata
    assert "optimal sign" in metadata

    # Check structure of metadata entries
    assert "LongName" in metadata["kappa"]
    assert "Description" in metadata["kappa"]
    assert "Units" in metadata["kappa"]

    # Check that classification has Levels
    assert "Levels" in metadata["classification"]
    assert "accepted" in metadata["classification"]["Levels"]
    assert "rejected" in metadata["classification"]["Levels"]

    # Check optimal sign has Levels with integers
    assert "Levels" in metadata["optimal sign"]
    assert 1 in metadata["optimal sign"]["Levels"]
    assert -1 in metadata["optimal sign"]["Levels"]

    # Test with minimal component table
    minimal_table = pd.DataFrame({"Component": ["ICA_00"]})
    minimal_metadata = collect.get_metadata(minimal_table)
    # Should always have Component metadata
    assert "Component" in minimal_metadata
    # Should not have other metrics
    assert "kappa" not in minimal_metadata


def test_get_metadata_external_regressors():
    """Test get_metadata with external regressor columns."""
    component_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01"],
            "external regressor correlation motion_x": [0.5, 0.3],
            "external regressor correlation motion_y": [0.2, 0.6],
        }
    )

    metadata = collect.get_metadata(component_table)

    # Check external regressor metadata
    assert "external regressor correlation motion_x" in metadata
    assert "external regressor correlation motion_y" in metadata

    # Check structure
    motion_x_meta = metadata["external regressor correlation motion_x"]
    assert "LongName" in motion_x_meta
    assert "Description" in motion_x_meta
    assert "Units" in motion_x_meta
    assert "motion_x" in motion_x_meta["Description"]
    assert "Pearson correlation coefficient" in motion_x_meta["Units"]
