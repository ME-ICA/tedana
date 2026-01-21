"""Tests for tedana.metrics.dependence."""

import numpy as np
import pytest
from scipy import stats

from tedana.metrics import dependence


def test_calculate_varex_correctness():
    """Test numerical correctness of calculate_varex."""
    # Create simple test case with known values
    component_maps = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

    # Calculate expected values manually
    # compvar = sum of squared betas for each component
    # compvar = [1^2+2^2+3^2, 2^2+3^2+4^2, 3^2+4^2+5^2]
    # compvar = [14, 29, 50]
    # total = 93
    # varex = 100 * [14/93, 29/93, 50/93]

    varex = dependence.calculate_varex(component_maps=component_maps)

    expected = 100 * np.array([14.0, 29.0, 50.0]) / 93.0
    assert np.allclose(varex, expected)

    # Test that variance explained sums to 100
    assert np.isclose(varex.sum(), 100.0)

    # Test with single component
    component_maps_single = np.array([[1.0], [2.0], [3.0]])
    varex_single = dependence.calculate_varex(component_maps=component_maps_single)
    assert varex_single.shape == (1,)
    assert np.isclose(varex_single[0], 100.0)


def test_calculate_marginal_r2_correctness():
    """Test numerical correctness of calculate_marginal_r2."""
    # Create simple test case with known values
    rng = np.random.default_rng(0)
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]  # corr at about r = 0.5
    mixing = rng.multivariate_normal(mean, cov, size=100000)
    mixing = stats.zscore(mixing, axis=0)
    # corr = np.corrcoef(mixing)[0, 1]
    # shared_variance = 100 * (corr**2)  # 25% of variance is shared

    data_optcom = np.sum(mixing, axis=1)[None, :]
    data_optcom = stats.zscore(data_optcom, axis=1)

    varex = dependence.calculate_marginal_r2(data_optcom=data_optcom, mixing=mixing)
    # 100% + (2 * 25%) = 150%
    assert np.isclose(varex.sum(), 150.0, atol=5)
    # they contribute equally to the variance explained, so 75% each
    assert np.isclose(varex[0], varex[1])

    with pytest.raises(ValueError):
        dependence.calculate_marginal_r2(data_optcom=data_optcom[:, :-1], mixing=mixing)

    varex_2 = dependence.calculate_marginal_r2_2(data_optcom=data_optcom, mixing=mixing)
    assert np.allclose(varex, varex_2)


def test_calculate_relative_varex_smoke():
    """Test smoke test of calculate_relative_varex."""
    n_voxels, n_components, n_volumes = 1000, 10, 100
    data_optcom = np.random.random((n_voxels, n_volumes))
    component_maps = np.random.random((n_voxels, n_components))
    relative_varex = dependence.calculate_relative_varex(
        data_optcom=data_optcom,
        component_maps=component_maps,
    )
    assert relative_varex.shape == (n_components,)


def test_calculate_semi_partial_r2_smoke():
    """Test smoke test of calculate_semi_partial_r2."""
    n_voxels, n_components, n_volumes = 1000, 10, 100
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    relative_varex = dependence.calculate_semi_partial_r2(data_optcom=data_optcom, mixing=mixing)
    assert relative_varex.shape == (n_components,)

    semipartial_r2 = dependence.calculate_semipartial_r2(data_optcom=data_optcom, mixing=mixing)
    assert semipartial_r2.shape == (n_components,)

    assert np.allclose(semipartial_r2, relative_varex)


def test_calculate_partial_r2_smoke():
    """Test smoke test of calculate_partial_r2."""
    n_voxels, n_components, n_volumes = 1000, 10, 100
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    relative_varex = dependence.calculate_partial_r2(data_optcom=data_optcom, mixing=mixing)
    assert relative_varex.shape == (n_components,)


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
