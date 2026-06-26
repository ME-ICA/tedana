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


def test_calculate_semipartial_r2_smoke():
    """Test smoke test of calculate_semipartial_r2."""
    n_voxels, n_components, n_volumes = 1000, 10, 100
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    semipartial_r2 = dependence.calculate_semipartial_r2(data_optcom=data_optcom, mixing=mixing)
    assert semipartial_r2.shape == (n_components,)

    with pytest.raises(ValueError):
        dependence.calculate_semipartial_r2(data_optcom=data_optcom[:, :-1], mixing=mixing)


def test_calculate_partial_r2_smoke():
    """Test smoke test of calculate_partial_r2."""
    n_voxels, n_components, n_volumes = 1000, 10, 100
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    semipartial_r2 = dependence.calculate_semipartial_r2(data_optcom=data_optcom, mixing=mixing)
    total_r2 = dependence.calculate_total_r2(data_optcom=data_optcom, mixing=mixing)
    partial_r2 = dependence.calculate_partial_r2(
        semipartial_r2=semipartial_r2,
        total_r2=total_r2,
    )
    assert partial_r2.shape == (n_components,)


def test_calculate_total_r2_smoke():
    """Test smoke test of calculate_total_r2."""
    n_voxels, n_components, n_volumes = 1000, 10, 100
    data_optcom = np.random.random((n_voxels, n_volumes))
    mixing = np.random.random((n_volumes, n_components))
    total_r2 = dependence.calculate_total_r2(data_optcom=data_optcom, mixing=mixing)
    assert np.isscalar(total_r2)

    with pytest.raises(ValueError):
        dependence.calculate_total_r2(data_optcom=data_optcom[:, :-1], mixing=mixing)


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
        descending=[kappa, dice_ft2, signal_minus_noise_t, countsig_ft2],
        ascending=[countnoise],
    )

    # Component with index 4 should have the best score (lowest value)
    # because it has highest kappa, dice, signal-noise, countsig and lowest countnoise
    assert np.argmin(d_table_score) == 4

    # Component with index 0 should have worst score (highest value)
    assert np.argmax(d_table_score) == 0

    # Test with single component
    d_table_single = dependence.generate_decision_table_score(
        descending=[np.array([1.0]), np.array([0.5]), np.array([2.0]), np.array([20])],
        ascending=[np.array([10])],
    )
    assert d_table_single.shape == (1,)
    # Single component: all metrics rank as 1, inverted ranks are 0
    # countnoise rank is 1 (not inverted)
    # Mean of [0, 0, 0, 1, 0] = 1/5 = 0.2
    assert np.isclose(d_table_single[0], 0.2)

    # Use only descending metrics
    d_table_descending = dependence.generate_decision_table_score(
        descending=[kappa, dice_ft2, signal_minus_noise_t, countsig_ft2],
    )
    # Component with index 4 should have the best score (lowest value)
    # because it has highest kappa, dice, signal-noise, countsig and lowest countnoise
    assert np.argmin(d_table_descending) == 4
    # Component with index 0 should have worst score (highest value)
    assert np.argmax(d_table_descending) == 0

    # Use only ascending metrics
    d_table_ascending = dependence.generate_decision_table_score(
        ascending=[countnoise],
    )
    # Component with index 4 should have the best score (lowest value)
    # because it has lowest countnoise
    assert np.argmin(d_table_ascending) == 4
    # Component with index 0 should have worst score (highest value)
    assert np.argmax(d_table_ascending) == 0

    with pytest.raises(ValueError, match="At least one of"):
        dependence.generate_decision_table_score(descending=[], ascending=[])
    with pytest.raises(ValueError, match="All metric arrays must be 1-D"):
        dependence.generate_decision_table_score(
            descending=[np.array([1.0, 2.0])],
            ascending=[np.array([10])],
        )


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
    stat_thresh = 1.96

    countnoise = dependence.compute_countnoise(
        stat_maps=stat_maps, stat_cl_maps=stat_cl_maps, value_threshold=stat_thresh
    )

    # For component 0: abs(stat_maps[:, 0]) = [3.0, 2.5, 1.0, 2.5]
    # Values > 1.96: [3.0, 2.5, 2.5] at indices [0, 1, 3]
    # stat_cl_maps at these indices: [1, 1, 0]
    # Noise voxels (>thresh and not in cluster): only index 3
    # Count: 1

    # For component 1: abs(stat_maps[:, 1]) = [1.0, 0.5, 2.5, 1.5]
    # Values > 1.96: [2.5] at index [2]
    # stat_cl_maps at index 2: 1 (in cluster)
    # Noise voxels: 0

    expected = np.array([1, 0])
    assert np.array_equal(countnoise, expected)


def _make_te_variance_inputs(seed=0, n_vox=200, n_echos=5):
    """Build controlled echo-wise betas with known T2*/S0 character.

    Returns inputs at the *seconds* scale (tedana's internal convention, per
    BIDS), with one perfectly T2*-driven component, one perfectly S0-driven
    component, and one pure-noise component.
    """
    rng = np.random.default_rng(seed)
    # Seconds per BIDS convention -- this is what tedana passes internally.
    tes = np.array([0.0142, 0.03893, 0.06366, 0.08839, 0.11312])[:n_echos]
    # Spatially varying T2* (so the permutation test has power), in seconds, and
    # realistic S0 magnitudes (thousands).
    t2s = rng.uniform(0.03, 0.07, n_vox)
    s0 = rng.uniform(3000.0, 8000.0, n_vox)
    adaptive_mask = np.full(n_vox, n_echos, dtype=int)

    echo_betas = np.zeros((n_vox, n_echos, 3))
    for v in range(n_vox):
        phi_s0 = np.exp(-tes / t2s[v])
        phi_t2 = s0[v] * np.exp(-tes / t2s[v]) * tes / t2s[v] ** 2
        phi_s0 /= np.linalg.norm(phi_s0)
        phi_t2 /= np.linalg.norm(phi_t2)
        echo_betas[v, :, 0] = phi_t2 + 0.02 * rng.standard_normal(n_echos)  # T2*
        echo_betas[v, :, 1] = phi_s0 + 0.02 * rng.standard_normal(n_echos)  # S0
        echo_betas[v, :, 2] = rng.standard_normal(n_echos)  # noise

    spatial_weights = np.ones((n_vox, 3))
    return tes, t2s, s0, adaptive_mask, echo_betas, spatial_weights


def test_compute_te_variance_valid_at_seconds_scale():
    """compute_te_variance must return finite fractions for seconds-scale T2*.

    Regression test for the unit bug: the t2s_min/t2s_max validity thresholds
    were millisecond-scale (5/500), which rejected every voxel when tedana
    passes seconds-scale T2* (~0.05 s), yielding all-NaN output. With the
    default thresholds expressed in seconds the voxels are retained.
    """
    tes, t2s, s0, amask, betas, weights = _make_te_variance_inputs()

    kappa_star, rho_star, _, _, _ = dependence.compute_te_variance(
        echowise_pes=betas,
        tes=tes,
        s0_hat=s0,
        t2s_hat=t2s,
        adaptive_mask=amask,
        spatial_weights=weights,
    )

    assert np.all(np.isfinite(kappa_star))
    assert np.all(np.isfinite(rho_star))
    # The perfectly T2*-driven component reads mostly T2*; the S0 one mostly S0.
    assert kappa_star[0] > 0.8
    assert rho_star[1] > 0.8


def test_compute_te_variance_unit_invariant_with_matching_bounds():
    """The decomposition is unit-invariant when the validity bounds are scaled too.

    kappa_star/rho_star depend only on basis *shapes* (ratios of TE/T2*), so a
    seconds->milliseconds rescale of TE, T2*, and the validity thresholds must
    leave the result unchanged.
    """
    tes, t2s, s0, amask, betas, weights = _make_te_variance_inputs()

    ks_s, rs_s, _, _, r2_s = dependence.compute_te_variance(
        echowise_pes=betas,
        tes=tes,
        s0_hat=s0,
        t2s_hat=t2s,
        adaptive_mask=amask,
        spatial_weights=weights,
        t2s_min=0.005,
        t2s_max=0.5,
    )
    ks_ms, rs_ms, _, _, r2_ms = dependence.compute_te_variance(
        echowise_pes=betas,
        tes=tes * 1000.0,
        s0_hat=s0,
        t2s_hat=t2s * 1000.0,
        adaptive_mask=amask,
        spatial_weights=weights,
        t2s_min=5.0,
        t2s_max=500.0,
    )

    assert np.allclose(ks_s, ks_ms, atol=1e-6, equal_nan=True)
    assert np.allclose(rs_s, rs_ms, atol=1e-6, equal_nan=True)
    assert np.allclose(r2_s, r2_ms, atol=1e-6, equal_nan=True)


def test_compute_te_variance_permutation_not_degenerate_at_seconds_scale():
    """Permutation p-values must not be degenerate for seconds-scale T2*.

    With the millisecond-scale thresholds, every seconds-scale voxel was
    dropped and the permutation null collapsed (degenerate p-values). With
    seconds-scale thresholds a perfectly T2*-aligned component yields a low
    p_t2, the S0-aligned component a low p_s0, and noise neither.
    """
    tes, t2s, s0, amask, betas, weights = _make_te_variance_inputs()

    _, _, p_t2, p_s0 = dependence.compute_te_variance_permutation(
        echowise_pes=betas,
        tes=tes,
        s0_hat=s0,
        t2s_hat=t2s,
        adaptive_mask=amask,
        spatial_weights=weights,
        n_perm=500,
        n_threads=1,
        seed=42,
    )

    assert p_t2[0] < 0.05  # T2*-aligned component detected
    assert p_s0[1] < 0.05  # S0-aligned component detected
    assert p_t2[2] > 0.05  # noise not flagged as T2*-specific


def test_compute_te_variance_permutation_reproducible_and_thread_invariant():
    """Permutation p-values are reproducible and independent of n_threads.

    Guards the memory refactor (on-the-fly, per-permutation seeded index
    generation): the same seed must give identical p-values, and running the
    permutations across multiple threads must match the single-thread result.
    """
    tes, t2s, s0, amask, betas, weights = _make_te_variance_inputs()
    kwargs = dict(
        echowise_pes=betas,
        tes=tes,
        s0_hat=s0,
        t2s_hat=t2s,
        adaptive_mask=amask,
        spatial_weights=weights,
        n_perm=200,
        seed=7,
    )

    _, _, p_t2_a, p_s0_a = dependence.compute_te_variance_permutation(n_threads=1, **kwargs)
    _, _, p_t2_b, p_s0_b = dependence.compute_te_variance_permutation(n_threads=1, **kwargs)
    _, _, p_t2_par, p_s0_par = dependence.compute_te_variance_permutation(n_threads=2, **kwargs)

    # Same seed -> identical results.
    assert np.array_equal(p_t2_a, p_t2_b)
    assert np.array_equal(p_s0_a, p_s0_b)
    # Threading must not change the result.
    assert np.array_equal(p_t2_a, p_t2_par)
    assert np.array_equal(p_s0_a, p_s0_par)


def test_compute_te_variance_permutation_chunk_invariant():
    """Processing components in chunks must not change the result.

    Component chunking bounds peak memory (it avoids holding the einsum
    intermediates for every component at once), so it must produce results
    identical to computing all components in a single chunk.
    """
    tes, t2s, s0, amask, betas, weights = _make_te_variance_inputs(n_vox=150)
    kwargs = dict(
        echowise_pes=betas,
        tes=tes,
        s0_hat=s0,
        t2s_hat=t2s,
        adaptive_mask=amask,
        spatial_weights=weights,
        n_perm=150,
        n_threads=1,
        seed=11,
    )

    ks_all, rs_all, p_t2_all, p_s0_all = dependence.compute_te_variance_permutation(
        n_comp_chunk=1000, **kwargs
    )
    ks_ch, rs_ch, p_t2_ch, p_s0_ch = dependence.compute_te_variance_permutation(
        n_comp_chunk=1, **kwargs
    )

    assert np.array_equal(p_t2_all, p_t2_ch)
    assert np.array_equal(p_s0_all, p_s0_ch)
    assert np.allclose(ks_all, ks_ch, equal_nan=True)
    assert np.allclose(rs_all, rs_ch, equal_nan=True)
