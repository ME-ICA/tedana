"""Tests for tedana.metrics._utils."""

import numpy as np
import pytest

from tedana.metrics._utils import (
    add_external_dependencies,
    dependency_resolver,
    determine_signs,
    flip_components,
    get_value_thresholds,
)


def test_determine_signs():
    """Test determine_signs function for component sign determination."""
    # Test with right-skewed data (should return positive sign)
    weights_right_skew = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [10, 15]])
    signs = determine_signs(weights_right_skew, axis=0)
    assert signs.shape == (2,)
    assert all(signs == 1)  # Right-skewed data should have positive signs

    # Test with left-skewed data
    weights_left_skew = np.array([[-10, -15], [-4, -5], [-3, -4], [-2, -3], [-1, -2]])
    signs = determine_signs(weights_left_skew, axis=0)
    assert signs.shape == (2,)
    assert all(signs == -1)  # Left-skewed data should have negative signs

    # Test with zero skew (should default to 1)
    weights_symmetric = np.array([[1, 2], [-1, -2], [0, 0]])
    signs = determine_signs(weights_symmetric, axis=0)
    assert signs.shape == (2,)
    assert all(signs == 1)

    # Test single component
    weights_single = np.array([[1], [2], [3], [4], [10]])
    signs = determine_signs(weights_single, axis=0)
    assert signs.shape == (1,)
    assert all(signs == 1)


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


def test_dependency_resolver():
    """Test dependency_resolver function."""
    # Create a simple dependency dictionary
    dependencies = {
        "metric_a": ["input1"],
        "metric_b": ["metric_a", "input2"],
        "metric_c": ["metric_b"],
        "map Z": ["input1"],
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

    with pytest.raises(ValueError, match="The following metrics are no longer supported:"):
        dependency_resolver(dependencies, ["map Z"], base_inputs)


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


def test_get_value_thresholds_value_threshold_single_component():
    """value_threshold with one component returns length-1 array."""
    maps = np.array([[1.0], [2.0], [3.0]])
    result = get_value_thresholds(maps=maps, value_threshold=2.5)
    assert result.shape == (1,)
    assert result.dtype == np.float64
    np.testing.assert_array_almost_equal(result, [2.5])


def test_get_value_thresholds_value_threshold_multiple_components():
    """value_threshold is broadcast to one value per component."""
    maps = np.array([[1, 2], [3, 4], [5, 6]])
    result = get_value_thresholds(maps=maps, value_threshold=3.0)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, [3.0, 3.0])


def test_get_value_thresholds_value_threshold_zero():
    """value_threshold can be zero."""
    maps = np.array([[1, 2], [3, 4]])
    result = get_value_thresholds(maps=maps, value_threshold=0.0)
    np.testing.assert_array_almost_equal(result, [0.0, 0.0])


def test_get_value_thresholds_proportion_threshold_50_per_column():
    """proportion_threshold computes percentile per column (axis=0)."""
    # Columns have different values; 50th percentile of col0 is 3, col1 is 4
    maps = np.array([[1, 2], [3, 4], [5, 6]])
    result = get_value_thresholds(maps=maps, proportion_threshold=50)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, [3.0, 4.0])


def test_get_value_thresholds_proportion_threshold_uses_absolute_values():
    """proportion_threshold is applied to absolute values of maps."""
    maps = np.array([[-4, 2], [-2, 4], [-6, 6]])  # col0: 4,2,6 → 50th = 4
    result = get_value_thresholds(maps=maps, proportion_threshold=50)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, [4.0, 4.0])


def test_get_value_thresholds_proportion_threshold_0_min():
    """proportion_threshold=0 gives minimum absolute value per column."""
    maps = np.array([[3, 1], [5, 2], [4, 3]])
    result = get_value_thresholds(maps=maps, proportion_threshold=0)
    np.testing.assert_array_almost_equal(result, [3.0, 1.0])


def test_get_value_thresholds_proportion_threshold_100_max():
    """proportion_threshold=100 gives maximum absolute value per column."""
    maps = np.array([[3, 1], [5, 2], [4, 3]])
    result = get_value_thresholds(maps=maps, proportion_threshold=100)
    np.testing.assert_array_almost_equal(result, [5.0, 3.0])


def test_get_value_thresholds_proportion_threshold_single_component():
    """proportion_threshold with one component returns length-1 array."""
    maps = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = get_value_thresholds(maps=maps, proportion_threshold=50)
    assert result.shape == (1,)
    # 50th percentile of [1,2,3,4,5] with method 'higher' is 3
    np.testing.assert_array_almost_equal(result, [3.0])


def test_get_value_thresholds_neither_threshold_raises():
    """Providing neither value_threshold nor proportion_threshold raises."""
    maps = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Only one of value_threshold or proportion_threshold"):
        get_value_thresholds(maps=maps)


def test_get_value_thresholds_both_thresholds_raise():
    """Providing both value_threshold and proportion_threshold raises."""
    maps = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Only one of value_threshold or proportion_threshold"):
        get_value_thresholds(
            maps=maps,
            value_threshold=1.0,
            proportion_threshold=50,
        )


def test_get_value_thresholds_proportion_threshold_above_100_raises():
    """proportion_threshold > 100 raises."""
    maps = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="proportion_threshold must be between 0 and 100"):
        get_value_thresholds(maps=maps, proportion_threshold=101)


def test_get_value_thresholds_proportion_threshold_below_0_raises():
    """proportion_threshold < 0 raises."""
    maps = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="proportion_threshold must be between 0 and 100"):
        get_value_thresholds(maps=maps, proportion_threshold=-0.1)


def test_get_value_thresholds_proportion_threshold_boundary_0_and_100_valid():
    """proportion_threshold 0 and 100 are valid (inclusive)."""
    maps = np.array([[1, 2], [3, 4]])
    r0 = get_value_thresholds(maps=maps, proportion_threshold=0)
    r100 = get_value_thresholds(maps=maps, proportion_threshold=100)
    assert r0.shape == r100.shape == (2,)
    np.testing.assert_array_almost_equal(r0, [1.0, 2.0])
    np.testing.assert_array_almost_equal(r100, [3.0, 4.0])


def test_get_value_thresholds_keyword_only_args():
    """Maps must be passed as keyword (function uses *)."""
    maps = np.array([[1, 2], [3, 4]])
    # Should work with keyword
    result = get_value_thresholds(maps=maps, value_threshold=1.0)
    assert result.shape == (2,)
    # Positional maps would raise TypeError
    with pytest.raises(TypeError):
        get_value_thresholds(maps, value_threshold=1.0)


def test_get_value_thresholds_return_type_ndarray():
    """Return value is a numpy ndarray."""
    maps = np.array([[1, 2], [3, 4]])
    result_val = get_value_thresholds(maps=maps, value_threshold=1.0)
    result_prop = get_value_thresholds(maps=maps, proportion_threshold=50)
    assert isinstance(result_val, np.ndarray)
    assert isinstance(result_prop, np.ndarray)
