"""Tests for tedana.metrics._utils."""

import numpy as np
import pytest

from tedana.metrics._utils import (
    add_external_dependencies,
    dependency_resolver,
    determine_signs,
    flip_components,
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
