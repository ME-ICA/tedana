"""Tests for tedana.metrics.collect."""

import pandas as pd

from tedana.metrics import collect


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
            "d_table_score_scrub": [4.0, 2.0],
            "kappa proportion": [0.5, 0.4],
            "marginal R-squared": [0.3, 0.25],
            "partial R-squared": [0.2, 0.15],
            "semi-partial R-squared": [0.1, 0.05],
            "varex kappa ratio": [0.5, 0.4],
            "Var Exp of rejected to accepted": [0.1, 0.05],
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
    assert "kappa proportion" in metadata
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
