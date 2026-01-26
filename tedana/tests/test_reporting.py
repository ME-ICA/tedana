"""Tests for tedana.reporting."""

import numpy as np
import pandas as pd
import pytest

from tedana import reporting
from tedana.tests.test_external_metrics import sample_mixing_matrix
from tedana.tests.test_selection_utils import sample_selector


def test_smoke_trim_edge_zeros():
    """Ensures that trim_edge_zeros works with random inputs."""
    arr = np.random.random((100, 100))
    assert reporting.static_figures._trim_edge_zeros(arr) is not None


def test_calculate_rejected_components_impact():
    selector = sample_selector()
    mixing = sample_mixing_matrix()

    component_table = selector.component_table_

    reporting.calculate_rejected_components_impact(selector, mixing)

    # Total matches expecting value in testing data
    assert np.round(
        selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"], 4
    ) == np.round(4.0927834, 4)

    assert "Var Exp of rejected to accepted" in selector.component_table_.columns
    rej = component_table[component_table["classification"] == "rejected"]
    acc = component_table[component_table["classification"] == "accepted"]

    assert rej["Var Exp of rejected to accepted"].isna().all()
    assert not acc["Var Exp of rejected to accepted"].isna().all()


def test_calculate_rejected_components_impact_no_rej():
    selector = sample_selector()
    mixing = sample_mixing_matrix()

    component_table = selector.component_table_

    component_table.drop(
        component_table[component_table["classification"] == "rejected"].index, inplace=True
    )

    reporting.calculate_rejected_components_impact(selector, mixing)

    assert "Var Exp of rejected to accepted" in selector.component_table_.columns
    assert selector.component_table_["Var Exp of rejected to accepted"].isna().all()
    assert np.isnan(
        selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"]
    )


def test_calculate_rejected_components_impact_no_acc():
    selector = sample_selector()
    mixing = sample_mixing_matrix()

    component_table = selector.component_table_

    component_table.drop(
        component_table[component_table["classification"] == "accepted"].index, inplace=True
    )

    reporting.calculate_rejected_components_impact(selector, mixing)

    assert "Var Exp of rejected to accepted" in selector.component_table_.columns
    assert selector.component_table_["Var Exp of rejected to accepted"].isna().all()
    assert np.isnan(
        selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"]
    )


def test_plot_heatmap_nonfinite_distances_warns_and_succeeds(tmp_path):
    """plot_heatmap should not crash when correlation-derived distances are non-finite.

    This can happen when an external regressor has zero variance (constant time series),
    which yields NaNs in the correlation computations used for hierarchical clustering.
    """

    # Construct a correlation matrix with one regressor that has a constant
    # correlation pattern across components, which yields zero variance and
    # can produce non-finite correlations when clustering regressors.
    correlation_df = pd.DataFrame(
        {
            "ICA_00": [0.5, 0.1],
            "ICA_01": [0.5, -0.2],
        },
        index=["constant", "varying"],
    )

    # component_table needs at least one model column that matches the regex
    # used in plot_heatmap: "(R2stat .* model)".
    component_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01"],
            "R2stat demo model": [0.1, 0.9],
        },
        index=[0, 1],
    )

    out_file = tmp_path / "heatmap.png"

    with pytest.warns(UserWarning, match=r"Non-finite correlations detected.*constant"):
        reporting.static_figures.plot_heatmap(
            correlation_df=correlation_df,
            component_table=component_table,
            out_file=str(out_file),
        )

    assert out_file.exists()
