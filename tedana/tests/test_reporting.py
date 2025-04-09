"""Tests for tedana.reporting."""

import numpy as np

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

    assert selector.cross_component_metrics_["rejected_components_impact"] <= 1
    assert "R2 of fit of rejected to accepted" in selector.component_table_.columns

    rej = component_table[component_table["classification"] == "rejected"]
    acc = component_table[component_table["classification"] == "accepted"]
    
    assert rej['R2 of fit of rejected to accepted'].isna().all()
    assert not acc['R2 of fit of rejected to accepted'].isna().all()


# TODO: Test other functions in reporting?
