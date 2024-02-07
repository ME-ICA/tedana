"""Tests for tedana.reporting."""

import numpy as np

from tedana import reporting


def test_smoke_trim_edge_zeros():
    """Ensures that trim_edge_zeros works with random inputs."""
    arr = np.random.random((100, 100))
    assert reporting.static_figures._trim_edge_zeros(arr) is not None


# TODO: Test other functions in reporting?
