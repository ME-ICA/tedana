"""Tests for tedana.metrics.frequency."""

import numpy as np
import pytest

from tedana.metrics.frequency import calculate_hfc


class TestCalculateHfc:
    """Tests for calculate_hfc."""

    def test_output_shape(self):
        """calculate_hfc returns a 1-D array with length equal to n_components."""
        rng = np.random.default_rng(0)
        n_vols, n_components = 200, 5
        mixing = rng.standard_normal((n_vols, n_components))
        hfc = calculate_hfc(mixing=mixing, tr=2.0)
        assert hfc.shape == (n_components,)

    def test_output_range(self):
        """HFC values are in [0, 1]."""
        rng = np.random.default_rng(1)
        mixing = rng.standard_normal((300, 10))
        hfc = calculate_hfc(mixing=mixing, tr=2.0)
        assert np.all(hfc >= 0)
        assert np.all(hfc <= 1)

    def test_high_freq_higher_than_low_freq(self):
        """A pure high-frequency component has higher HFC than a pure low-frequency one."""
        n_vols = 300
        tr = 2.0
        t = np.arange(n_vols) * tr

        low_freq_component = np.sin(2 * np.pi * 0.02 * t)  # 0.02 Hz, well below Nyquist
        high_freq_component = np.sin(2 * np.pi * 0.20 * t)  # 0.20 Hz, near Nyquist=0.25

        mixing = np.column_stack([low_freq_component, high_freq_component])
        hfc = calculate_hfc(mixing=mixing, tr=tr)

        assert hfc[1] > hfc[0], (
            f"High-frequency component HFC ({hfc[1]:.3f}) should exceed "
            f"low-frequency component HFC ({hfc[0]:.3f})"
        )

    def test_single_component(self):
        """calculate_hfc works when there is only one component."""
        rng = np.random.default_rng(2)
        mixing = rng.standard_normal((100, 1))
        hfc = calculate_hfc(mixing=mixing, tr=1.5)
        assert hfc.shape == (1,)
        assert 0 <= hfc[0] <= 1

    def test_different_tr_values(self):
        """HFC is computed without error for various tr values."""
        rng = np.random.default_rng(3)
        mixing = rng.standard_normal((150, 4))
        for tr in [0.5, 1.0, 2.0, 3.0]:
            hfc = calculate_hfc(mixing=mixing, tr=tr)
            assert hfc.shape == (4,)
            assert np.all(hfc >= 0)
            assert np.all(hfc <= 1)

    def test_different_f_hp_values(self):
        """Different f_hp values do not break the function."""
        rng = np.random.default_rng(4)
        mixing = rng.standard_normal((200, 3))
        for f_hp in [0.005, 0.01, 0.05]:
            hfc = calculate_hfc(mixing=mixing, tr=2.0, f_hp=f_hp)
            assert hfc.shape == (3,)
            assert np.all(hfc >= 0)
            assert np.all(hfc <= 1)

    def test_keyword_only_args(self):
        """calculate_hfc requires keyword arguments."""
        rng = np.random.default_rng(5)
        mixing = rng.standard_normal((100, 2))
        with pytest.raises(TypeError):
            calculate_hfc(mixing, 2.0)  # positional args not allowed
