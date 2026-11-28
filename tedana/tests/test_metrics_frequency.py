"""Tests for tedana.metrics.frequency."""

import numpy as np
import pytest

from tedana.metrics.frequency import calculate_hfc, calculate_phf


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
        """A high-frequency component has higher HFC than a low-frequency one."""
        n_vols = 300
        tr = 2.0
        low_freq_fft = np.zeros(n_vols // 2 + 1, dtype=complex)
        high_freq_fft = np.zeros(n_vols // 2 + 1, dtype=complex)
        low_freq_fft[10:14] = 1
        high_freq_fft[110:114] = 1
        low_freq_component = np.fft.irfft(low_freq_fft, n=n_vols)
        high_freq_component = np.fft.irfft(high_freq_fft, n=n_vols)

        mixing = np.column_stack([low_freq_component, high_freq_component])
        hfc = calculate_hfc(mixing=mixing, tr=tr)

        assert hfc[1] > hfc[0], (
            f"High-frequency component HFC ({hfc[1]:.3f}) should exceed "
            f"low-frequency component HFC ({hfc[0]:.3f})"
        )

    def test_uses_power_spectrum(self):
        """HFC is based on FFT power, not magnitude."""
        n_vols = 16
        tr = 1.0
        fft_vals = np.zeros(n_vols // 2 + 1, dtype=complex)
        fft_vals[1:5] = [1, 1, 1, 2]
        mixing = np.fft.irfft(fft_vals, n=n_vols)[:, np.newaxis]

        hfc = calculate_hfc(mixing=mixing, tr=tr, f_hp=0)

        # With power weights [1, 1, 1, 4], the cumulative fraction closest
        # to 0.5 is at the third non-DC bin. A magnitude spectrum would pick
        # the second bin instead.
        frequencies = np.fft.rfftfreq(n_vols, d=tr)[1:]
        expected = frequencies[2] / (1 / (2 * tr))
        np.testing.assert_allclose(hfc, [expected])

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


def _band_limited(n_vols, tr, f_lo, f_hi):
    """Build a time series with equal power in every FFT bin between f_lo and f_hi Hz."""
    freqs = np.fft.rfftfreq(n_vols, d=tr)
    spectrum = np.zeros(freqs.size, dtype=complex)
    spectrum[(freqs >= f_lo) & (freqs <= f_hi)] = 1
    return np.fft.irfft(spectrum, n=n_vols)


class TestCalculatePhf:
    """Tests for calculate_phf."""

    def test_output_shape_and_range(self):
        """calculate_phf returns one value per component, each in [0, 1]."""
        rng = np.random.default_rng(0)
        mixing = rng.standard_normal((200, 5))
        phf = calculate_phf(mixing=mixing, tr=2.0)
        assert phf.shape == (5,)
        assert np.all(phf >= 0)
        assert np.all(phf <= 1)

    def test_exact_fraction(self):
        """PHF equals the share of power above the cutoff, ignoring power at or below f_hp."""
        n_vols, tr = 200, 1.0  # bins every 0.005 Hz
        freqs = np.fft.rfftfreq(n_vols, d=tr)
        spectrum = np.zeros(freqs.size, dtype=complex)
        spectrum[np.isclose(freqs, 0.005)] = 10  # at or below f_hp: excluded
        spectrum[np.isclose(freqs, 0.05)] = 1  # power 1, below the cutoff
        spectrum[np.isclose(freqs, 0.2)] = np.sqrt(3)  # power 3, above the cutoff
        mixing = np.fft.irfft(spectrum, n=n_vols)[:, np.newaxis]

        phf = calculate_phf(mixing=mixing, tr=tr, f_cutoff=0.1, f_hp=0.01)
        np.testing.assert_allclose(phf, [0.75])

    def test_nyquist_bin_weighting(self):
        """Equal-variance signals below the cutoff and at Nyquist each get half the power."""
        n_vols, tr = 200, 1.0
        t = np.arange(n_vols)
        low = np.sqrt(2) * np.cos(2 * np.pi * 0.05 * t)  # variance 1
        nyquist = np.cos(np.pi * t)  # alternating +/-1, variance 1
        phf = calculate_phf(mixing=(low + nyquist)[:, np.newaxis], tr=tr)
        np.testing.assert_allclose(phf, [0.5])

    def test_low_and_high_frequency_components(self):
        """Power entirely below or above the cutoff gives PHF of 0 or 1."""
        n_vols, tr = 300, 1.0
        low = _band_limited(n_vols, tr, 0.02, 0.08)
        high = _band_limited(n_vols, tr, 0.2, 0.4)
        phf = calculate_phf(mixing=np.column_stack([low, high]), tr=tr)
        np.testing.assert_allclose(phf, [0.0, 1.0], atol=1e-10)

    def test_cutoff_is_fixed_in_hz_across_trs(self):
        """A 0.2-0.25 Hz signal is high-frequency at every TR, unlike with HFC > 0.35."""
        scan_length = 600  # seconds
        for tr in [0.5, 1.0, 1.5, 2.0]:
            n_vols = int(scan_length / tr)
            mixing = _band_limited(n_vols, tr, 0.2, 0.25)[:, np.newaxis]
            phf = calculate_phf(mixing=mixing, tr=tr)
            np.testing.assert_allclose(phf, [1.0], atol=1e-10)

        # HFC uses a Nyquist-relative scale, so at TR 0.5 s the same signal scores
        # below the 0.35 cutoff used in the ICA-AROMA demo tree.
        tr = 0.5
        mixing = _band_limited(int(scan_length / tr), tr, 0.2, 0.25)[:, np.newaxis]
        assert calculate_hfc(mixing=mixing, tr=tr)[0] < 0.35

    def test_white_noise_matches_bandwidth(self):
        """For white noise, PHF approaches the share of the retained band above the cutoff."""
        rng = np.random.default_rng(1)
        tr = 1.0
        mixing = rng.standard_normal((20000, 3))
        phf = calculate_phf(mixing=mixing, tr=tr)
        expected = (0.5 - 0.1) / (0.5 - 0.01)
        np.testing.assert_allclose(phf, expected, atol=0.02)

    def test_custom_cutoff(self):
        """A higher cutoff gives a lower or equal PHF."""
        rng = np.random.default_rng(2)
        mixing = rng.standard_normal((300, 4))
        phf_low = calculate_phf(mixing=mixing, tr=1.0, f_cutoff=0.1)
        phf_high = calculate_phf(mixing=mixing, tr=1.0, f_cutoff=0.2)
        assert np.all(phf_high <= phf_low)

    def test_nyquist_below_cutoff_returns_nan(self, caplog):
        """PHF is NaN, with a warning, when the Nyquist frequency is at or below the cutoff."""
        rng = np.random.default_rng(3)
        mixing = rng.standard_normal((100, 2))
        phf = calculate_phf(mixing=mixing, tr=5.0)  # Nyquist = 0.1 Hz
        assert phf.shape == (2,)
        assert np.all(np.isnan(phf))
        assert "cannot be calculated" in caplog.text

    def test_aliasing_warning(self, caplog):
        """A warning is logged when the Nyquist frequency is under twice the cutoff."""
        rng = np.random.default_rng(4)
        mixing = rng.standard_normal((100, 2))
        phf = calculate_phf(mixing=mixing, tr=3.0)  # Nyquist ~0.167 Hz
        assert np.all(np.isfinite(phf))
        assert "alias" in caplog.text

    def test_invalid_cutoff(self):
        """f_cutoff must be above f_hp."""
        mixing = np.random.default_rng(5).standard_normal((100, 2))
        with pytest.raises(ValueError, match="must be greater than f_hp"):
            calculate_phf(mixing=mixing, tr=1.0, f_cutoff=0.01, f_hp=0.01)

    def test_keyword_only_args(self):
        """calculate_phf requires keyword arguments."""
        mixing = np.random.default_rng(6).standard_normal((100, 2))
        with pytest.raises(TypeError):
            calculate_phf(mixing, 2.0)  # positional args not allowed
