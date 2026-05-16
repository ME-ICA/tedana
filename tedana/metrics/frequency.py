"""Frequency-related metrics."""

import numpy as np


def calculate_hfc(*, mixing: np.ndarray, tr: float, f_hp: float = 0.01) -> np.ndarray:
    """Calculate the high-frequency content (HFC) score for each component.

    Determines the normalized frequency at which the lower and higher
    frequencies of each component's power spectrum contribute equally to the
    total power above ``f_hp`` Hz.  Values near 1 indicate that most power
    sits close to the Nyquist frequency (likely noise); values near 0 indicate
    low-frequency, BOLD-like components.

    Parameters
    ----------
    mixing : (T x C) array_like
        ICA mixing matrix where T is time points and C is components.
    tr : float
        Repetition time of the fMRI data in seconds.
    f_hp : float, optional
        High-pass cutoff frequency in Hz.  Frequencies at or below this value
        are excluded before computing the score.  Default is 0.01 Hz.

    Returns
    -------
    hfc : (C,) numpy.ndarray
        High-frequency content score for each component, on a scale from 0
        (all power at ``f_hp``) to 1 (all power at Nyquist).
    """
    mixing = np.asarray(mixing, dtype=float)

    # One-sided magnitude spectrum; skip the DC bin (index 0)
    fft_vals = np.fft.rfft(mixing, axis=0)
    mixing_fft = np.abs(fft_vals[1:, :])  # (F, C)

    sampling_rate = 1.0 / tr
    nyquist_freq = sampling_rate / 2.0

    # Use the actual rFFT bin centers so odd-length series do not incorrectly
    # force the highest non-DC bin to the Nyquist frequency.
    frequencies = np.fft.rfftfreq(mixing.shape[0], d=tr)[1:]

    # Restrict to frequencies above the high-pass cutoff
    included = np.where(frequencies > f_hp)[0]
    mixing_fft = mixing_fft[included, :]
    frequencies = frequencies[included]

    # Normalize frequency axis to [0, 1] within the retained band
    frequencies_normalized = (frequencies - f_hp) / (nyquist_freq - f_hp)

    # Cumulative power fraction across frequency for each component
    fcumsum_fract = np.cumsum(mixing_fft, axis=0) / np.sum(mixing_fft, axis=0)

    # Frequency at which cumulative power reaches 50 %
    cutoff_idx = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)
    hfc = frequencies_normalized[cutoff_idx]
    return hfc
