"""Frequency-related metrics."""

import logging

import numpy as np

LGR = logging.getLogger("GENERAL")


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

    # One-sided power spectrum; skip the DC bin (index 0)
    fft_vals = np.fft.rfft(mixing, axis=0)
    mixing_fft = np.abs(fft_vals[1:, :]) ** 2  # (F, C)

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


def calculate_phf(
    *,
    mixing: np.ndarray,
    tr: float,
    f_cutoff: float = 0.1,
    f_hp: float = 0.01,
) -> np.ndarray:
    """Calculate the high-frequency power fraction (PHF) for each component.

    PHF is the fraction of each component's power above ``f_hp`` Hz that lies
    above ``f_cutoff`` Hz.
    Unlike :func:`calculate_hfc`, whose scale is a fraction of the Nyquist
    frequency, the cutoff here is a fixed frequency in Hz, so a given PHF
    threshold means the same thing at every repetition time.

    Parameters
    ----------
    mixing : (T x C) array_like
        Mixing matrix where T is time points and C is components.
    tr : float
        Repetition time of the fMRI data in seconds.
    f_cutoff : float, optional
        Frequency, in Hz, above which power counts as high-frequency.
        Default is 0.1 Hz.
    f_hp : float, optional
        High-pass cutoff frequency in Hz.  Frequencies at or below this value
        are excluded before computing the fraction.  Default is 0.01 Hz.

    Returns
    -------
    phf : (C,) numpy.ndarray
        Fraction of power above ``f_cutoff``, from 0 (no power above
        ``f_cutoff``) to 1 (all power above ``f_cutoff``).
        All values are NaN if the Nyquist frequency is at or below ``f_cutoff``.

    Notes
    -----
    ``PHF > 0.5`` is equivalent to the component's median frequency (the
    frequency below which half of its power above ``f_hp`` lies) being above
    ``f_cutoff``.

    Cardiac and respiratory signals are not band-limited before sampling, so they
    can alias below ``f_cutoff``, where no spectral metric can distinguish them
    from BOLD fluctuations.
    This becomes common as the Nyquist frequency approaches ``f_cutoff``
    (for example, TR > 2.5 s for a 0.1 Hz cutoff).
    """
    if f_cutoff <= f_hp:
        raise ValueError(f"f_cutoff ({f_cutoff}) must be greater than f_hp ({f_hp}).")

    mixing = np.asarray(mixing, dtype=float)
    if mixing.ndim == 1:
        mixing = mixing[:, np.newaxis]

    nyquist_freq = 1.0 / (2.0 * tr)
    if nyquist_freq <= f_cutoff:
        LGR.warning(
            f"The Nyquist frequency ({nyquist_freq:.3f} Hz) is at or below the PHF cutoff "
            f"({f_cutoff} Hz), so PHF cannot be calculated. Returning NaN."
        )
        return np.full(mixing.shape[1], np.nan)

    if nyquist_freq < 2 * f_cutoff:
        LGR.warning(
            f"The Nyquist frequency ({nyquist_freq:.3f} Hz) is less than twice the PHF cutoff "
            f"({f_cutoff} Hz). Cardiac and respiratory noise are likely to alias below the "
            "cutoff, where PHF cannot detect them."
        )

    # One-sided power spectrum, skipping the DC bin
    n_vols = mixing.shape[0]
    power = np.abs(np.fft.rfft(mixing, axis=0)[1:, :]) ** 2
    frequencies = np.fft.rfftfreq(n_vols, d=tr)[1:]
    if n_vols % 2 == 0:
        # Every other bin pools a positive and a negative frequency, but the Nyquist
        # bin does not, so halve it to give each bin the same weight per unit variance.
        power[-1, :] /= 2

    retained = frequencies > f_hp
    total_power = power[retained, :].sum(axis=0)
    high_power = power[frequencies > f_cutoff, :].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        phf = high_power / total_power

    return phf
