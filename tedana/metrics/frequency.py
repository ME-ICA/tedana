"""Frequency-related metrics."""

import numpy as np
import pandas as pd


def feature_frequency(mixing_fft: np.ndarray, TR: float, f_hp: float = 0.01):
    """Extract the high-frequency content feature scores.

    This function determines the frequency, as fraction of the Nyquist
    frequency, at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    Parameters
    ----------
    mixing_fft : numpy.ndarray of shape (F, C)
        Stored array is (frequency x component), with frequencies
        ranging from 0 Hz to Nyquist frequency.
    TR : float
        TR (in seconds) of the fMRI data
    f_hp: float, optional
        High-pass cutoff frequency in spectrum computations.

    Returns
    -------
    HFC : array_like
        Array of the HFC ('High-frequency content') feature scores
        for the components of the melodic_FTmix file
    metric_metadata : None or dict
        If the ``metric_metadata`` input was None, then None will be returned.
        Otherwise, this will be a dictionary containing existing information,
        as well as new metadata for the ``HFC`` metric.
    """
    metric_metadata = {
        'HFC': {
            'LongName': 'High-frequency content',
            'Description': (
                'The proportion of the power spectrum for each component that falls above '
                f'{f_hp} Hz.'
            ),
            'Units': 'arbitrary',
        },
    }

    # Determine sample frequency
    Fs = 1 / TR

    # Determine Nyquist-frequency
    Ny = Fs / 2

    n_frequencies = mixing_fft.shape[0]

    # Determine which frequencies are associated with every row in the
    # melodic_FTmix file (assuming the rows range from 0Hz to Nyquist)
    frequencies = Ny * np.arange(1, n_frequencies + 1) / n_frequencies

    # Only include frequencies higher than f_hp Hz
    included_freqs_idx = np.squeeze(np.array(np.where(frequencies > f_hp)))
    mixing_fft = mixing_fft[included_freqs_idx, :]
    frequencies = frequencies[included_freqs_idx]

    # Set frequency range to [0-1]
    frequencies_normalized = (frequencies - f_hp) / (Ny - f_hp)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = np.cumsum(mixing_fft, axis=0) / np.sum(mixing_fft, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
    cutoff_idx = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

    # Now get the fractions associated with those indices index, these are the final feature scores
    high_frequency_content = frequencies_normalized[cutoff_idx]
    metric_df = pd.DataFrame(data=high_frequency_content, columns=['HFC'])

    return metric_df, metric_metadata
