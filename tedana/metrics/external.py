"""Metrics unrelated to TE-(in)dependence."""

import numpy as np


def correlate_regressor(external_regressor, mixing):
    """Correlate external regressors with mixing components.

    Parameters
    ----------
    external_regressor : array, shape (n_samples)
        External regressor. The regressor will be correlated with each component's time series.
    mixing : array, shape (n_samples, n_components)
        Mixing matrix from ICA.

    Returns
    -------
    corrs : array, shape (n_components)
        Absolute correlations between external regressor and mixing components.
    """
    assert external_regressor.ndim == 1
    assert external_regressor.shape[0] == mixing.shape[0]
    corrs = np.abs(np.corrcoef(external_regressor, mixing.T)[0, 1:])
    return corrs
