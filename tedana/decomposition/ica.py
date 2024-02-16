"""ICA and related signal decomposition methods for tedana."""

import logging
import warnings

import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def tedica(data, n_components, fixed_seed, maxit=500, maxrestart=10):
    """Perform ICA on ``data`` and return mixing matrix.

    Parameters
    ----------
    data : (S x T) :obj:`numpy.ndarray`
        Dimensionally reduced optimally combined functional data, where `S` is
        samples and `T` is time
    n_components : :obj:`int`
        Number of components retained from PCA decomposition
    fixed_seed : :obj:`int`
        Seed for ensuring reproducibility of ICA results
    maxit : :obj:`int`, optional
        Maximum number of iterations for ICA. Default is 500.
    maxrestart : :obj:`int`, optional
        Maximum number of attempted decompositions to perform with different
        random seeds. ICA will stop running if there is convergence prior to
        reaching this limit. Default is 10.

    Returns
    -------
    mmix : (T x C) :obj:`numpy.ndarray`
        Z-scored mixing matrix for converting input data to component space,
        where `C` is components and `T` is the same as in `data`
    fixed_seed : :obj:`int`
        Random seed from final decomposition.

    Notes
    -----
    Uses `sklearn` implementation of FastICA for decomposition
    """
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
    RepLGR.info(
        "Independent component analysis was then used to "
        "decompose the dimensionally reduced dataset."
    )

    if fixed_seed == -1:
        fixed_seed = np.random.randint(low=1, high=1000)

    for i_attempt in range(maxrestart):
        ica = FastICA(
            n_components=n_components,
            algorithm="parallel",
            fun="logcosh",
            max_iter=maxit,
            random_state=fixed_seed,
        )

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered in order to capture
            # convergence failures.
            warnings.simplefilter("always")

            ica.fit(data)

            w = list(filter(lambda i: issubclass(i.category, UserWarning), w))
            if len(w):
                LGR.warning(
                    f"ICA with random seed {fixed_seed} failed to converge after {ica.n_iter_} "
                    "iterations"
                )
                if i_attempt < maxrestart - 1:
                    fixed_seed += 1
                    LGR.warning(f"Random seed updated to {fixed_seed}")
            else:
                LGR.info(
                    f"ICA with random seed {fixed_seed} converged in {ica.n_iter_} iterations"
                )
                break

    mmix = ica.mixing_
    mmix = stats.zscore(mmix, axis=0)
    return mmix, fixed_seed
