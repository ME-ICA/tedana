"""ICA and related signal decomposition methods for tedana"""
import logging
import warnings

import numpy as np
from robustica import RobustICA
from scipy import stats
from sklearn.decomposition import FastICA

from tedana.config import (
    DEFAULT_ICA_METHOD,
    DEFAULT_N_MAX_ITER,
    DEFAULT_N_MAX_RESTART,
    DEFAULT_N_ROBUST_RUNS,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def tedica(
    data,
    n_components,
    fixed_seed,
    ica_method=DEFAULT_ICA_METHOD,
    n_robust_runs=DEFAULT_N_ROBUST_RUNS,
    maxit=DEFAULT_N_MAX_ITER,
    maxrestart=DEFAULT_N_MAX_RESTART,
):
    """
    Perform ICA on `data` with the user selected ica method and returns mixing matrix

    Parameters
    ----------
    data : (S x T) :obj:`numpy.ndarray`
        Dimensionally reduced optimally combined functional data, where `S` is
        samples and `T` is time
    n_components : :obj:`int`
        Number of components retained from PCA decomposition.
    fixed_seed : :obj:`int`
        Seed for ensuring reproducibility of ICA results.
    ica_method : :obj: `str'
        slected ICA method, can be fastica or robutica.
    n_robust_runs : :obj: `int'
        selected number of robust runs when robustica is used. Default is 30.
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

    """
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
    RepLGR.info(
        "Independent component analysis was then used to "
        "decompose the dimensionally reduced dataset."
    )

    ica_method = ica_method.lower()

    if ica_method == "robustica":
        mmix, fixed_seed = r_ica(
            data,
            n_components=n_components,
            fixed_seed=fixed_seed,
            n_robust_runs=n_robust_runs,
            max_it=maxit,
        )
    elif ica_method == "fastica":
        mmix, fixed_seed = f_ica(
            data,
            n_components=n_components,
            fixed_seed=fixed_seed,
            maxit=maxit,
            maxrestart=maxrestart,
        )
    else:
        raise ValueError("The selected ICA method is invalid!")

    return mmix, fixed_seed


def r_ica(data, n_components, fixed_seed, n_robust_runs, max_it):
    """
    Perform robustica on `data` by running FastICA multiple times (n_robust runes)
    and returns mixing matrix

    Parameters
    ----------
    data : (S x T) :obj:`numpy.ndarray`
        Dimensionally reduced optimally combined functional data, where `S` is
        samples and `T` is time
    n_components : :obj:`int`
        Number of components retained from PCA decomposition.
    fixed_seed : :obj:`int`
        Seed for ensuring reproducibility of ICA results.
    n_robust_runs : :obj: `int'
        selected number of robust runs when robustica is used. Default is 30.
    maxit : :obj:`int`, optional
        Maximum number of iterations for ICA. Default is 500.

    Returns
    -------
    mmix : (T x C) :obj:`numpy.ndarray`
        Z-scored mixing matrix for converting input data to component space,
        where `C` is components and `T` is the same as in `data`
    fixed_seed : :obj:`int`
        Random seed from final decomposition.

    """
    if n_robust_runs > 200:
        LGR.warning(
            "The selected n_robust_runs is a very big number! The process will take a long time!"
        )

    RepLGR.info("RobustICA package was used for ICA decomposition \\citep{Anglada2022}.")

    if fixed_seed == -1:
        fixed_seed = np.random.randint(low=1, high=1000)

    try:
        rica = RobustICA(
            n_components=n_components,
            robust_runs=n_robust_runs,
            whiten="arbitrary-variance",
            max_iter=max_it,
            random_state=fixed_seed,
            robust_dimreduce=False,
            fun="logcosh",
            robust_method="DBSCAN",
        )

        S, mmix = rica.fit_transform(data)
        q = rica.evaluate_clustering(
            rica.S_all, rica.clustering.labels_, rica.signs_, rica.orientation_
        )

    except:
        rica = RobustICA(
            n_components=n_components,
            robust_runs=n_robust_runs,
            whiten="arbitrary-variance",
            max_iter=max_it,
            random_state=fixed_seed,
            robust_dimreduce=False,
            fun="logcosh",
            robust_method="AgglomerativeClustering",
        )

        S, mmix = rica.fit_transform(data)
        q = rica.evaluate_clustering(
            rica.S_all, rica.clustering.labels_, rica.signs_, rica.orientation_
        )

    iq = np.array(np.mean(q[q["cluster_id"] >= 0].iq))  # The cluster labeled -1 is noise

    if iq < 0.6:
        LGR.warning(
            "The resultant mean Index Quality is low. It  is recommended to rerun the "
            "process with a different seed."
        )

    mmix = mmix[:, q["cluster_id"] >= 0]
    mmix = stats.zscore(mmix, axis=0)

    LGR.info(
        "RobustICA with {0} robust runs and seed {1} was used. "
        "The mean Index Quality is {2}".format(n_robust_runs, fixed_seed, iq)
    )
    return mmix, fixed_seed


def f_ica(data, n_components, fixed_seed, maxit, maxrestart):
    """
    Perform FastICA on `data` and returns mixing matrix

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
                    "ICA with random seed {0} failed to converge after {1} "
                    "iterations".format(fixed_seed, ica.n_iter_)
                )
                if i_attempt < maxrestart - 1:
                    fixed_seed += 1
                    LGR.warning("Random seed updated to {0}".format(fixed_seed))
            else:
                LGR.info(
                    "ICA with random seed {0} converged in {1} "
                    "iterations".format(fixed_seed, ica.n_iter_)
                )
                break

    mmix = ica.mixing_
    mmix = stats.zscore(mmix, axis=0)
    return mmix, fixed_seed