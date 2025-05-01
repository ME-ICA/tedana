"""ICA and related signal decomposition methods for tedana."""

import logging
import warnings

import numpy as np
from robustica import RobustICA, abs_pearson_dist
from scipy import stats
from sklearn import manifold
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from tedana.config import (
    DEFAULT_ICA_METHOD,
    DEFAULT_N_MAX_ITER,
    DEFAULT_N_MAX_RESTART,
    DEFAULT_N_ROBUST_RUNS,
    WARN_IQ,
    WARN_N_ROBUST_RUNS,
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
    """Perform ICA on `data` with the user selected ica method and returns mixing matrix.

    Parameters
    ----------
    data : (S x T) :obj:`numpy.ndarray`
        Dimensionally reduced optimally combined functional data, where `S` is
        samples and `T` is time
    n_components : :obj:`int`
        Number of components retained from PCA decomposition.
    fixed_seed : :obj:`int`
        Seed for ensuring reproducibility of ICA results.
    ica_method : :obj: `str`
        selected ICA method, can be fastica or robustica.
    n_robust_runs : :obj: `int`
        selected number of robust runs when robustica is used. Default is 30.
    maxit : :obj:`int`, optional
        Maximum number of iterations for ICA. Default is 500.
    maxrestart : :obj:`int`, optional
        Maximum number of attempted decompositions to perform with different
        random seeds. ICA will stop running if there is convergence prior to
        reaching this limit. Default is 10.

    Returns
    -------
    mixing : (T x C) :obj:`numpy.ndarray`
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

    # Default r_ica results to None to avoid errors in the case of fastica
    c_labels, similarity_t_sne = None, None
    index_quality, fastica_convergence_warning_count = None, None

    if ica_method == "robustica":
        (
            mixing,
            fixed_seed,
            c_labels,
            similarity_t_sne,
            fastica_convergence_warning_count,
            index_quality,
        ) = r_ica(
            data,
            n_components=n_components,
            fixed_seed=fixed_seed,
            n_robust_runs=n_robust_runs,
            max_it=maxit,
        )
    elif ica_method == "fastica":
        mixing, fixed_seed = f_ica(
            data,
            n_components=n_components,
            fixed_seed=fixed_seed,
            maxit=maxit,
            maxrestart=maxrestart,
        )
    else:
        raise ValueError("The selected ICA method is invalid!")

    return (
        mixing,
        fixed_seed,
        c_labels,
        similarity_t_sne,
        fastica_convergence_warning_count,
        index_quality,
    )


def r_ica(data, n_components, fixed_seed, n_robust_runs, max_it):
    """Perform robustica on `data` and returns mixing matrix.

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
    mixing : (T x C) :obj:`numpy.ndarray`
        Z-scored mixing matrix for converting input data to component space,
        where `C` is components and `T` is the same as in `data`
    fixed_seed : :obj:`int`
        Random seed from final decomposition.
    c_labels : (n_pca_components x n_robust_runs,) :obj:`numpy.ndarray`
        A one dimensional array that has the cluster label of each run.
    similarity_t_sne : (n_pca_components x n_robust_runs,2) :obj:`numpy.ndarray`
        An array containing the 2D coordinates of projected data.
    fastica_convergence_warning_count : :obj:`int`
        The number of iterations of fastICA that failed to converge.
    index_quality : :obj:`float`
        The mean cluster index quality for robustICA.
        robustICA cites https://doi.org/10.1109/NNSP.2003.1318025 for the measure
    """
    if n_robust_runs > WARN_N_ROBUST_RUNS:
        LGR.warning(
            "The selected n_robust_runs is a very big number! The process will take a long time!"
        )

    RepLGR.info("RobustICA package was used for ICA decomposition \\citep{anglada2022robustica}.")

    if fixed_seed == -1:
        fixed_seed = np.random.randint(low=1, high=1000)

    robust_method = "DBSCAN"
    robust_ica_converged = False
    while not robust_ica_converged:
        try:
            robust_ica = RobustICA(
                n_components=n_components,
                robust_runs=n_robust_runs,
                whiten="arbitrary-variance",
                max_iter=max_it,
                random_state=fixed_seed,
                robust_dimreduce=False,
                fun="logcosh",
                robust_method=robust_method,
            )

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter(
                    "always", category=ConvergenceWarning
                )  # Ensure all warnings are captured
                s, mixing = robust_ica.fit_transform(data)

            # Count specific FastICA convergence warnings
            fastica_convergence_warning_count = 0
            for w in caught_warnings:
                if issubclass(
                    w.category, ConvergenceWarning
                ) and "FastICA did not converge" in str(w.message):
                    fastica_convergence_warning_count += 1

            nonconverge_message = (
                "For RobustICA, FastICA did not converge in "
                f"{fastica_convergence_warning_count} of {n_robust_runs} interations."
            )
            if fastica_convergence_warning_count / n_robust_runs > 0.25:
                LGR.warning(
                    f"{nonconverge_message} "
                    "Failing >1/4 of the time means inputted data are not appropriate for ICA. "
                    "Consider rerunning with fewer initial PCA components."
                )
            elif fastica_convergence_warning_count / n_robust_runs > 0.1:
                # Log a warning if there's non-convergence in >10% of the attempted iterations
                LGR.warning(
                    f"{nonconverge_message} "
                    "Consider rerunning with fewer initial PCA components."
                )
            elif fastica_convergence_warning_count > 0:
                # Log info if there's non-convergence in <=10% of the attempted iterations
                LGR.info(f"{nonconverge_message}")

            q = robust_ica.evaluate_clustering(
                robust_ica.S_all,
                robust_ica.clustering.labels_,
                robust_ica.signs_,
                robust_ica.orientation_,
            )
            robust_ica_converged = True

        except Exception:
            if robust_method == "DBSCAN":
                # if RobustICA failed wtih DBSCAN, run again with AgglomerativeClustering
                robust_method = "AgglomerativeClustering"
                LGR.warning(
                    "DBSCAN clustering method did not converge. "
                    "Agglomerative clustering will be tried now."
                )
            else:
                raise ValueError("RobustICA failed to converge")

    LGR.info(
        f"The {robust_method} clustering algorithm was used for clustering "
        f"components across different runs"
    )

    # Excluding outliers (cluster -1) from the index quality calculation
    index_quality = np.array(np.mean(q[q["cluster_id"] >= 0].iq))

    if index_quality < WARN_IQ:
        LGR.warning(
            f"The resultant mean Index Quality is low ({index_quality}). "
            "It is recommended to rerun the process with a different seed."
        )

    # Excluding outliers (cluster -1) when calculating the mixing matrix
    mixing = mixing[:, q["cluster_id"] >= 0]
    mixing = stats.zscore(mixing, axis=0)

    LGR.info(
        f"RobustICA with {n_robust_runs} robust runs and seed {fixed_seed} was used. "
        f"{mixing.shape[1]} components identified. "
        f"The mean Index Quality is {index_quality}."
    )

    no_outliers = np.count_nonzero(robust_ica.clustering.labels_ == -1)
    if no_outliers:
        LGR.info(
            f"The {robust_method} clustering algorithm detected outliers when clustering "
            f"components for different runs. These outliers are excluded when calculating "
            f"the index quality and the mixing matrix to maximise the robustness of the "
            f"decomposition."
        )

    c_labels = robust_ica.clustering.labels_

    perplexity = min(robust_ica.S_all.shape[1] - 1, 80)

    perplexity = perplexity - 1 if perplexity < 81 else 80
    t_sne = manifold.TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        n_iter=2500,
        random_state=10,
    )

    p_dissimilarity = abs_pearson_dist(robust_ica.S_all)
    similarity_t_sne = t_sne.fit_transform(p_dissimilarity)

    return (
        mixing,
        fixed_seed,
        c_labels,
        similarity_t_sne,
        fastica_convergence_warning_count,
        index_quality,
    )


def f_ica(data, n_components, fixed_seed, maxit, maxrestart):
    """Perform FastICA on `data` and returns mixing matrix.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
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
    mixing : (T x C) :obj:`numpy.ndarray`
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

    mixing = ica.mixing_
    mixing = stats.zscore(mixing, axis=0)
    return mixing, fixed_seed
