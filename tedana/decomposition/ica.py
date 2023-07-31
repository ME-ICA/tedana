"""
ICA and related signal decomposition methods for tedana
"""
import logging
import warnings

import sys

import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA
from robustica import RobustICA ####BTBTBT

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def tedica(data, n_components, fixed_seed, ica_method="robustica", n_robust_runs=30, maxit=500, maxrestart=10): ####BTBTBTB
    """
    Perform ICA on `data` and returns mixing matrix

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

    if ica_method=='robustica':
        mmix, Iq = r_ica(data, n_components, n_robust_runs, maxit)
        fixed_seed=-99999
    elif ica_method=='fastica':
        mmix, fixed_seed=f_ica(data, n_components, fixed_seed, maxit=500, maxrestart=10)
        Iq = 0
    else:
        LGR.warning("The selected ICA method is invalid!")
        sys.exit()




    return mmix, fixed_seed


def r_ica(data, n_components, n_robust_runs, max_it): ####BTBTBTB:

    if n_robust_runs>100:
        LGR.warning("The selected n_robust_runs is a very big number!")


    RepLGR.info(
        "RobustICA package was used for ICA decomposition \\citep{Anglada2022}."
    )
    rica0 = RobustICA(n_components=n_components, robust_runs=n_robust_runs, whiten='arbitrary-variance',max_iter= max_it,
                      robust_dimreduce=False, fun='logcosh')
    S0, mmix = rica0.fit_transform(data)

    q0 = rica0.evaluate_clustering(rica0.S_all, rica0.clustering.labels_, rica0.signs_, rica0.orientation_)

    
    Iq0 = np.array(np.mean(q0.iq))
        

    mmix = stats.zscore(mmix, axis=0)

    LGR.info(
        "RobustICA with {0} robust runs was used \n"
        "The mean index quality is {1}".format(n_robust_runs, Iq0)
    )
    return mmix, Iq0


def f_ica(data, n_components, fixed_seed, maxit, maxrestart):
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
