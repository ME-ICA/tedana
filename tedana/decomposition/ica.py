"""
ICA and related signal decomposition methods for tedana
"""
import logging
import warnings

import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def tedica(data, n_components, fixed_seed, maxit=500, maxrestart=10):
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

    Notes
    -----
    Uses `sklearn` implementation of FastICA for decomposition
    """
    warnings.filterwarnings(action='ignore', module='scipy',
                            message='^internal gelsd')
    RepLGR.info("Independent component analysis was then used to "
                "decompose the dimensionally reduced dataset.")

    if fixed_seed == -1:
        fixed_seed = np.random.randint(low=1, high=1000)

    for i_attempt in range(maxrestart):
        ica = FastICA(n_components=n_components, algorithm='parallel',
                      fun='logcosh', max_iter=maxit, random_state=fixed_seed)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered in order to capture
            # convergence failures.
            warnings.simplefilter('always')

            ica.fit(data)

            w = list(filter(lambda i: issubclass(i.category, UserWarning), w))
            if len(w):
                LGR.warning('ICA attempt {0} failed to converge after {1} '
                            'iterations'.format(i_attempt + 1, ica.n_iter_))
                if i_attempt < maxrestart - 1:
                    fixed_seed += 1
                    LGR.warning('Random seed updated to {0}'.format(fixed_seed))
            else:
                LGR.info('ICA attempt {0} converged in {1} '
                         'iterations'.format(i_attempt + 1, ica.n_iter_))
                break

    mmix = ica.mixing_
    mmix = stats.zscore(mmix, axis=0)
    return mmix
