"""
Statistical functions
"""
import logging

import numpy as np
from scipy import stats
from scipy.special import ndtri
from nilearn.mass_univariate import permuted_ols

from tedana import utils

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def getfbounds(n_echos):
    """
    Gets F-statistic boundaries based on number of echos

    Parameters
    ----------
    n_echos : :obj:`int`
        Number of echoes

    Returns
    -------
    fmin, fmid, fmax : :obj:`float`
        F-statistic thresholds for alphas of 0.05, 0.025, and 0.01,
        respectively.
    """
    f05 = stats.f.ppf(q=(1 - 0.05), dfn=1, dfd=(n_echos - 1))
    f025 = stats.f.ppf(q=(1 - 0.025), dfn=1, dfd=(n_echos - 1))
    f01 = stats.f.ppf(q=(1 - 0.01), dfn=1, dfd=(n_echos - 1))
    return f05, f025, f01


def p_to_z(p, tail='two'):
    """Convert p-values to z-values.
    """
    eps = np.spacing(1)
    p = np.array(p)
    p[p < eps] = eps
    if tail == 'two':
        z = ndtri(1 - (p / 2))
        z = np.array(z)
    elif tail == 'one':
        z = ndtri(1 - p)
        z = np.array(z)
        z[z < 0] = 0
    else:
        raise ValueError('Argument "tail" must be one of ["one", "two"]')

    if z.shape == ():
        z = z[()]
    return z


def computefeats2(data, mmix, mask=None, normalize=True):
    """
    Converts `data` to component space using `mmix`

    Parameters
    ----------
    data : (S x T) array_like
        Input data
    mmix : (T [x C]) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like or None, optional
        Boolean mask array. Default: None
    normalize : bool, optional
        Whether to z-score output. Default: True

    Returns
    -------
    data_Z : (S x C) :obj:`numpy.ndarray`
        Data in component space
    """
    if data.ndim != 2:
        raise ValueError('Parameter data should be 2d, not {0}d'.format(data.ndim))
    elif mmix.ndim not in [2]:
        raise ValueError('Parameter mmix should be 2d, not '
                         '{0}d'.format(mmix.ndim))
    elif (mask is not None) and (mask.ndim != 1):
        raise ValueError('Parameter mask should be 1d, not {0}d'.format(mask.ndim))
    elif (mask is not None) and (data.shape[0] != mask.shape[0]):
        raise ValueError('First dimensions (number of samples) of data ({0}) '
                         'and mask ({1}) do not match.'.format(data.shape[0],
                                                               mask.shape[0]))
    elif data.shape[1] != mmix.shape[0]:
        raise ValueError('Second dimensions (number of volumes) of data ({0}) '
                         'and mmix ({1}) do not match.'.format(data.shape[0],
                                                               mmix.shape[0]))

    # demean masked data
    if mask is not None:
        data = data[mask, ...]
    LGR.info('Starting OLS')
    pe_logp_vals, pe_t_vals, _ = permuted_ols(mmix, data.T, model_intercept=True, n_perm=500, verbose=1)
    pe_p_vals = np.power(10., -pe_logp_vals)
    pe_z_vals = p_to_z(pe_p_vals, tail='two')
    pe_z_vals *= np.sign(pe_t_vals)
    LGR.info('Finished OLS')
    return pe_z_vals.T


def get_coeffs(data, X, mask=None, add_const=False):
    """
    Performs least-squares fit of `X` against `data`

    Parameters
    ----------
    data : (S [x E] x T) array_like
        Array where `S` is samples, `E` is echoes, and `T` is time
    X : (T [x C]) array_like
        Array where `T` is time and `C` is predictor variables
    mask : (S [x E]) array_like
        Boolean mask array
    add_const : bool, optional
        Add intercept column to `X` before fitting. Default: False

    Returns
    -------
    betas : (S [x E] x C) :obj:`numpy.ndarray`
        Array of `S` sample betas for `C` predictors
    """
    if data.ndim not in [2, 3]:
        raise ValueError('Parameter data should be 2d or 3d, not {0}d'.format(data.ndim))
    elif X.ndim not in [2]:
        raise ValueError('Parameter X should be 2d, not {0}d'.format(X.ndim))
    elif data.shape[-1] != X.shape[0]:
        raise ValueError('Last dimension (dimension {0}) of data ({1}) does not '
                         'match first dimension of '
                         'X ({2})'.format(data.ndim, data.shape[-1], X.shape[0]))

    # mask data and flip (time x samples)
    if mask is not None:
        if mask.ndim not in [1, 2]:
            raise ValueError('Parameter data should be 1d or 2d, not {0}d'.format(mask.ndim))
        elif data.shape[0] != mask.shape[0]:
            raise ValueError('First dimensions of data ({0}) and mask ({1}) do not '
                             'match'.format(data.shape[0], mask.shape[0]))
        mdata = data[mask, :].T
    else:
        mdata = data.T

    # coerce X to >=2d
    X = np.atleast_2d(X)

    if len(X) == 1:
        X = X.T

    if add_const:  # add intercept, if specified
        X = np.column_stack([X, np.ones((len(X), 1))])

    betas = np.linalg.lstsq(X, mdata, rcond=None)[0].T
    if add_const:  # drop beta for intercept, if specified
        betas = betas[:, :-1]

    if mask is not None:
        betas = utils.unmask(betas, mask)

    return betas
