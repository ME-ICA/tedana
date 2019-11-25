"""
Statistical functions
"""
import logging

import numpy as np
from scipy import stats

from tedana import utils

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

@due.dcite(references.T2Z_TRANSFORM,
           description='Introduces T-to-Z transform.')
@due.dcite(references.T2Z_IMPLEMENTATION,
           description='Python implementation of T-to-Z transform.')

def t_to_z(t_values, dof):
    """
    From Vanessa Sochat's TtoZ package.
    """

    # check if t_values is np.array, and convert if required
    t_values = np.asanyarray(t_values)

    # Select just the nonzero voxels
    nonzero = t_values[t_values != 0]

    # We will store our results here
    z_values = np.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = np.zeros(len(nonzero))
    k1 = (nonzero <= c)
    k2 = (nonzero > c)
    # Subset the data into two sets
    t1 = nonzero[k1]
    t2 = nonzero[k2]

    # Calculate p values for <=0
    p_values_t1 = stats.t.cdf(t1, df=dof)
    z_values_t1 = stats.norm.ppf(p_values_t1)

    # Calculate p values for > 0
    p_values_t2 = stats.t.cdf(-t2, df=dof)
    z_values_t2 = -stats.norm.ppf(p_values_t2)
    z_values[k1] = z_values_t1
    z_values[k2] = z_values_t2
    # Write new image to file
    out = np.zeros(t_values.shape)
    out[t_values != 0] = z_values
    return out

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
    # normalize data (minus mean and divide by std)
    data_vn = stats.zscore(data, axis=-1)

    # get betas and z-values of `data`~`mmix`
    # mmix is normalized internally
    data_R, data_Z = get_coeffs(data_vn, mmix, mask=None, add_const=False, compute_zvalues=True)
    if data_Z.ndim == 1:
        data_Z = np.atleast_2d(data_Z).T

    # normalize data (only division by std)
    if normalize:
        # minus mean and divided by std
        data_Zm = stats.zscore(data_Z, axis=0)
        # adding back the mean
        data_Z = data_Zm + (data_Z.mean(axis=0, keepdims=True) /
                            data_Z.std(axis=0, keepdims=True))

    return data_Z


def get_coeffs(data, X, mask=None, add_const=False, compute_zvalues=True, min_df=1):
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
    compute_zvalues : bool, optional
        Compute z-values of the betas (predictors)
    min_df : integer, optional
        Integer to give warning if # df <= min_df

    Returns
    -------
    betas : (S [x E] x C) :obj:`numpy.ndarray`
        Array of `S` sample betas for `C` predictors
    z_values : (S [x E] x C) :obj:`numpy.ndarray`
        Array of `S` sample z-values for `C` predictors

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

    # least squares estimation
    betas = np.dot(np.linalg.pinv(X),mdata)

    if compute_zvalues:
        # compute t-values of betas (estimates) and then convert to z-values
        # first compute number of degrees of freedom
        df = mdata.shape[0] - X.shape[1]
        if df == 0:
            LGR.error('ERROR: No degrees of freedom left in least squares calculation. Stopping!!')
        else:
            elif df <= min_df:
                LGR.warning('Number of degrees of freedom in least-square estimation is less than {}'.format(min_df+1))
            # compute residual sum of squares (RSS)
            RSS = np.sum(np.power(mdata - np.dot(X, betas.T),2),axis=0)/df
            RSS = RSS[:,np.newaxis]
            C = np.diag(np.linalg.pinv(np.dot(X.T,X)))
            C = C[:,np.newaxis]
            std_betas = np.sqrt(np.dot(RSS,C.T))
            z_values = t_to_z(betas / std_betas,df)

    if add_const:  # drop beta for intercept, if specified
        betas = betas[:, :-1]
        if compute_zvalues:
            z_values = z_values[:, :-1]

    if mask is not None:
        betas = utils.unmask(betas, mask)
        if compute_zvalues:
            z_values = utils.unmask(z_values, mask)

    if compute_zvalues:
        return betas, z_values
    else:
        return betas

    
