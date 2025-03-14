"""Statistical functions."""

import logging

import numpy as np
from scipy import linalg, stats

from tedana import utils

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def getfbounds(n_independent_sources):
    """
    Get F-statistic boundaries based on number of echos.

    Parameters
    ----------
    n_independent_sources : :obj:`int`
        The number of independent sources to calculate DOF for goodness of fit metrics (fstat).
        Typically the number of echos in the multi-echo data
        May be a lower value for EPTI acquisitions.

    Returns
    -------
    fmin, fmid, fmax : :obj:`float`
        F-statistic thresholds for alphas of 0.05, 0.025, and 0.01,
        respectively.
    """
    f05 = stats.f.ppf(q=(1 - 0.05), dfn=1, dfd=(n_independent_sources - 1))
    f025 = stats.f.ppf(q=(1 - 0.025), dfn=1, dfd=(n_independent_sources - 1))
    f01 = stats.f.ppf(q=(1 - 0.01), dfn=1, dfd=(n_independent_sources - 1))
    return f05, f025, f01


def computefeats2(data, mixing, mask=None, normalize=True):
    """
    Convert `data` to component space using `mixing`.

    Parameters
    ----------
    data : (S x T) array_like
        Input data
    mixing : (T [x C]) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like or None, optional
        Boolean mask array. Default: None
    normalize : bool, optional
        Whether to z-score output. Default: True

    Returns
    -------
    data_z : (S x C) :obj:`numpy.ndarray`
        Data in component space
    """
    if data.ndim != 2:
        raise ValueError(f"Parameter data should be 2d, not {data.ndim}d")
    elif mixing.ndim not in [2]:
        raise ValueError(f"Parameter mixing should be 2d, not {mixing.ndim}d")
    elif (mask is not None) and (mask.ndim != 1):
        raise ValueError(f"Parameter mask should be 1d, not {mask.ndim}d")
    elif (mask is not None) and (data.shape[0] != mask.shape[0]):
        raise ValueError(
            f"First dimensions (number of samples) of data ({data.shape[0]}) "
            f"and mask ({mask.shape[0]}) do not match."
        )
    elif data.shape[1] != mixing.shape[0]:
        raise ValueError(
            f"Second dimensions (number of volumes) of data ({data.shape[0]}) "
            f"and mixing ({mixing.shape[0]}) do not match."
        )

    # demean masked data
    if mask is not None:
        data = data[mask, ...]
    # normalize data (subtract mean and divide by standard deviation) in the last dimension
    # so that least-squares estimates represent "approximate" correlation values (data_r)
    # assuming mixing matrix (mixing) values are also normalized
    data_vn = stats.zscore(data, axis=-1)

    # get betas of `data`~`mixing` and limit to range [-0.999, 0.999]
    data_r = get_coeffs(data_vn, mixing, mask=None)
    # Avoid abs(data_r) => 1, otherwise Fisher's transform will return Inf or -Inf
    data_r[data_r < -0.999] = -0.999
    data_r[data_r > 0.999] = 0.999

    # R-to-Z transform
    data_z = np.arctanh(data_r)
    if data_z.ndim == 1:
        data_z = np.atleast_2d(data_z).T

    # normalize data (only division by std)
    if normalize:
        # subtract mean and dividing by standard deviation
        data_zm = stats.zscore(data_z, axis=0)
        # adding back the mean
        data_z = data_zm + (data_z.mean(axis=0, keepdims=True) / data_z.std(axis=0, keepdims=True))

    return data_z


def get_coeffs(data, x, mask=None, add_const=False):
    """
    Perform least-squares fit of `x` against `data`.

    Parameters
    ----------
    data : (S [x E] x T) array_like
        Array where `S` is samples, `E` is echoes, and `T` is time
    x : (T [x C]) array_like
        Array where `T` is time and `C` is predictor variables
    mask : (S [x E]) array_like
        Boolean mask array
    add_const : bool, optional
        Add intercept column to `x` before fitting. Default: False

    Returns
    -------
    betas : (S [x E] x C) :obj:`numpy.ndarray`
        Array of `S` sample betas for `C` predictors
    """
    if data.ndim not in [2, 3]:
        raise ValueError(f"Parameter data should be 2d or 3d, not {data.ndim}d")
    elif x.ndim not in [2]:
        raise ValueError(f"Parameter x should be 2d, not {x.ndim}d")
    elif data.shape[-1] != x.shape[0]:
        raise ValueError(
            f"Last dimension (dimension {data.ndim}) of data ({data.shape[-1]}) does not "
            f"match first dimension of x ({x.shape[0]})"
        )

    # mask data and flip (time x samples)
    if mask is not None:
        if mask.ndim not in [1, 2]:
            raise ValueError(f"Parameter data should be 1d or 2d, not {mask.ndim}d")
        elif data.shape[0] != mask.shape[0]:
            raise ValueError(
                f"First dimensions of data ({data.shape[0]}) and "
                f"mask ({mask.shape[0]}) do not match"
            )
        mdata = data[mask, :].T
    else:
        mdata = data.T

    # coerce x to >=2d
    x = np.atleast_2d(x)

    if len(x) == 1:
        x = x.T

    if add_const:  # add intercept, if specified
        x = np.column_stack([x, np.ones((len(x), 1))])

    betas = np.linalg.lstsq(x, mdata, rcond=None)[0].T
    if add_const:  # drop beta for intercept, if specified
        betas = betas[:, :-1]

    if mask is not None:
        betas = utils.unmask(betas, mask)

    return betas


def t_to_z(t_values, dof):
    """
    Convert t-values to z-values.

    Parameters
    ----------
    t_values
    dof

    Returns
    -------
    out

    Notes
    -----
    From Vanessa Sochat's TtoZ package.
    https://github.com/vsoch/TtoZ
    """
    if not isinstance(t_values, np.ndarray):
        ret_float = True
        t_values = np.array([t_values])
    else:
        ret_float = False

    RepLGR.info(
        "T-statistics were converted to z-statistics using Dr. "
        "Vanessa Sochat's implementation \\citep{sochat2015ttoz} of the method "
        "described in \\citep{hughett2008accurate}."
    )

    # Select just the nonzero voxels
    nonzero = t_values[t_values != 0]

    # We will store our results here
    z_values = np.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = np.zeros(len(nonzero))
    k1 = nonzero <= c
    k2 = nonzero > c

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

    if ret_float:
        out = out[0]
    return out


def fit_model(x, y, output_residual=False):
    """
    Linear regression for a model y = betas * x + error.

    Parameters
    ----------
    x : (T X R) :obj:`numpy.ndarray`
        2D array with the regressors for the specified model an time
    y : (T X C) :obj:`numpy.ndarray`
        Time by mixing matrix components for the time series for fitting
    output_residual : :obj:`bool`
        If true, then this just outputs the residual of the fit.
        If false, then outputs beta fits, sse, and df

    Returns
    -------
    residual : (T X C) :obj:`numpy.ndarray`
        The residual time series for the fit (only if output_residual is True)
    betas : (R X C) :obj:`numpy.ndarray`
        The magnitude fits for the model (only if output_residual is False)
    sse : (C) :obj:`numpy.ndarray`
        The sum of square error for the model (only if output_residual is False)
    df : :obj:`int`
        The degrees of freeom for the model (only if output_residual is False)
        (timepoints - number of regressors)
    """
    betas, _, _, _ = linalg.lstsq(x, y)
    # matrix-multiplication on the regressors with the betas -> to create a new 'estimated'
    # component matrix  = fitted regressors (least squares beta solution * regressors)
    fitted_regressors = np.matmul(x, betas)
    residual = y - fitted_regressors
    if output_residual:
        return residual
    else:
        # sum the differences between the actual ICA components and the 'estimated'
        # component matrix (beta-fitted regressors)
        sse = np.sum(np.square(residual), axis=0)
        # calculate how many individual values [timepoints] are free to vary after
        # the least-squares solution [beta] betw X & Y is calculated
        df = y.shape[0] - betas.shape[0]
        return betas, sse, df
