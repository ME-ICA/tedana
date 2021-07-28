"""
Statistical functions
"""
import logging

import numpy as np
from scipy import stats

from tedana import utils
from tedana.due import BibTeX, Doi, due

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")


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
        raise ValueError("Parameter data should be 2d, not {0}d".format(data.ndim))
    elif mmix.ndim not in [2]:
        raise ValueError("Parameter mmix should be 2d, not {0}d".format(mmix.ndim))
    elif (mask is not None) and (mask.ndim != 1):
        raise ValueError("Parameter mask should be 1d, not {0}d".format(mask.ndim))
    elif (mask is not None) and (data.shape[0] != mask.shape[0]):
        raise ValueError(
            "First dimensions (number of samples) of data ({0}) "
            "and mask ({1}) do not match.".format(data.shape[0], mask.shape[0])
        )
    elif data.shape[1] != mmix.shape[0]:
        raise ValueError(
            "Second dimensions (number of volumes) of data ({0}) "
            "and mmix ({1}) do not match.".format(data.shape[0], mmix.shape[0])
        )

    # demean masked data
    if mask is not None:
        data = data[mask, ...]
    # normalize data (subtract mean and divide by standard deviation) in the last dimension
    # so that least-squares estimates represent "approximate" correlation values (data_R)
    # assuming mixing matrix (mmix) values are also normalized
    data_vn = stats.zscore(data, axis=-1)

    # get betas of `data`~`mmix` and limit to range [-0.999, 0.999]
    data_R = get_coeffs(data_vn, mmix, mask=None)
    # Avoid abs(data_R) => 1, otherwise Fisher's transform will return Inf or -Inf
    data_R[data_R < -0.999] = -0.999
    data_R[data_R > 0.999] = 0.999

    # R-to-Z transform
    data_Z = np.arctanh(data_R)
    if data_Z.ndim == 1:
        data_Z = np.atleast_2d(data_Z).T

    # normalize data (only division by std)
    if normalize:
        # subtract mean and dividing by standard deviation
        data_Zm = stats.zscore(data_Z, axis=0)
        # adding back the mean
        data_Z = data_Zm + (data_Z.mean(axis=0, keepdims=True) / data_Z.std(axis=0, keepdims=True))

    return data_Z


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
        raise ValueError("Parameter data should be 2d or 3d, not {0}d".format(data.ndim))
    elif X.ndim not in [2]:
        raise ValueError("Parameter X should be 2d, not {0}d".format(X.ndim))
    elif data.shape[-1] != X.shape[0]:
        raise ValueError(
            "Last dimension (dimension {0}) of data ({1}) does not "
            "match first dimension of "
            "X ({2})".format(data.ndim, data.shape[-1], X.shape[0])
        )

    # mask data and flip (time x samples)
    if mask is not None:
        if mask.ndim not in [1, 2]:
            raise ValueError("Parameter data should be 1d or 2d, not {0}d".format(mask.ndim))
        elif data.shape[0] != mask.shape[0]:
            raise ValueError(
                "First dimensions of data ({0}) and mask ({1}) do not "
                "match".format(data.shape[0], mask.shape[0])
            )
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


@due.dcite(
    BibTeX(
        """
           @article{hughett2007accurate,
             title={Accurate Computation of the F-to-z and t-to-z Transforms
                    for Large Arguments},
             author={Hughett, Paul},
             journal={Journal of Statistical Software},
             volume={23},
             number={1},
             pages={1--5},
             year={2007},
             publisher={Foundation for Open Access Statistics}
           }
           """
    ),
    description="Introduces T-to-Z transform.",
)
@due.dcite(Doi("10.5281/zenodo.32508"), description="Python implementation of T-to-Z transform.")
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
        "Vanessa Sochat's implementation (Sochat, 2015) of the method "
        "described in Hughett (2007)."
    )
    RefLGR.info(
        "Sochat, V. (2015). TtoZ Original Release. Zenodo. http://doi.org/10.5281/zenodo.32508."
    )
    RefLGR.info(
        "Hughett, P. (2007). Accurate Computation of the F-to-z and "
        "t-to-z Transforms for Large Arguments. Journal of "
        "Statistical Software, 23(1), 1-5."
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
