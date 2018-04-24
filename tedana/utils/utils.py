"""Utilities for meica package"""
import numpy as np
import nibabel as nib
from scipy.optimize import leastsq
from scipy.stats import scoreatpercentile

from ..due import due, BibTeX


def cat2echos(data, Ne):
    """
    Separates z- and echo-axis in `data`

    Parameters
    ----------
    data : array_like
        Array of shape (nx, ny, nz*Ne, nt)
    Ne : int
        Number of echoes that were in original (uncombined) data array

    Returns
    -------
    ndarray
        Array of shape (nx, ny, nz, Ne, nt)
    """

    nx, ny = data.shape[0:2]
    nz = data.shape[2] // Ne
    if len(data.shape) > 3:
        nt = data.shape[3]
    else:
        nt = 1
    return np.reshape(data, (nx, ny, nz, Ne, nt), order='F')


def makeadmask(cdat, minimum=True, getsum=False):
    """
    Create a mask.
    """
    nx, ny, nz, Ne, _ = cdat.shape

    mask = np.ones((nx, ny, nz), dtype=np.bool)

    if minimum:
        mask = cdat.prod(axis=-1).prod(-1) != 0
        return mask
    else:
        # Make a map of longest echo that a voxel can be sampled with,
        # with minimum value of map as X value of voxel that has median
        # value in the 1st echo. N.b. larger factor leads to bias to lower TEs
        emeans = cdat.mean(-1)
        medv = emeans[:, :, :, 0] == scoreatpercentile(emeans[:, :, :, 0][emeans[:, :, :, 0] != 0],
                                                       33, interpolation_method='higher')
        lthrs = np.squeeze(np.array([emeans[:, :, :, ee][medv] / 3 for ee in range(Ne)]))

        if len(lthrs.shape) == 1:
            lthrs = np.atleast_2d(lthrs).T
        lthrs = lthrs[:, lthrs.sum(0).argmax()]

        mthr = np.ones([nx, ny, nz, Ne])
        for i_echo in range(Ne):
            mthr[:, :, :, i_echo] *= lthrs[i_echo]
        mthr = np.abs(emeans) > mthr
        masksum = np.array(mthr, dtype=np.int).sum(-1)
        mask = masksum != 0
        if getsum:
            return mask, masksum
        else:
            return mask


def uncat2echos(data, Ne):
    """
    Combines z- and echo-axis in `data`

    Parameters
    ----------
    data : array_like
        Array of shape (nx, ny, nz, Ne, nt)
    Ne : int
        Number of echoes; should be equal to `data.shape[3]`

    Returns
    -------
    ndarray
        Array of shape (nx, ny, nz*Ne, nt)
    """

    nx, ny = data.shape[0:2]
    nz = data.shape[2] * Ne
    if len(data.shape) > 4:
        nt = data.shape[4]
    else:
        nt = 1
    return np.reshape(data, (nx, ny, nz, nt), order='F')


def make_mask(catdata):
    """
    Generates a 3D mask of `catdata`

    Only voxels that are consistently (i.e., across time AND echoes) non-zero
    in `catdata` are True in output

    Parameters
    ----------
    catdata : (X x Y x Z x E x T) array_like
        Multi-echo data array, where X, Y, Z are spatial dimensions, E
        corresponds to individual echo data, and T is time

    Returns
    -------
    mask : (X x Y x Z) np.ndarray
        Boolean array
    """

    catdata = np.asarray(catdata)
    return catdata.prod(axis=-1).prod(axis=-1).astype('bool')


def make_opt_com(medata):
    """
    Makes optimal combination from input multi-echo data

    Parameters
    ----------
    medata : tedana.interfaces.data.MultiEchoData
    """

    pass


def fmask(data, mask):
    """
    Masks `data` with non-zero entries of `mask`

    Parameters
    ----------
    data : array_like
        Array of shape (nx, ny, nz[, Ne[, nt]])
    mask : array_like
        Boolean array of shape (nx, ny, nz)

    Returns
    -------
    ndarray
        Masked array of shape (nx*ny*nz[, Ne[, nt]])
    """

    s = data.shape

    N = s[0] * s[1] * s[2]
    news = []
    news.append(N)

    if len(s) > 3:
        news.extend(s[3:])

    tmp1 = np.reshape(data, news)
    fdata = tmp1.compress((mask > 0).ravel(), axis=0)

    return fdata.squeeze()


def unmask(data, mask):
    """
    Unmasks `data` using non-zero entries of `mask`

    Parameters
    ----------
    data : array_like
        Masked array of shape (nx*ny*nz[, Ne[, nt]])
    mask : array_like
        Boolean array of shape (nx, ny, nz)

    Returns
    -------
    ndarray
        Array of shape (nx, ny, nz[, Ne[, nt]])
    """

    M = (mask != 0).ravel()
    Nm = M.sum()

    nx, ny, nz = mask.shape

    if len(data.shape) > 1:
        nt = data.shape[1]
    else:
        nt = 1

    out = np.zeros((nx * ny * nz, nt), dtype=data.dtype)
    out[M, :] = np.reshape(data, (Nm, nt))

    return np.squeeze(np.reshape(out, (nx, ny, nz, nt)))


def moments(data):
    """
    Returns gaussian parameters of a 2D distribution by calculating its moments

    Parameters
    ----------
    data : array_like
        2D data array

    Returns
    -------
    height : float
    center_x : float
    center_y : float
    width_x : float
    width_y : float

    References
    ----------
    `Scipy Cookbook`_

    .. _Scipy Cookbook: http://scipy-cookbook.readthedocs.io/items/FittingData.html#Fitting-a-2D-gaussian  # noqa
    """

    total = data.sum()
    X, Y = np.indices(data.shape)
    center_x = (X * data).sum() / total
    center_y = (Y * data).sum() / total
    col = data[:, int(center_y)]
    width_x = np.sqrt(abs((np.arange(col.size) - center_y)**2 * col).sum() / col.sum())
    row = data[int(center_x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - center_x)**2 * row).sum() / row.sum())
    height = data.max()
    return height, center_x, center_y, width_x, width_y


def gaussian(height, center_x, center_y, width_x, width_y):
    """
    Returns gaussian function

    Parameters
    ----------
    height : float
    center_x : float
    center_y : float
    width_x : float
    width_y : float

    Returns
    -------
    lambda
        Gaussian function with provided parameters

    References
    ----------
    `Scipy Cookbook`_

    .. _Scipy Cookbook: http://scipy-cookbook.readthedocs.io/items/FittingData.html#Fitting-a-2D-gaussian  # noqa
    """

    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(-(((center_x - x) / width_x)**2 +
                                        ((center_y - y) / width_y)**2) / 2)


def fitgaussian(data):
    """
    Returns estimated gaussian parameters of a 2D distribution found by a fit

    Parameters
    ----------
    data : array_like
        2D data array

    Returns
    -------
    p : array_like
        Array with height, center_x, center_y, width_x, width_y of `data`

    References
    ----------
    `Scipy Cookbook`_

    .. _Scipy Cookbook: http://scipy-cookbook.readthedocs.io/items/FittingData.html#Fitting-a-2D-gaussian  # noqa
    """

    params = moments(data)

    def errorfunction(p, data):
        return np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)

    (p, _) = leastsq(errorfunction, params, data)
    return p


def niwrite(data, affine, name, head, header=None):
    """
    Write out nifti file.
    """
    data[np.isnan(data)] = 0
    if header is None:
        this_header = head.copy()
        this_header.set_data_shape(list(data.shape))
    else:
        this_header = header

    outni = nib.Nifti1Image(data, affine, header=this_header)
    outni.set_data_dtype('float64')
    outni.to_filename(name)


@due.dcite(BibTeX('@article{dice1945measures,'
                  'author={Dice, Lee R},'
                  'title={Measures of the amount of ecologic association between species},'
                  'year = {1945},'
                  'publisher = {Wiley Online Library},'
                  'journal = {Ecology},'
                  'volume={26},'
                  'number={3},'
                  'pages={297--302}}'),
           description='Introduction of Sorenson-Dice index by Dice in 1945.')
@due.dcite(BibTeX('@article{sorensen1948method,'
                  'author={S{\\o}rensen, Thorvald},'
                  'title={A method of establishing groups of equal amplitude '
                  'in plant sociology based on similarity of species and its '
                  'application to analyses of the vegetation on Danish commons},'
                  'year = {1948},'
                  'publisher = {Wiley Online Library},'
                  'journal = {Biol. Skr.},'
                  'volume={5},'
                  'pages={1--34}}'),
           description='Introduction of Sorenson-Dice index by Sorenson in 1948.')
def dice(arr1, arr2):
    """
    Compute Dice's similarity index between two numpy arrays. Arrays will be
    binarized before comparison.

    Parameters
    ----------
    arr1, arr2 : array-like
        Input arrays, arrays to binarize and compare.

    Returns
    -------
    dsi : float
        Dice-Sorenson index.

    References
    ----------
    REF_

    .. _REF: https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
    """
    arr1 = np.array(arr1 != 0).astype(int)
    arr2 = np.array(arr2 != 0).astype(int)

    if arr1.shape != arr2.shape:
        raise ValueError('Shape mismatch: arr1 and arr2 must have the same shape.')

    arr_sum = arr1.sum() + arr2.sum()
    if arr_sum == 0:
        dsi = 0
    else:
        intersection = np.logical_and(arr1, arr2)
        dsi = (2. * intersection.sum()) / arr_sum

    return dsi


def andb(arrs):
    """
    Sums arrays in `arrs`

    Parameters
    ----------
    arrs : list
        List of boolean or integer arrays to be summed

    Returns
    -------
    result : ndarray
        Integer array of summed `arrs`
    """

    same_shape = []
    for arr in arrs:
        for arr2 in arrs:
            same_shape.append(arr.shape == arr2.shape)

    if not np.all(same_shape):
        raise ValueError('All input arrays must have same shape')

    result = np.zeros(arrs[0].shape)
    for arr in arrs:
        result += np.array(arr, dtype=np.int)
    return result
