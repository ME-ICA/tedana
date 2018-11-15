"""
Utilities for tedana package
"""
import nibabel as nib
import numpy as np
from nibabel.filename_parser import splitext_addext
from nilearn._utils import check_niimg
from scipy.optimize import leastsq
from sklearn.utils import check_array

from tedana.due import due, BibTeX

FORMATS = {'.nii': 'NIFTI'}


def get_dtype(data):
    """
    Determines neuroimaging format of `data`

    Parameters
    ----------
    data : :obj:`list` of :obj:`str` or :obj:`str` or img_like
        Data to determine format of

    Returns
    -------
    dtype : {'NIFTI', 'OTHER'} str
        Format of input data
    """

    if isinstance(data, list):
        dtypes = np.unique([get_dtype(d) for d in data])
        if dtypes.size > 1:
            raise ValueError('Provided data detected to have varying formats: '
                             '{}'.format(dtypes))
        return dtypes[0]
    elif isinstance(data, str):
        dtype = splitext_addext(data)[1]
    else:  # img_like?
        if not hasattr(data, 'valid_exts'):
            raise TypeError('Input data format cannot be detected.')
        dtype = data.valid_exts[0]

    return FORMATS.get(dtype, 'OTHER')


def getfbounds(n_echos):
    """
    Gets estimated F-statistic boundaries based on number of echos

    Parameters
    ----------
    n_echos : :obj:`int`
        Number of echoes

    Returns
    -------
    fmin, fmid, fmax : :obj:`float`
        Minimum, mid, and max F bounds
    """

    if not isinstance(n_echos, int):
        raise TypeError('Input n_echos must be type int. Type {} '
                        'invalid'.format(type(n_echos)))
    elif n_echos <= 0 or n_echos > 11:
        raise ValueError('Input `n_echos` must be >0 and <12. Provided '
                         'value: {}'.format(n_echos))
    idx = n_echos - 1

    F05s = [None, None, 18.5, 10.1, 7.7, 6.6, 6.0, 5.6, 5.3, 5.1, 5.0]
    F025s = [None, None, 38.5, 17.4, 12.2, 10, 8.8, 8.1, 7.6, 7.2, 6.9]
    F01s = [None, None, 98.5, 34.1, 21.2, 16.2, 13.8, 12.2, 11.3, 10.7, 10.]
    return F05s[idx], F025s[idx], F01s[idx]


def load_image(data):
    """
    Takes input `data` and returns a sample x time array

    Parameters
    ----------
    data : (X x Y x Z [x T]) array_like or img_like object
        Data array or data file to be loaded and reshaped

    Returns
    -------
    fdata : (S [x T]) :obj:`numpy.ndarray`
        Reshaped `data`, where `S` is samples and `T` is time
    """

    if isinstance(data, str):
        data = check_niimg(data).get_data()
    elif isinstance(data, nib.spatialimages.SpatialImage):
        data = check_niimg(data).get_data()

    fdata = data.reshape((-1,) + data.shape[3:]).squeeze()

    return fdata


def make_adaptive_mask(data, mask=None, minimum=True, getsum=False):
    """
    Makes map of `data` specifying longest echo a voxel can be sampled with

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    mask : :obj:`str` or img_like, optional
        Binary mask for voxels to consider in TE Dependent ANAlysis. Default is
        to generate mask from data with good signal across echoes
    minimum : :obj:`bool`, optional
        Use `make_min_mask()` instead of generating a map with echo-specific
        times. Default: True
    getsum : :obj:`bool`, optional
        Return `masksum` in addition to `mask`. Default: False

    Returns
    -------
    mask : (S,) :obj:`numpy.ndarray`
        Boolean array of voxels that have sufficient signal in at least one
        echo
    masksum : (S,) :obj:`numpy.ndarray`
        Valued array indicating the number of echos with sufficient signal in a
        given voxel. Only returned if `getsum = True`
    """

    if minimum:
        return make_min_mask(data, roi=mask)

    # take temporal mean of echos and extract non-zero values in first echo
    echo_means = data.mean(axis=-1)  # temporal mean of echos
    first_echo = echo_means[echo_means[:, 0] != 0, 0]

    # get 33rd %ile of `first_echo` and find corresponding index
    # NOTE: percentile is arbitrary
    perc = np.percentile(first_echo, 33, interpolation='higher')
    perc_val = (echo_means[:, 0] == perc)

    # extract values from all echos at relevant index
    # NOTE: threshold of 1/3 voxel value is arbitrary
    lthrs = np.squeeze(echo_means[perc_val].T) / 3

    # if multiple samples were extracted per echo, keep the one w/the highest signal
    if lthrs.ndim > 1:
        lthrs = lthrs[:, lthrs.sum(axis=0).argmax()]

    # determine samples where absolute value is greater than echo-specific thresholds
    # and count # of echos that pass criterion
    masksum = (np.abs(echo_means) > lthrs).sum(axis=-1)

    if mask is None:
        # make it a boolean mask to (where we have at least 1 echo with good signal)
        mask = masksum.astype(bool)
    else:
        # if the user has supplied a binary mask
        mask = load_image(mask).astype(bool)
        masksum = masksum * mask

    if getsum:
        return mask, masksum

    return mask


def make_min_mask(data, roi=None):
    """
    Generates a 3D mask of `data`

    Only samples that are consistently (i.e., across time AND echoes) non-zero
    in `data` are True in output

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    roi : :obj:`str`, optional
        Binary mask for region-of-interest to consider in TE Dependent ANAlysis

    Returns
    -------
    mask : (S,) :obj:`numpy.ndarray`
        Boolean array
    """

    data = np.asarray(data).astype(bool)
    mask = data.prod(axis=-1).prod(axis=-1).astype(bool)

    if roi is None:
        return mask
    else:
        roi = load_image(roi).astype(bool)
        return np.logical_and(mask, roi)


def unmask(data, mask):
    """
    Unmasks `data` using non-zero entries of `mask`

    Parameters
    ----------
    data : (M [x E [x T]]) array_like
        Masked array, where `M` is the number of `True` values in `mask`
    mask : (S,) array_like
        Boolean array of `S` samples that was used to mask `data`. It should
        have exactly `M` True values.

    Returns
    -------
    out : (S [x E [x T]]) :obj:`numpy.ndarray`
        Unmasked `data` array
    """

    out = np.zeros(mask.shape + data.shape[1:], dtype=data.dtype)
    out[mask] = data
    return out


def moments(data):
    """
    Returns gaussian parameters of a 2D distribution by calculating its moments

    Parameters
    ----------
    data : array_like
        2D data array

    Returns
    -------
    height : :obj:`float`
    center_x : :obj:`float`
    center_y : :obj:`float`
    width_x : :obj:`float`
    width_y : :obj:`float`

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
    height : :obj:`float`
    center_x : :obj:`float`
    center_y : :obj:`float`
    width_x : :obj:`float`
    width_y : :obj:`float`

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
    arr1, arr2 : array_like
        Input arrays, arrays to binarize and compare.

    Returns
    -------
    dsi : :obj:`float`
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
    arrs : :obj:`list`
        List of boolean or integer arrays to be summed

    Returns
    -------
    result : :obj:`numpy.ndarray`
        Integer array of summed `arrs`
    """

    # coerce to integer and ensure all arrays are the same shape
    arrs = [check_array(arr, dtype=int, ensure_2d=False, allow_nd=True) for arr in arrs]
    if not np.all([arr1.shape == arr2.shape for arr1 in arrs for arr2 in arrs]):
        raise ValueError('All input arrays must have same shape.')

    # sum across arrays
    result = np.sum(arrs, axis=0)

    return result
