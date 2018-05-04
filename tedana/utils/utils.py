"""Utilities for meica package"""
import numpy as np
import nibabel as nib
from nibabel.filename_parser import splitext_addext
from nilearn._utils import check_niimg
import nilearn.masking as nimask
from scipy.optimize import leastsq

from ..due import due, BibTeX


def load_image(data):
    """
    Takes input `data` and returns a sample x time array

    Parameters
    ----------
    data : (X x Y x Z [x T]) array_like or niimg-like object
        Data array or data file to be loaded / reshaped

    Returns
    -------
    fdata : (S x T) np.ndarray
        Reshaped `data`, where `S` is samples and `T` is time
    """

    if isinstance(data, str):
        root, ext, addext = splitext_addext(data)
        if ext == '.gii':
            fdata = np.column_stack([f.data for f in nib.load(data).darrays])
            return fdata
        else:
            data = check_niimg(data).get_data()

    fdata = data.reshape((-1,) + data.shape[3:], order='F')

    return fdata.squeeze()


def cat2echos(data, n_echos=None):
    """
    Coerces input `data` files to required 3D array output

    Parameters
    ----------
    data : (X x Y x M x T) array_like or list-of-img-like
        Input multi-echo data array, where `X` and `Y` are spatial dimensions,
        `M` is the Z-spatial dimensions with all the input echos concatenated,
        and `T` is time. A list of image-like objects (e.g., .nii or .gii) are
        accepted, as well
    n_echos : int, optional
        Number of echos in provided data array. Only necessary if `data` is
        array_like. Default: None

    Returns
    -------
    fdata : (S x E x T) np.ndarray
        Output data where `S` is samples, `E` is echos, and `T` is time
    """

    # data files were provided
    if isinstance(data, list):
        # individual echo files were provided
        if len(data) > 2:
            fdata = np.stack([load_image(f) for f in data], axis=1)
            if fdata.ndim < 3:
                fdata = fdata[..., np.newaxis]
            return fdata
        # a z-concatenated file was provided; load data and pipe it down
        elif len(data) == 1:
            if n_echos is None:
                raise ValueError('Number of echos `n_echos` must be specified '
                                 'if z-concatenated data file provided.')
            data = check_niimg(data[0]).get_data()
        # only two echo files were provided, which doesn't fly
        else:
            raise ValueError('Cannot run `tedana` with only two echos: '
                             '{}'.format(data))

    (nx, ny), nz = data.shape[:2], data.shape[2] // n_echos
    fdata = load_image(data.reshape(nx, ny, nz, n_echos, -1, order='F'))

    return fdata


def makeadmask(data, minimum=True, getsum=False):
    """
    Makes map of `data` specifying longest echo a voxel can be sampled with

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    minimum : bool, optional
        Use `make_min_mask()` instead of generating a map with echo-specific
        times. Default: True
    getsum : bool, optional
        Return `masksum` in addition to `mask`. Default: False

    Returns
    -------
    mask : (S, ) np.ndarray
        Boolean array of voxels that have sufficient signal in at least one
        echo
    masksum : (S, ) np.ndarray
        Valued array indicating the number of echos with sufficient signal in a
        given voxel. Only returned if `getsum = True`
    """

    if minimum:
        return make_min_mask(data)

    n_samp, n_echos, n_vols = data.shape
    echo_means = data.mean(axis=-1)  # temporal mean of echos
    first_echo = echo_means[..., 0]
    # make a map of longest echo with which a voxel can be sampled, with min
    # value of map as X value of voxel that has median value in the 1st echo
    # N.B. larger factor (%ile??) leads to bias to lower TEs
    perc_33 = np.percentile(first_echo[first_echo.nonzero()], 33,
                            interpolation='higher')  # why take 33rd %ile?
    med_val = (first_echo == perc_33)
    lthrs = np.vstack([echo_means[..., echo][med_val] / 3 for echo in
                       range(n_echos)])  # why divide by three?
    lthrs = lthrs[:, lthrs.sum(0).argmax()]
    mthr = np.ones(data.shape[:-1])
    for echo in range(n_echos):
        mthr[..., echo] *= lthrs[echo]

    masksum = (np.abs(echo_means) > mthr).astype('int').sum(axis=-1)
    mask = (masksum != 0)

    if getsum:
        return mask, masksum

    return mask


def make_min_mask(data):
    """
    Generates a 3D mask of `data`

    Only voxels that are consistently (i.e., across time AND echoes) non-zero
    in `data` are True in output

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time

    Returns
    -------
    mask : (S, ) np.ndarray
        Boolean array
    """

    data = np.asarray(data).astype(bool)
    return data.prod(axis=-1).prod(axis=-1).astype(bool)


def get_input_type(input):
    pass


def niwrite(data, affine, name, head, outtype='.nii.gz'):
    """
    Write out nifti file.

    Parameters
    ----------
    data : array_like
    affine : (4 x 4) array_like
        Affine for output file
    name : str
        Name to save output file to
    head : object
    outtype : str, optional
        Output type of file. Default: '.nii.gz'
    """

    # get rid of NaN
    data[np.isnan(data)] = 0
    # set header info
    header = head.copy()
    header.set_data_shape(list(data.shape))
    outni = nib.Nifti1Image(data, affine, header=header)
    outni.set_data_dtype('float64')
    outni.to_filename(name)


def uncat2echos(data):
    """
    Combines Z- and echo-axis in `data`

    Parameters
    ----------
    data : (X x Y x Z x E x T) array_like
        Multi-echo data array

    Returns
    -------
    fdata : (X x Y x M x T) np.ndarray
        Z-concatenated multi-echo data array, where M is Z * number of echos
    """

    if data.ndim < 4:
        raise ValueError('Input data must have at least four dimensions; '
                         'provided data has only {0}'.format(data.ndim))

    (nx, ny), nz = data.shape[:2], np.prod(data.shape[2:4])
    return data.reshape(nx, ny, nz, -1, order='F')


def fmask(data, mask=None):
    """
    Masks `data` with non-zero entries of `mask`

    Parameters
    ----------
    data : (X x Y x Z [x E [x T]) array_like or niimg-like object
        Data array or data file to be masked
    mask : (X x Y x Z) array_like or niimg-like object
        Boolean array or mask file

    Returns
    -------
    fdata : (S x E x T) np.ndarray
        Masked `data`, where `S` is samples, `E` is echoes, and `T` is time
    """

    if mask is not None and not type(data) == type(mask):
        raise TypeError('Provided `data` and `mask` must be of same type.')

    if isinstance(data, str):
        root, ext, addext = splitext_addext(data)
        if ext == '.gii':
            # mask need not apply for gii files
            fdata = np.column_stack([f.data for f in nib.load(data).darrays])
        else:
            # use nilearn for other files
            data = check_niimg(data)
            if mask is not None:
                # TODO: check that this uses same order to flatten
                fdata = nimask.apply_mask(data, mask).T
            else:
                fdata = data.get_data().reshape((-1,) + data.shape[3:])
    elif isinstance(data, np.ndarray):
        # flatten data over first three dimensions and apply mask
        fdata = data.reshape((-1,) + data.shape[3:])
        if mask is not None:
            fdata = fdata[mask.flatten() > 0]

    return fdata.squeeze()


def unmask(data, mask):
    """
    Unmasks `data` using non-zero entries of `mask`

    Parameters
    ----------
    data : (M x E x T) array_like
        Masked array, where `M` is the number of samples
    mask : (S,) array_like
        Boolean array of `S` samples that was used to mask `data`

    Returns
    -------
    out : (S x E x T) np.ndarray
        Unmasked `data` array
    """

    out = np.zeros((mask.shape + data.shape[1:]))
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
