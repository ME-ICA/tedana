"""
Utilities for tedana package
"""
import os.path as op

import nibabel as nib
from nibabel.filename_parser import splitext_addext
from nilearn.image import new_img_like
from nilearn._utils import check_niimg
import numpy as np
from scipy.optimize import leastsq
from sklearn.utils import check_array

from ..due import due, BibTeX

FORMATS = {'.nii': 'NIFTI',
           '.gii': 'GIFTI'}


def get_dtype(data):
    """
    Determines neuroimaging format of `data`

    Parameters
    ----------
    data : list-of-str or str or img_like
        Data to determine format of

    Returns
    -------
    dtype : {'NIFTI', 'GIFTI', 'OTHER'} str
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
    n_echos : int
        Number of echoes

    Returns
    -------
    fmin, fmid, fmax : float
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
        if get_dtype(data) == 'GIFTI':
            fdata = np.column_stack([f.data for f in nib.load(data).darrays])
            return fdata
        elif get_dtype(data) == 'NIFTI':
            data = check_niimg(data).get_data()
    elif isinstance(data, nib.spatialimages.SpatialImage):
        data = check_niimg(data).get_data()

    fdata = data.reshape((-1,) + data.shape[3:]).squeeze()

    return fdata


def load_data(data, n_echos=None):
    """
    Coerces input `data` files to required 3D array output

    Parameters
    ----------
    data : (X x Y x M x T) array_like or list-of-img_like
        Input multi-echo data array, where `X` and `Y` are spatial dimensions,
        `M` is the Z-spatial dimensions with all the input echos concatenated,
        and `T` is time. A list of image-like objects (e.g., .nii or .gii) are
        accepted, as well
    n_echos : int, optional
        Number of echos in provided data array. Only necessary if `data` is
        array_like. Default: None

    Returns
    -------
    fdata : (S x E x T) :obj:`numpy.ndarray`
        Output data where `S` is samples, `E` is echos, and `T` is time
    ref_img : str
        Filepath to reference image for saving output files
    """
    if n_echos is None:
        raise ValueError('Number of echos must be specified. '
                         'Confirm that TE times are provided with the `-e` argument.')

    if isinstance(data, list):
        if len(data) == 1:  # a z-concatenated file was provided
            data = data[0]
        elif len(data) == 2:  # inviable -- need more than 2 echos
            raise ValueError('Cannot run `tedana` with only two echos: '
                             '{}'.format(data))
        else:  # individual echo files were provided (surface or volumetric)
            if get_dtype(data) == 'GIFTI':  # NOTE: only handles .[L/R].func.gii files
                echos = np.array_split(data, n_echos)
                fdata = np.stack([np.vstack([load_image(f) for f in e]) for e in echos], axis=1)
                return np.atleast_3d(fdata), data[0]
            else:
                fdata = np.stack([load_image(f) for f in data], axis=1)
                return np.atleast_3d(fdata), data[0]

    img = check_niimg(data)
    (nx, ny), nz = img.shape[:2], img.shape[2] // n_echos
    fdata = load_image(img.get_data().reshape(nx, ny, nz, n_echos, -1, order='F'))

    # create reference image
    ref_img = img.__class__(np.zeros((nx, ny, nz)), affine=img.affine,
                            header=img.header, extra=img.extra)
    ref_img.header.extensions = []
    ref_img.header.set_sform(ref_img.header.get_sform(), code=1)

    return fdata, ref_img


def make_adaptive_mask(data, minimum=True, getsum=False):
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
    mask : (S, ) :obj:`numpy.ndarray`
        Boolean array of voxels that have sufficient signal in at least one
        echo
    masksum : (S, ) :obj:`numpy.ndarray`
        Valued array indicating the number of echos with sufficient signal in a
        given voxel. Only returned if `getsum = True`
    """

    if minimum:
        return make_min_mask(data)

    # take temporal mean of echos and extract non-zero values in first echo
    echo_means = data.mean(axis=-1)  # temporal mean of echos
    first_echo = echo_means[echo_means[:, 0] != 0, 0]

    # get 33rd %ile of `first_echo` and find corresponding index
    perc_33 = np.percentile(first_echo, 33, interpolation='higher')  # QUESTION: why 33%ile?
    med_val = (echo_means[:, 0] == perc_33)

    # extract values from all echos at relevant index
    lthrs = np.squeeze(echo_means[med_val].T) / 3  # QUESTION: why divide by 3?

    # if multiple samples were extracted per echo, keep the one w/the highest signal
    if lthrs.ndim > 1:
        lthrs = lthrs[:, lthrs.sum(axis=0).argmax()]

    # determine samples where absolute value is greater than echo-specific thresholds
    # and count # of echos that pass criterion
    masksum = (np.abs(echo_means) > lthrs).sum(axis=-1)
    # make it a boolean mask to (where we have at least 1 echo with good signal)
    mask = masksum.astype(bool)

    if getsum:
        return mask, masksum

    return mask


def make_min_mask(data):
    """
    Generates a 3D mask of `data`

    Only samples that are consistently (i.e., across time AND echoes) non-zero
    in `data` are True in output

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time

    Returns
    -------
    mask : (S, ) :obj:`numpy.ndarray`
        Boolean array
    """

    data = np.asarray(data).astype(bool)
    return data.prod(axis=-1).prod(axis=-1).astype(bool)


def filewrite(data, filename, ref_img, gzip=False, copy_header=True,
              copy_meta=False):
    """
    Writes `data` to `filename` in format of `ref_img`

    If `ref_img` dtype is GIFTI, then `data` is assumed to be stacked L/R
    hemispheric and will be split and saved as two files

    Parameters
    ----------
    data : (S [x T]) array_like
        Data to be saved
    filename : str
        Filepath where data should be saved to
    ref_img : str or img_like
        Reference image
    gzip : bool, optional
        Whether to gzip output (if not specified in `filename`). Only applies
        if output dtype is NIFTI. Default: False
    copy_header : bool, optional
        Whether to copy header from `ref_img` to new image. Default: True
    copy_meta : bool, optional
        Whether to copy meta from `ref_img` to new image. Only applies if
        output dtype is GIFTI. Default: False

    Returns
    -------
    name : str
        Path of saved image (with added extensions, as appropriate)
    """

    # get datatype and reference image for comparison
    dtype = get_dtype(ref_img)
    if isinstance(ref_img, list):
        ref_img = ref_img[0]

    # ensure that desired output type (from name) is compatible with `dtype`
    root, ext, add = splitext_addext(filename)
    if ext != '' and FORMATS[ext] != dtype:
        raise ValueError('Cannot write {} data to {} file. Please ensure file'
                         'formats are compatible'.format(dtype, FORMATS[ext]))

    if dtype == 'NIFTI':
        out = new_nii_like(ref_img, data, copy_header=copy_header)
        name = '{}.{}'.format(root, 'nii.gz' if gzip else 'nii')
        out.to_filename(name)
    elif dtype == 'GIFTI':
        # remove possible hemispheric denotations from root
        root = op.join(op.dirname(root), op.basename(root).split('.')[0])
        # save hemispheres separately
        for n, (hdata, hemi) in enumerate(zip(np.split(data, 2, axis=0),
                                              ['L', 'R'])):
            out = new_gii_like(ref_img[n], hdata,
                               copy_header=copy_header,
                               copy_meta=copy_meta)
            name = '{}.{}.func.gii'.format(root, hemi)
            out.to_filename(name)

    return name


def new_nii_like(ref_img, data, affine=None, copy_header=True):
    """
    Coerces `data` into NiftiImage format like `ref_img`

    Parameters
    ----------
    ref_img : str or img_like
        Reference image
    data : (S [x T]) array_like
        Data to be saved
    affine : (4 x 4) array_like, optional
        Transformation matrix to be used. Default: `ref_img.affine`
    copy_header : bool, optional
        Whether to copy header from `ref_img` to new image. Default: True

    Returns
    -------
    nii : :obj:`nibabel.nifti1.Nifti1Image`
        NiftiImage
    """

    ref_img = check_niimg(ref_img)
    nii = new_img_like(ref_img,
                       data.reshape(ref_img.shape[:3] + data.shape[1:]),
                       affine=affine,
                       copy_header=copy_header)
    nii.set_data_dtype(data.dtype)

    return nii


def new_gii_like(ref_img, data, copy_header=True, copy_meta=False):
    """
    Coerces `data` into GiftiImage format like `ref_img`

    Parameters
    ----------
    ref_img : str or img_like
        Reference image
    data : (S [x T]) array_like
        Data to be saved
    copy_header : bool, optional
        Whether to copy header from `ref_img` to new image. Default: True
    copy_meta : bool, optional
        Whether to copy meta from `ref_img` to new image. Default: False

    Returns
    -------
    gii : :obj:`nibabel.gifti.gifti.GiftiImage`
        GiftiImage
    """

    if isinstance(ref_img, str):
        ref_img = nib.load(ref_img)

    if data.ndim == 1:
        data = np.atleast_2d(data).T

    darrays = [make_gii_darray(ref_img.darrays[n], d, copy_meta=copy_meta)
               for n, d in enumerate(data.T)]
    gii = nib.gifti.GiftiImage(header=ref_img.header if copy_header else None,
                               extra=ref_img.extra,
                               meta=ref_img.meta if copy_meta else None,
                               labeltable=ref_img.labeltable,
                               darrays=darrays)

    return gii


def make_gii_darray(ref_array, data, copy_meta=False):
    """
    Converts `data` into GiftiDataArray format like `ref_array`

    Parameters
    ----------
    ref_array : str or img_like
        Reference array
    data : (S,) array_like
        Data to be saved
    copy_meta : bool, optional
        Whether to copy meta from `ref_img` to new image. Default: False

    Returns
    -------
    gii : :obj:`nibabel.gifti.gifti.GiftiDataArray`
        Output data array instance
    """

    if not isinstance(ref_array, nib.gifti.GiftiDataArray):
        raise TypeError('Provided reference is not a GiftiDataArray.')
    darray = nib.gifti.GiftiDataArray(data,
                                      intent=ref_array.intent,
                                      datatype=data.dtype,
                                      encoding=ref_array.encoding,
                                      endian=ref_array.endian,
                                      coordsys=ref_array.coordsys,
                                      ordering=ref_array.ind_ord,
                                      meta=ref_array.meta if copy_meta else None)

    return darray


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
