"""
Utilities for tedana package
"""
import logging

import numpy as np
from scipy import stats
import nibabel as nib
from nilearn._utils import check_niimg
from sklearn.utils import check_array

from tedana.due import due, BibTeX

LGR = logging.getLogger(__name__)


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


def make_adaptive_mask(data, mask=None, getsum=False):
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
    masksum = np.ones(data.shape[0]) * data.shape[1]

    echo_means = data.mean(axis=-1)  # temporal mean of echos

    # New, coarse method for identifying bad echoes
    prev_echo_means = echo_means[:, 0]
    for i_echo in range(1, data.shape[1]):
        cur_echo_means = echo_means[:, i_echo]
        unindexed_voxels = masksum == data.shape[1]
        # If a voxel's signal goes up from one echo to next, that's bad
        bad_voxels = cur_echo_means > prev_echo_means
        bad_voxels = np.logical_and(bad_voxels, unindexed_voxels)
        masksum[bad_voxels] = i_echo
        prev_echo_means = cur_echo_means

    # Old, fine method for identifying bad echoes
    # take temporal mean of echos and extract non-zero values in first echo
    first_echo = echo_means[echo_means[:, 0] != 0, 0]

    # get 33rd %ile of first_echo and find corresponding index
    # NOTE: percentile is arbitrary
    perc = np.percentile(first_echo, 33, interpolation='higher')
    perc_val_idx = echo_means[:, 0] == perc

    # extract values from all echos at relevant index
    # NOTE: threshold of 1/3 voxel value is arbitrary
    lthrs = np.squeeze(echo_means[perc_val_idx, ...].T) / 3

    # if multiple samples were extracted per echo, keep the one w/the highest signal
    if lthrs.ndim > 1:
        lthrs = lthrs[:, lthrs.sum(axis=0).argmax()]

    # determine samples where absolute value is greater than echo-specific thresholds
    # and count # of echos that pass criterion
    temp_masksum = (np.abs(echo_means) > lthrs).sum(axis=-1)

    # combine results from coarse and fine methods
    masksum = np.minimum(masksum, temp_masksum)

    # Finalize masks
    if mask is None:
        # make it a boolean mask to (where we have at least 1 echo with good signal)
        mask = masksum.astype(bool)
    else:
        # if the user has supplied a binary mask
        mask = load_image(mask).astype(bool)
        masksum = masksum * mask
        # reduce mask based on masksum
        # TODO: Use visual report to make checking the reduced mask easier
        if np.any(masksum[mask] == 0):
            n_bad_voxels = np.sum(masksum[mask] == 0)
            LGR.warning('{0} voxels in user-defined mask do not have good '
                        'signal. Removing voxels from mask.'.format(n_bad_voxels))
            mask = masksum.astype(bool)

    if getsum:
        return mask, masksum

    return mask


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
