"""Utilities for tedana package."""

import logging
import os.path as op
import platform
import sys
import warnings

import nibabel as nib
import numpy as np
from bokeh import __version__ as bokeh_version
from mapca import __version__ as mapca_version
from matplotlib import __version__ as matplotlib_version
from nibabel import __version__ as nibabel_version
from nilearn import __version__ as nilearn_version
from nilearn._utils import check_niimg
from numpy import __version__ as numpy_version
from pandas import __version__ as pandas_version
from scipy import __version__ as scipy_version
from scipy import ndimage
from sklearn import __version__ as sklearn_version
from sklearn.utils import check_array
from threadpoolctl import __version__ as threadpoolctl_version

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def reshape_niimg(data):
    """Take input `data` and return a sample x time array.

    Parameters
    ----------
    data : (X x Y x Z [x T]) array_like or img_like object
        Data array or data file to be loaded and reshaped

    Returns
    -------
    fdata : (S [x T]) :obj:`numpy.ndarray`
        Reshaped `data`, where `S` is samples and `T` is time
    """
    if isinstance(data, (str, nib.spatialimages.SpatialImage)):
        data = check_niimg(data).get_fdata()
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"Unsupported type {type(data)}")

    fdata = data.reshape((-1,) + data.shape[3:]).squeeze()

    return fdata


def make_adaptive_mask(data, mask=None, getsum=False, threshold=1):
    """
    Make map of `data` specifying longest echo a voxel can be sampled with.

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
    threshold : :obj:`int`, optional
        Minimum echo count to retain in the mask. Default is 1, which is
        equivalent not thresholding.

    Returns
    -------
    mask : (S,) :obj:`numpy.ndarray`
        Boolean array of voxels that have sufficient signal in at least one
        echo
    masksum : (S,) :obj:`numpy.ndarray`
        Valued array indicating the number of echos with sufficient signal in a
        given voxel. Only returned if `getsum = True`
    """
    RepLGR.info(
        "An adaptive mask was then generated, in which each voxel's "
        "value reflects the number of echoes with 'good' data."
    )

    # take temporal mean of echos and extract non-zero values in first echo
    echo_means = data.mean(axis=-1)  # temporal mean of echos
    first_echo = echo_means[echo_means[:, 0] != 0, 0]

    # get 33rd %ile of `first_echo` and find corresponding index
    # NOTE: percentile is arbitrary
    # TODO: "interpolation" param changed to "method" in numpy 1.22.0
    #       confirm method="higher" is the same as interpolation="higher"
    #       Current minimum version for numpy in tedana is 1.16 where
    #       there is no "method" parameter. Either wait until we bump
    #       our minimum numpy version to 1.22 or add a version check
    #       or try/catch statement.
    perc = np.percentile(first_echo, 33, interpolation="higher")
    perc_val = echo_means[:, 0] == perc

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
        # make it a boolean mask to (where we have at least `threshold` echoes with good signal)
        mask = (masksum >= threshold).astype(bool)
        masksum[masksum < threshold] = 0
    else:
        # if the user has supplied a binary mask
        mask = reshape_niimg(mask).astype(bool)
        masksum = masksum * mask
        # reduce mask based on masksum
        # TODO: Use visual report to make checking the reduced mask easier
        if np.any(masksum[mask] < threshold):
            n_bad_voxels = np.sum(masksum[mask] < threshold)
            LGR.warning(
                f"{n_bad_voxels} voxels in user-defined mask do not have good "
                "signal. Removing voxels from mask."
            )
            masksum[masksum < threshold] = 0
            mask = masksum.astype(bool)

    if getsum:
        return mask, masksum

    return mask


def unmask(data, mask):
    """
    Unmasks `data` using non-zero entries of `mask`.

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


def dice(arr1, arr2, axis=None):
    """
    Compute Dice's similarity index between two numpy arrays.

    Arrays will be
    binarized before comparison.

    This method was first proposed in :footcite:t:`dice1945measures` and
    :footcite:t:`sorensen1948method`.

    Parameters
    ----------
    arr1, arr2 : array_like
        Input arrays, arrays to binarize and compare.
    axis : None or int, optional
        Axis along which the DSIs are computed.
        The default is to compute the DSI of the flattened arrays.

    Returns
    -------
    dsi : :obj:`float`
        Dice-Sorenson index.

    Notes
    -----
    This implementation was based on
    https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137.

    References
    ----------
    .. footbibliography::
    """
    arr1 = np.array(arr1 != 0).astype(int)
    arr2 = np.array(arr2 != 0).astype(int)

    if arr1.shape != arr2.shape:
        raise ValueError("Shape mismatch: arr1 and arr2 must have the same shape.")

    if axis is not None and axis > (arr1.ndim - 1):
        raise ValueError(f"Axis provided {axis} not supported by the input arrays.")

    arr_sum = arr1.sum(axis=axis) + arr2.sum(axis=axis)
    intersection = np.logical_and(arr1, arr2)
    # Count number of zero-elements in the denominator and report
    total_zeros = np.count_nonzero(arr_sum == 0)
    if total_zeros > 0:
        LGR.warning(
            f"{total_zeros} of {arr_sum.size} components have empty maps, resulting in Dice "
            "values of 0. "
            "Please check your component table for dice columns with 0-values."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="invalid value encountered in true_divide"
        )
        dsi = (2.0 * intersection.sum(axis=axis)) / arr_sum
    dsi = np.nan_to_num(dsi)

    return dsi


def andb(arrs):
    """
    Sum arrays in `arrs`.

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
        raise ValueError("All input arrays must have same shape.")

    # sum across arrays
    result = np.sum(arrs, axis=0)

    return result


def get_spectrum(data: np.array, tr: float = 1.0):
    """
    Return the power spectrum and corresponding frequencies.

    Done when provided with a component time course and repitition time.

    Parameters
    ----------
    data : (S, ) array_like
            A timeseries S, on which you would like to perform an fft.
    tr : :obj:`float`
            Reptition time (TR) of the data
    """
    # adapted from @dangom
    power_spectrum = np.abs(np.fft.rfft(data)) ** 2
    freqs = np.fft.rfftfreq(power_spectrum.size * 2 - 1, tr)
    idx = np.argsort(freqs)
    return power_spectrum[idx], freqs[idx]


def threshold_map(img, min_cluster_size, threshold=None, mask=None, binarize=True, sided="bi"):
    """
    Cluster-extent threshold and binarize image.

    Parameters
    ----------
    img : img_like or array_like
        Image object or 3D array to be clustered
    min_cluster_size : int
        Minimum cluster size (in voxels)
    threshold : float or None, optional
        Cluster-defining threshold for img. If None (default), assume img is
        already thresholded.
    mask : (S,) array_like or None, optional
        Boolean array for masking resultant data array. Default is None.
    binarize : bool, optional
        Default is True.
    sided : {'bi', 'two', 'one'}, optional
        How to apply thresholding. One-sided thresholds on the positive side.
        Two-sided thresholds positive and negative values together. Bi-sided
        thresholds positive and negative values separately. Default is 'bi'.

    Returns
    -------
    clust_thresholded : (M) :obj:`numpy.ndarray`
        Cluster-extent thresholded (and optionally binarized) map.
    """
    if not isinstance(img, np.ndarray):
        arr = img.get_fdata()
    else:
        arr = img.copy()

    if mask is not None:
        mask = mask.astype(bool)
        arr *= mask.reshape(arr.shape)

    if binarize:
        clust_thresholded = np.zeros(arr.shape, bool)
    else:
        clust_thresholded = np.zeros(arr.shape, int)

    if sided == "two":
        test_arr = np.abs(arr)
    else:
        test_arr = arr.copy()

    # Positive values (or absolute values) first
    if threshold is not None:
        thresh_arr = test_arr >= threshold
    else:
        thresh_arr = test_arr > 0

    # 6 connectivity
    struc = ndimage.generate_binary_structure(3, 1)
    labeled, _ = ndimage.label(thresh_arr, struc)
    unique, counts = np.unique(labeled, return_counts=True)
    clust_sizes = dict(zip(unique, counts))
    clust_sizes = {k: v for k, v in clust_sizes.items() if v >= min_cluster_size}
    for i_clust in clust_sizes.keys():
        if np.all(thresh_arr[labeled == i_clust] == 1):
            if binarize:
                clust_thresholded[labeled == i_clust] = True
            else:
                clust_thresholded[labeled == i_clust] = arr[labeled == i_clust]

    # Now negative values *if bi-sided*
    if sided == "bi":
        if threshold is not None:
            thresh_arr = test_arr <= (-1 * threshold)
        else:
            thresh_arr = test_arr < 0

        labeled, _ = ndimage.label(thresh_arr, struc)
        unique, counts = np.unique(labeled, return_counts=True)
        clust_sizes = dict(zip(unique, counts))
        clust_sizes = {k: v for k, v in clust_sizes.items() if v >= min_cluster_size}
        for i_clust in clust_sizes.keys():
            if np.all(thresh_arr[labeled == i_clust] == 1):
                if binarize:
                    clust_thresholded[labeled == i_clust] = True
                else:
                    clust_thresholded[labeled == i_clust] = arr[labeled == i_clust]

    # reshape to (S,)
    clust_thresholded = clust_thresholded.ravel()

    # if mask provided, mask output
    if mask is not None:
        clust_thresholded = clust_thresholded[mask]

    return clust_thresholded


def sec2millisec(arr):
    """
    Convert seconds to milliseconds.

    Parameters
    ----------
    arr : array_like
        Values in seconds.

    Returns
    -------
    array_like
        Values in milliseconds.
    """
    return arr * 1000


def millisec2sec(arr):
    """
    Convert milliseconds to seconds.

    Parameters
    ----------
    arr : array_like
        Values in milliseconds.

    Returns
    -------
    array_like
        Values in seconds.
    """
    return arr / 1000.0


def setup_loggers(logname=None, repname=None, quiet=False, debug=False):
    """Set up loggers for tedana.

    Parameters
    ----------
    logname : str, optional
        Name of log file, by default None
    repname : str, optional
        Name of report file, by default None
    quiet : bool, optional
        Whether to suppress logging to console, by default False
    debug : bool, optional
        Whether to set logging level to debug, by default False
    """
    # Set up the general logger
    log_formatter = logging.Formatter(
        "%(asctime)s\t%(module)s.%(funcName)-12s\t%(levelname)-8s\t%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_formatter = logging.Formatter(
        "%(levelname)-8s %(module)s:%(funcName)s:%(lineno)d %(message)s"
    )
    # set up general logging file and open it for writing
    if logname:
        log_handler = logging.FileHandler(logname)
        log_handler.setFormatter(log_formatter)
        LGR.addHandler(log_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    LGR.addHandler(stream_handler)

    if quiet:
        LGR.setLevel(logging.WARNING)
    elif debug:
        LGR.setLevel(logging.DEBUG)
    else:
        LGR.setLevel(logging.INFO)

    # Loggers for report and references
    text_formatter = logging.Formatter("%(message)s")
    if repname:
        rep_handler = logging.FileHandler(repname)
        rep_handler.setFormatter(text_formatter)
        RepLGR.setLevel(logging.INFO)
        RepLGR.addHandler(rep_handler)
        RepLGR.propagate = False


def teardown_loggers():
    """Close loggers."""
    for local_logger in (RepLGR, LGR):
        for handler in local_logger.handlers[:]:
            handler.close()
            local_logger.removeHandler(handler)


def get_resource_path():
    """Return the path to general resources, terminated with separator.

    Resources are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), "resources") + op.sep)


def get_system_version_info():
    """
    Return information about the system tedana is being run on.

    Returns
    -------
    dict
        Info about system where tedana is run on and
        and python and python library versioning info for key
        modules used by tedana.
    """
    system_info = platform.uname()

    python_libraries = {
        "bokeh": bokeh_version,
        "matplotlib": matplotlib_version,
        "mapca": mapca_version,
        "nibabel": nibabel_version,
        "nilearn": nilearn_version,
        "numpy": numpy_version,
        "pandas": pandas_version,
        "scikit-learn": sklearn_version,
        "scipy": scipy_version,
        "threadpoolctl": threadpoolctl_version,
    }

    return {
        "System": system_info.system,
        "Node": system_info.node,
        "Release": system_info.release,
        "Version": system_info.version,
        "Machine": system_info.machine,
        "Processor": system_info.processor,
        "Python": sys.version,
        "Python_Libraries": python_libraries,
    }
