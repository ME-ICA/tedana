"""Utilities for tedana package."""

import logging
import os.path as op
import platform
import sys
import warnings
from typing import Union

import nibabel as nib
import numpy as np
import numpy.typing as npt
from bokeh import __version__ as bokeh_version
from mapca import __version__ as mapca_version
from matplotlib import __version__ as matplotlib_version
from nibabel import __version__ as nibabel_version
from nilearn import __version__ as nilearn_version
from nilearn._utils import check_niimg
from numpy import __version__ as numpy_version
from pandas import __version__ as pandas_version
from robustica import __version__ as robustica_version
from scipy import __version__ as scipy_version
from scipy import ndimage
from scipy.special import lpmv
from sklearn import __version__ as sklearn_version
from sklearn.utils import check_array
from threadpoolctl import __version__ as threadpoolctl_version
from tqdm import __version__ as tqdm_version

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


def make_adaptive_mask(data, mask, n_independent_echos=None, threshold=1, methods=["dropout"]):
    """Make map of `data` specifying longest echo a voxel can be sampled with.

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is time.
    mask : :obj:`str` or img_like
        Binary mask for voxels to consider in TE Dependent ANAlysis.
        This must be provided, as the mask is used to identify exemplar voxels.
        Without a mask limiting the voxels to consider,
        the adaptive mask will generally select voxels outside the brain as exemplars.
    n_independent_echos : :obj:`int`, optional
        Number of independent echoes to use in goodness of fit metrics (fstat).
        Primarily used for EPTI acquisitions.
        If None, number of echoes will be used. Default is None.
    threshold : :obj:`int`, optional
        Minimum echo count to retain in the mask.
        Default is 1, which is equivalent to not thresholding.
    methods : :obj:`list`, optional
        List of methods to use for adaptive mask generation. Default is ["dropout"].
        Valid methods are "decay", "dropout", and "none".

    Returns
    -------
    mask : (S,) :obj:`numpy.ndarray`
        Boolean array of voxels that have sufficient signal in at least ``threshold`` echos.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Valued array indicating the number of echos with sufficient signal in a given voxel.

    Notes
    -----
    The adaptive mask can flag "bad" echoes via two methods: dropout and decay.
    Either or both methods are applied to the mean magnitude across time for each voxel and echo.

    Dropout

    Remove voxels with relatively low mean magnitudes from the mask.

    This method uses distributions of values across the mask.
    Therefore, it is sensitive to the quality of the mask.
    A bad mask may result in a bad adaptive mask.

    This method is implemented as follows:

    a.  Calculate the 33rd percentile of values in the first echo,
        based on voxel-wise mean over time.
    b.  Identify the voxel where the first echo's mean value is equal to the 33rd percentile.
        Basically, this identifies "exemplar" voxel reflecting the 33rd percentile.

        -   The 33rd percentile is arbitrary.
        -   If more than one voxel has a value exactly equal to the 33rd percentile,
            keep all of them.
    c.  For the exemplar voxel from the first echo, calculate 1/3 of the mean value for each echo.

        -   This is the threshold for "good" data.
        -   The 1/3 value is arbitrary.
        -   If there was more than one exemplar voxel, retain the the highest value for each echo.
    d.  For each voxel, identify the last echo with a mean value greater than the
        corresponding echo's threshold.

        -   Preceding echoes (including ones with mean values less than the threshold)
            are considered "good" data. That means, if echoes 1-3 in a voxel are
            [good, good, bad] the adaptive mask will assign a 2, and if they are
            [good, bad, good], the adaptive mask will assign a 3.

    Decay

    Determine the echo at which the signal stops decreasing for each voxel.
    If a voxel's signal stops decreasing as echo time increases, then we can infer that the
    voxel has either fully dephased (i.e., "bottomed out") or been contaminated by noise.
    This essentially identifies the last echo with "good" data.
    For a scan that collects many echoes for T2* estimation or has a relatively short echo
    spacing, it is possible that a later echo will have a higher value,
    but the overall trend still shows a decay.
    This method should not be used in those situations.

    The element-wise minimum value between any selected methods is used to construct the adaptive
    mask.
    """
    RepLGR.info(
        f"An adaptive mask was then generated using the {'+'.join(methods)} method(s), "
        "in which each voxel's value reflects the number of echoes with 'good' data."
    )
    mask = reshape_niimg(mask).astype(bool)
    data = data[mask, :, :]

    if (methods is None) or (len(methods) == 1 and methods[0].lower() == "none"):
        LGR.warning(
            "No methods provided for adaptive mask generation. "
            "Only removing voxels with negative or NaN values"
        )
        RepLGR.info(
            "An adaptive mask was then generated that retained echoes with negative or NaN values."
        )
    else:
        RepLGR.info(
            f"An adaptive mask was then generated using the {'+'.join(methods)} method(s), "
            "in which each voxel's value reflects the number of echoes with 'good' data."
        )
    assert all([method.lower() in ["decay", "dropout", "none"] for method in methods])

    n_samples, n_echos, _ = data.shape
    adaptive_masks = []

    # Generate a base adaptive mask that flags any NaN, zero, or negative values
    bad_data_vals = np.isnan(data) + (data <= 0)
    good_vox_echoes = 1 - np.any(bad_data_vals, axis=-1).astype(int)
    base_adaptive_mask = np.zeros(n_samples, dtype=int)
    for echo_idx in range(n_echos):
        # For voxels that were in the mask for the immediately previous echo
        # If they are still good in the current echo, increment the adaptive
        # mask value
        base_adaptive_mask[
            (base_adaptive_mask == (echo_idx)) * (good_vox_echoes[:, echo_idx] == 1)
        ] = (echo_idx + 1)

    adaptive_masks.append(base_adaptive_mask)

    if ("dropout" in methods) or ("decay" in methods):
        echo_means = data.mean(axis=-1)  # temporal mean of echos

    if "dropout" in methods:
        # take temporal mean of echos and extract non-zero values in first echo
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

        LGR.info("Echo-wise intensity thresholds for adaptive mask: %s", lthrs)

        # Find the last good echo for each voxel
        # Start with every voxel's value==0, increment up the echoes, and
        # change to a new value every time a later good echo is found
        dropout_adaptive_mask = np.zeros(n_samples, dtype=np.int16)
        for echo_idx in range(n_echos):
            dropout_adaptive_mask[(np.abs(echo_means[:, echo_idx]) > lthrs[echo_idx])] = (
                echo_idx + 1
            )

        adaptive_masks.append(dropout_adaptive_mask)

        adaptive_masks.append(dropout_adaptive_mask)

    if "decay" in methods:
        # Determine where voxels stop decreasing in signal from echo to echo
        echo_diffs = np.hstack((np.full((n_samples, 1), -1), np.diff(echo_means, axis=1)))
        diff_mask = echo_diffs >= 0  # flag where signal is not decreasing
        last_decreasing_echo = diff_mask.argmax(axis=1)
        last_decreasing_echo[last_decreasing_echo == 0] = n_echos  # if no increase, set to n_echos
        adaptive_masks.append(last_decreasing_echo)

    # Retain the most conservative of the selected adaptive mask estimates
    adaptive_mask = np.minimum.reduce(adaptive_masks)

    # TODO: Use visual report to make checking the reduced mask easier
    if np.any(adaptive_mask < threshold):
        n_bad_voxels = np.sum(adaptive_mask < threshold)
        LGR.warning(
            f"{n_bad_voxels} voxels in user-defined mask do not have good signal. "
            "Removing voxels from mask."
        )
        adaptive_mask[adaptive_mask < threshold] = 0

    if isinstance(n_independent_echos, int):
        # For EPTI sequences, the way we use adaptive mask thresholding fails
        # because sequential echoes have overlapping information and
        # there is no clear mapping between the independent sources and the echoes.
        # Since EPTI has less dropout, it is unclear how often this will cause issues.
        # To track this, we are flagging voxels that might mark less independent signal.
        # If such voxels appear often, this would show we might need to alter how the mask is used.
        # The thresh where there might not be 3 independent sources of data within the good echoes
        # For n_echos=100 & n_independent_echos=3, threshold_dof=66.
        # For n_echos=100 & n_independent_echos=4, threshold_dof=50
        threshold_3dof = np.floor(2 * n_echos / n_independent_echos)
        n_3dof_voxels = np.sum(np.logical_and(adaptive_mask < threshold_3dof, adaptive_mask >= 1))
        perc_3dof_voxels = 100 * n_3dof_voxels / np.sum(adaptive_mask >= 1)
        if perc_3dof_voxels > 0:
            LGR.warning(
                f"{n_3dof_voxels} voxels ({np.round(perc_3dof_voxels, decimals=1)}%) have fewer "
                f"than {np.round(threshold_3dof)} "
                "good voxels. These voxels will be used in all analyses, "
                "but might not include 3 independent echo measurements."
            )

        # There's a separate warning about DOF if it's possible there's a DOF reduction.
        if n_independent_echos > 3:
            # The threshold where the loss of good echoes might affect the DOF
            # For n_echos=100 & n_independent_echos=4, threshold_dof=75
            threshold_dof = np.floor((n_independent_echos - 1) * n_echos / n_independent_echos)
            n_dof_voxels = np.sum(
                np.logical_and(adaptive_mask < threshold_dof, adaptive_mask >= 1)
            )
            perc_dof_voxels = 100 * n_dof_voxels / np.sum(adaptive_mask >= 1)
            LGR.warning(
                f"{n_dof_voxels} voxels ({np.round(perc_dof_voxels, decimals=1)}%) have fewer "
                f"than {np.round(threshold_dof)} good voxels. "
                f"The degrees of freedom for fits across echoes will remain {n_independent_echos} "
                f"even if there might be fewer independent echo measurements."
            )
    modified_mask = adaptive_mask.astype(bool)
    adaptive_mask = unmask(adaptive_mask, mask)
    modified_mask = unmask(modified_mask, mask)

    return modified_mask, adaptive_mask


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


def create_legendre_polynomial_basis_set(
    n_vols: int, dtrank: Union[int, None] = None
) -> npt.NDArray:
    """
    Create Legendre polynomial basis set for detrending time series.

    Parameters
    ----------
    n_vols : :obj:`int`
        The number of time points in the fMRI time series
    dtrank : :obj:`int`, optional
        Specifies degree of Legendre polynomial basis function for estimating
        spatial global signal. Default: None
        If None, then this is set to 1+floor(n_vols/150)

    Returns
    -------
    legendre_arr : (T X R) :obj:`np.ndarray`
        A time by rank matrix of the first dtrank order Legendre polynomials.
        These include:
        Order 0: y = 1
        Order 1: y = x
        Order 2: y = 0.5*(3*x^2 - 1)
        Order 3: y = 0.5*(5*x^3 - 3*x)
        Order 4: y = 0.125*(35*x^4 - 30*x^2 + 3)
        Order 5: y = 0.125*(63*x^5 - 70*x^3 + 15x)
    """
    if dtrank is None:
        dtrank = int(1 + np.floor(n_vols / 150))

    bounds = np.linspace(-1, 1, n_vols)
    legendre_arr = np.column_stack([lpmv(0, vv, bounds) for vv in range(dtrank)])

    return legendre_arr


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
        "robustica": robustica_version,
        "scikit-learn": sklearn_version,
        "scipy": scipy_version,
        "threadpoolctl": threadpoolctl_version,
        "tqdm": tqdm_version,
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


def check_te_values(te_values):
    """Check if all TE values are in ms by checking if they are higher than 1.

    Parameters
    ----------
    te_values : list
        TE values to check.

    Returns
    -------
    list
        TE values in milliseconds.
    """
    te_values = np.array(te_values)
    if all(te_values > 1):
        return te_values.tolist()
    elif all((te_values > 0) & (te_values < 1)):
        # Raise a warning and convert to ms by multiplying by 1000
        LGR.warning("Assuming the provided TE values are in seconds. Converting to ms.")
        return (te_values * 1000).tolist()
    else:
        raise ValueError("TE values must be positive and in milliseconds.")


def log_newsletter_info():
    """Log information about the tedana newsletter."""
    # Add log encouraging users to subscribe to the tedana newsletter
    LGR.info(
        "Don't forget to subscribe to the tedana newsletter for updates! "
        "This is a very low volume email list."
    )
    LGR.info("https://groups.google.com/g/tedana-newsletter")
