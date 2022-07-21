"""
Utilities for tedana package
"""
import logging
import os.path as op
import re

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn._utils import check_niimg
from scipy import ndimage
from sklearn.utils import check_array

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
                "{0} voxels in user-defined mask do not have good "
                "signal. Removing voxels from mask.".format(n_bad_voxels)
            )
            masksum[masksum < threshold] = 0
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


def dice(arr1, arr2, axis=None):
    """
    Compute Dice's similarity index between two numpy arrays. Arrays will be
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
        raise ValueError("Axis provided {} not supported by the input arrays.".format(axis))

    arr_sum = arr1.sum(axis=axis) + arr2.sum(axis=axis)
    if np.all(arr_sum == 0):
        dsi = np.zeros(arr_sum.shape)
    else:
        intersection = np.logical_and(arr1, arr2)
        dsi = (2.0 * intersection.sum(axis=axis)) / arr_sum

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
        raise ValueError("All input arrays must have same shape.")

    # sum across arrays
    result = np.sum(arrs, axis=0)

    return result


def get_spectrum(data: np.array, tr: float = 1.0):
    """
    Returns the power spectrum and corresponding frequencies when provided
    with a component time course and repitition time.

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


def find_braces(string):
    """Search a string for matched braces.

    This is used to identify pairs of braces in BibTeX files.
    The outside-most pairs should correspond to BibTeX entries.

    Parameters
    ----------
    string : :obj:`str`
        A long string to search for paired braces.

    Returns
    -------
    :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces.
    """
    toret = {}
    pstack = []

    for idx, char in enumerate(string):
        if char == "{":
            pstack.append(idx)
        elif char == "}":
            if len(pstack) == 0:
                raise IndexError(f"No matching closing parens at: {idx}")

            toret[pstack.pop()] = idx

    if len(pstack) > 0:
        raise IndexError(f"No matching opening parens at: {pstack.pop()}")

    toret = list(toret.items())
    return toret


def reduce_idx(idx_list):
    """Identify outermost brace indices in list of indices.

    The purpose here is to find the brace pairs that correspond to BibTeX entries,
    while discarding brace pairs that appear within the entries
    (e.g., braces around article titles).

    Parameters
    ----------
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces.

    Returns
    -------
    reduced_idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces corresponding to BibTeX entries.
    """
    idx_list2 = [idx_item[0] for idx_item in idx_list]
    idx = np.argsort(idx_list2)
    idx_list = [idx_list[i] for i in idx]

    df = pd.DataFrame(data=idx_list, columns=["start", "end"])

    good_idx = []
    df["within"] = False
    for i, row in df.iterrows():
        df["within"] = df["within"] | ((df["start"] > row["start"]) & (df["end"] < row["end"]))
        if not df.iloc[i]["within"]:
            good_idx.append(i)

    idx_list = [idx_list[i] for i in good_idx]
    return idx_list


def index_bibtex_identifiers(string, idx_list):
    """Identify the BibTeX entry identifier before each entry.

    The purpose of this function is to take the raw BibTeX string and a list of indices of entries,
    starting and ending with the braces of each entry, and then extract the identifier before each.

    Parameters
    ----------
    string : :obj:`str`
        The full BibTeX file, as a string.
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces corresponding to BibTeX entries.

    Returns
    -------
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of BibTeX entries,
        from the starting @ to the final }.
    """
    at_idx = [(a.start(), a.end() - 1) for a in re.finditer("@[a-zA-Z0-9]+{", string)]
    df = pd.DataFrame(at_idx, columns=["real_start", "false_start"])
    df2 = pd.DataFrame(idx_list, columns=["false_start", "end"])
    df = pd.merge(left=df, right=df2, left_on="false_start", right_on="false_start")
    new_idx_list = list(zip(df.real_start, df.end))
    return new_idx_list


def find_citations(description):
    r"""Find citations in a text description.

    It looks for cases of \\citep{} and \\cite{} in a string.

    Parameters
    ----------
    description : :obj:`str`
        Description of a method, optionally with citations.

    Returns
    -------
    all_citations : :obj:`list` of :obj:`str`
        A list of all identifiers for citations.
    """
    paren_citations = re.findall(r"\\citep{([a-zA-Z0-9,/\.]+)}", description)
    intext_citations = re.findall(r"\\cite{([a-zA-Z0-9,/\.]+)}", description)
    inparen_citations = re.findall(r"\\citealt{([a-zA-Z0-9,/\.]+)}", description)
    all_citations = ",".join(paren_citations + intext_citations + inparen_citations)
    all_citations = all_citations.split(",")
    all_citations = sorted(list(set(all_citations)))
    return all_citations


def reduce_references(citations, reference_list):
    """Reduce the list of references to only include ones associated with requested citations.

    Parameters
    ----------
    citations : :obj:`list` of :obj:`str`
        A list of all identifiers for citations.
    reference_list : :obj:`list` of :obj:`str`
        List of all available BibTeX entries.

    Returns
    -------
    reduced_reference_list : :obj:`list` of :obj:`str`
        List of BibTeX entries for citations only.
    """
    reduced_reference_list = []
    for citation in citations:
        citation_found = False
        for reference in reference_list:
            check_string = "@[a-zA-Z]+{" + citation + ","
            if re.match(check_string, reference):
                reduced_reference_list.append(reference)
                citation_found = True
                continue

        if not citation_found:
            LGR.warning(f"Citation {citation} not found.")

    return reduced_reference_list


def get_description_references(description):
    """Find BibTeX references for citations in a methods description.

    Parameters
    ----------
    description : :obj:`str`
        Description of a method, optionally with citations.

    Returns
    -------
    bibtex_string : :obj:`str`
        A string containing BibTeX entries, limited only to the citations in the description.
    """
    bibtex_file = op.join(get_resource_path(), "references.bib")
    with open(bibtex_file, "r") as fo:
        bibtex_string = fo.read()

    braces_idx = find_braces(bibtex_string)
    red_braces_idx = reduce_idx(braces_idx)
    bibtex_idx = index_bibtex_identifiers(bibtex_string, red_braces_idx)
    citations = find_citations(description)
    reference_list = [bibtex_string[start : end + 1] for start, end in bibtex_idx]
    reduced_reference_list = reduce_references(citations, reference_list)

    bibtex_string = "\n".join(reduced_reference_list)
    return bibtex_string
