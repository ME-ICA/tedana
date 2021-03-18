"""
=============================
io module (:mod: `tedana.io`)
=============================

.. currentmodule:: tedana.io

The io module handles most file input and output in the `tedana` workflow,
and simplifies some naming function calls with module globals (see "Globals"
and "Notes" below). Other functions in the module help simplify writing out
data from multiple echoes or write very complex outputs.


Globals
-------
outdir
prefix
convention


Naming Functions
----------------
set_convention
set_prefix
gen_img_name
gen_json_name
gen_tsv_name
add_decomp_prefix


File Writing Functions
----------------------
write_split_ts
writefeats
writeresults
writeresults_echoes
filewrite


File Loading Functions
----------------------
load_data


Helper Functions
----------------
new_nii_like
split_ts

See Also
--------
`tedana.constants`


Notes
-----
The global variables are set by default in the module to guarantee that the
functions that use them won't fail if a workflow API is not used.
However, API calls can override the default settings. Additionally, the
naming functions beginning with "get" all leverage dictionaries defined in
the `constants.py` module, as the definitions are large.
"""
import logging
import os.path as op

import numpy as np
import nibabel as nib
from nilearn._utils import check_niimg
from nilearn.image import new_img_like

from tedana import utils
from tedana.stats import computefeats2, get_coeffs
from .constants import (
    bids, allowed_conventions, img_table, json_table, tsv_table
)


LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

global outdir = '.'
global prefix = ''
global convention = bids   # overridden in API or CLI calls


# Naming Functions
def set_convention(name: str) -> None:
    """Sets the convention for the io module

    Parameters
    ----------
    name: str in ('orig', 'bidsv1.5.0', 'bids')
        The convention name to set this module for

    Notes
    -----
    Uses the `io.convention` module-wide variable

    Raises
    ------
    ValueError if the name is not valid
    """
    if name in allowed_conventions:
        convention = name
    elif name == 'bids':
        convention = bids
    else:
        raise ValueError('Convention %s is invalid' % name)


def set_prefix(pref: str) -> None:
    """Sets the prefix for the io module

    Parameters
    ----------
    pref: str
        The prefix to set for the module. If the prefix is not blank,
        filenames will have the prefix and underscore before all filenames
    """
    if pref:
        pref += '_'
    prefix = pref


def gen_img_name(img_type: str, echo: str = 0) -> str:
    """Generates an image file full path to simplify file output

    Parameters
    ----------
    img_type : str
        The description of the image. Must be a key in constants.img_table
    echo : :obj: `int`
        The echo number of the image.

    Returns
    -------
    The full path for the image name

    Raises
    ------
    KeyError, if an invalid description is supplied or API convention is
        illegal
    ValueError, if an echo is supplied when it shouldn't be

    See Also
    --------
    constants.img_table, a dict for translating various naming types
    """
    if echo:
        img_type += ' split'
    format_string = img_table[img_type][convention]
    if echo and not ('{' in format_string):
        raise ValueError('Echo supplied when not supported!')
    elif echo:
        basename = format_string.format(echo)
    else:
        basename = format_string
    return op.join(outdir, prefix + basename)


def gen_json_name(json_type: str) -> str:
    """Generates a JSON file full path to simplify file output

    Parameters
    ----------
    json_type: str
        The description of the JSON. Must be a key in constants.json_table

    Returns
    -------
    The full path for the JSON name

    Raises
    ------
    KeyError, if an invalid description is supplied or API convention is
        illegal

    See Also
    --------
    constants.json_table, a dict for translating various json naming types
    """
    basename = json_table[json_type][convention]
    return op.join(outdir, prefix + basename + '.json')


def gen_tsv_name(tsv_type: str) -> str:
    """Generates a TSV file full path to simplify file output

    Parameters
    ----------
    tsv_type: str
        The description of the TSV. Must be a key in constants.tsv_table

    Returns
    -------
    The full path for the TSV name

    Raises
    ------
    KeyError, if an invalid description is supplied or API convention is
        illegal

    See Also
    --------
    constants.tsv_table, a dict for translating various tsv naming types
    """
    basename = tsv_table[tsv_type][convention]
    return op.join(outdir, prefix + basename + '.tsv')


def add_decomp_prefix(comp_num, prefix, max_value):
    """
    Create component name with leading zeros matching number of components

    Parameters
    ----------
    comp_num : :obj:`int`
        Component number
    prefix : :obj:`str`
        A prefix to prepend to the component name. An underscore is
        automatically added between the prefix and the component number.
    max_value : :obj:`int`
        The maximum component number in the whole decomposition. Used to
        determine the appropriate number of leading zeros in the component
        name.

    Returns
    -------
    comp_name : :obj:`str`
        Component name in the form <prefix>_<zeros><comp_num>
    """
    n_digits = int(np.log10(max_value)) + 1
    comp_name = '{0:08d}'.format(int(comp_num))
    comp_name = '{0}_{1}'.format(prefix, comp_name[8 - n_digits:])
    return comp_name


# File Writing Functions
def write_split_ts(data, mmix, mask, comptable, ref_img, echo=0):
    """
    Splits `data` into denoised / noise / ignored time series and saves to disk

    Parameters
    ----------
    data : (S x T) array_like
        Input time series
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    out_dir : :obj:`str`, optional
        Output directory.
    echo: :obj: `int`, optional
        Echo number to generate filenames, used by some verbose
        functions. Default 0.

    Returns
    -------
    varexpl : :obj:`float`
        Percent variance of data explained by extracted + retained components

    Notes
    -----
    This function writes out several files:

    ============================    ============================================
    Filename                        Content
    ============================    ============================================
    [prefix]Accepted_bold.nii.gz    High-Kappa time series.
    [prefix]Rejected_bold.nii.gz    Low-Kappa time series.
    [prefix]Denoised_bold.nii.gz    Denoised time series.
    ============================    ============================================
    """
    acc = comptable[comptable.classification == 'accepted'].index.values
    rej = comptable[comptable.classification == 'rejected'].index.values

    # mask and de-mean data
    mdata = data[mask]
    dmdata = mdata.T - mdata.T.mean(axis=0)

    # get variance explained by retained components
    betas = get_coeffs(dmdata.T, mmix, mask=None)
    varexpl = (1 - ((dmdata.T - betas.dot(mmix.T))**2.).sum() /
               (dmdata**2.).sum()) * 100
    LGR.info('Variance explained by ICA decomposition: {:.02f}%'.format(varexpl))

    # create component and de-noised time series and save to files
    hikts = betas[:, acc].dot(mmix.T[acc, :])
    lowkts = betas[:, rej].dot(mmix.T[rej, :])
    dnts = data[mask] - lowkts

    if len(acc) != 0:
        fout = filewrite(
                utils.unmask(hikts, mask), 'high kappa ts', ref_img,
                echo=echo
        )
        LGR.info('Writing high-Kappa time series: {}'.format(op.abspath(fout)))

    if len(rej) != 0:
        fout = filewrite(
                utils.unmask(lowkts, mask), 'low kappa ts', ref_img,
                echo=echo
        )
        LGR.info('Writing low-Kappa time series: {}'.format(op.abspath(fout)))

    fout = filewrite(
            utils.unmask(dnts, mask), 'denoised ts', ref_img, echo=echo
    )
    LGR.info('Writing denoised time series: {}'.format(op.abspath(fout)))
    return varexpl


def writefeats(data, mmix, mask, ref_img):
    """
    Converts `data` to component space with `mmix` and saves to disk

    Parameters
    ----------
    data : (S x T) array_like
        Input time series
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk

    Returns
    -------
    fname : :obj:`str`
        Filepath to saved file

    Notes
    -----
    This function writes out a file:

    =================================    =============================================
    Filename                             Content
    =================================    =============================================
    [prefix]_stat-z_components.nii.gz    Z-normalized spatial component maps.
    =================================    =============================================
    """

    # write feature versions of components
    feats = utils.unmask(computefeats2(data, mmix, mask), mask)
    fname = filewrite(feats, 'z-scored ICA accepted components', ref_img)
    return fname


def writeresults(ts, mask, comptable, mmix, n_vols, ref_img):
    """
    Denoises `ts` and saves all resulting files to disk

    Parameters
    ----------
    ts : (S x T) array_like
        Time series to denoise and save to disk
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Requires at least two columns: "component" and
        "classification".
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    n_vols : :obj:`int`
        Number of volumes in original time series
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk

    Notes
    -----
    This function writes out several files:

    =========================================    =====================================
    Filename                                     Content
    =========================================    =====================================
    desc-optcomAccepted_bold.nii.gz              High-Kappa time series.
    desc-optcomRejected_bold.nii.gz              Low-Kappa time series.
    desc-optcomDenoised_bold.nii.gz              Denoised time series.
    desc-ICA_components.nii.gz                   Spatial component maps for all
                                                 components.
    desc-ICAAccepted_components.nii.gz           Spatial component maps for accepted
                                                 components.
    desc-ICAAccepted_stat-z_components.nii.gz    Z-normalized spatial component maps
                                                 for accepted components.
    =========================================    =====================================

    See Also
    --------
    tedana.io.write_split_ts: Writes out time series files
    tedana.io.writefeats: Writes out component files
    """
    acc = comptable[comptable.classification == 'accepted'].index.values
    write_split_ts(ts, mmix, mask, comptable, ref_img)

    ts_B = get_coeffs(ts, mmix, mask)
    fout = filewrite(ts_B, 'ICA components', ref_img)
    LGR.info('Writing full ICA coefficient feature set: {}'.format(op.abspath(fout)))

    if len(acc) != 0:
        fout = filewrite(ts_B[:, acc], 'ICA accepted components', ref_img)
        LGR.info('Writing denoised ICA coefficient feature set: {}'.format(op.abspath(fout)))

        fout = writefeats(split_ts(ts, mmix, mask, comptable)[0],
                          mmix[:, acc], mask, ref_img)
        LGR.info('Writing Z-normalized spatial component maps: {}'.format(op.abspath(fout)))


def writeresults_echoes(catd, mmix, mask, comptable, ref_img):
    """
    Saves individually denoised echos to disk

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input data time series
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index should be the component number.
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk

    Notes
    -----
    This function writes out several files:

    =====================================    ===================================
    Filename                                 Content
    =====================================    ===================================
    echo-[echo]_desc-Accepted_bold.nii.gz    High-Kappa timeseries for echo
                                             number ``echo``.
    echo-[echo]_desc-Rejected_bold.nii.gz    Low-Kappa timeseries for echo
                                             number ``echo``.
    echo-[echo]_desc-Denoised_bold.nii.gz    Denoised timeseries for echo
                                             number ``echo``.
    =====================================    ===================================

    See Also
    --------
    tedana.io.write_split_ts: Writes out the files.
    """

    for i_echo in range(catd.shape[1]):
        LGR.info('Writing Kappa-filtered echo #{:01d} timeseries'.format(i_echo + 1))
        write_split_ts(
                catd[:, i_echo, :], mmix, mask, comptable, ref_img,
                echo=(i_echo + 1)
        )


def filewrite(data, img_type, ref_img, gzip=True, copy_header=True,
              echo=0):
    """
    Writes `data` to `filename` in format of `ref_img`

    Parameters
    ----------
    data : (S [x T]) array_like
        Data to be saved
    filename : :obj:`str`
        Filepath where data should be saved to
    ref_img : :obj:`str` or img_like
        Reference image
    gzip : :obj:`bool`, optional
        Whether to gzip output (if not specified in `filename`). Only applies
        if output dtype is NIFTI. Default: True
    copy_header : :obj:`bool`, optional
        Whether to copy header from `ref_img` to new image. Default: True
    echo : :obj: `int`, optional
        Indicate the echo index of the data being written.

    Returns
    -------
    name : :obj:`str`
        Path of saved image (with added extensions, as appropriate)
    """

    # get reference image for comparison
    if isinstance(ref_img, list):
        ref_img = ref_img[0]

    # generate out file for saving
    out = new_nii_like(ref_img, data, copy_header=copy_header)

    # FIXME: we only handle writing to nifti right now
    # get root of desired output file and save as nifti image
    root = gen_img_name(img_type, echo=echo)
    name = '{}.{}'.format(root, 'nii.gz' if gzip else 'nii')
    out.to_filename(name)

    return name


# File Loading Functions
def load_data(data, n_echos=None):
    """
    Coerces input `data` files to required 3D array output

    Parameters
    ----------
    data : (X x Y x M x T) array_like or :obj:`list` of img_like
        Input multi-echo data array, where `X` and `Y` are spatial dimensions,
        `M` is the Z-spatial dimensions with all the input echos concatenated,
        and `T` is time. A list of image-like objects (e.g., .nii) are
        accepted, as well
    n_echos : :obj:`int`, optional
        Number of echos in provided data array. Only necessary if `data` is
        array_like. Default: None

    Returns
    -------
    fdata : (S x E x T) :obj:`numpy.ndarray`
        Output data where `S` is samples, `E` is echos, and `T` is time
    ref_img : :obj:`str` or :obj:`numpy.ndarray`
        Filepath to reference image for saving output files or NIFTI-like array
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
            fdata = np.stack([utils.load_image(f) for f in data], axis=1)
            ref_img = check_niimg(data[0])
            ref_img.header.extensions = []
            return np.atleast_3d(fdata), ref_img

    img = check_niimg(data)
    (nx, ny), nz = img.shape[:2], img.shape[2] // n_echos
    fdata = utils.load_image(img.get_fdata().reshape(nx, ny, nz, n_echos, -1, order='F'))
    # create reference image
    ref_img = img.__class__(np.zeros((nx, ny, nz, 1)), affine=img.affine,
                            header=img.header, extra=img.extra)
    ref_img.header.extensions = []
    ref_img.header.set_sform(ref_img.header.get_sform(), code=1)

    return fdata, ref_img


# Helper Functions
def new_nii_like(ref_img, data, affine=None, copy_header=True):
    """
    Coerces `data` into NiftiImage format like `ref_img`

    Parameters
    ----------
    ref_img : :obj:`str` or img_like
        Reference image
    data : (S [x T]) array_like
        Data to be saved
    affine : (4 x 4) array_like, optional
        Transformation matrix to be used. Default: `ref_img.affine`
    copy_header : :obj:`bool`, optional
        Whether to copy header from `ref_img` to new image. Default: True

    Returns
    -------
    nii : :obj:`nibabel.nifti1.Nifti1Image`
        NiftiImage
    """

    ref_img = check_niimg(ref_img)
    newdata = data.reshape(ref_img.shape[:3] + data.shape[1:])
    if '.nii' not in ref_img.valid_exts:
        # this is rather ugly and may lose some information...
        nii = nib.Nifti1Image(newdata, affine=ref_img.affine,
                              header=ref_img.header)
    else:
        # nilearn's `new_img_like` is a very nice function
        nii = new_img_like(ref_img, newdata, affine=affine,
                           copy_header=copy_header)
    nii.set_data_dtype(data.dtype)

    return nii


def split_ts(data, mmix, mask, comptable):
    """
    Splits `data` time series into accepted component time series and remainder

    Parameters
    ----------
    data : (S x T) array_like
        Input data, where `S` is samples and `T` is time
    mmix : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    mask : (S,) array_like
        Boolean mask array
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. Requires at least two columns: "component" and
        "classification".

    Returns
    -------
    hikts : (S x T) :obj:`numpy.ndarray`
        Time series reconstructed using only components in `acc`
    resid : (S x T) :obj:`numpy.ndarray`
        Original data with `hikts` removed
    """
    acc = comptable[comptable.classification == 'accepted'].index.values

    cbetas = get_coeffs(data - data.mean(axis=-1, keepdims=True),
                        mmix, mask)
    betas = cbetas[mask]
    if len(acc) != 0:
        hikts = utils.unmask(betas[:, acc].dot(mmix.T[acc, :]), mask)
    else:
        hikts = None

    resid = data - hikts

    return hikts, resid
