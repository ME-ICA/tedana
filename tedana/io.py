"""
Functions to handle file input/output
"""
import logging
import os.path as op
from enum import Enum, unique

import numpy as np
import nibabel as nib
from nibabel.filename_parser import splitext_addext
from nilearn._utils import check_niimg
from nilearn.image import new_img_like

from tedana import utils
from tedana.stats import computefeats2, get_coeffs

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

outdir = '.'
prefix = ''
convention = 'kundu'


img_table = {
    'adaptive mask': ('adaptive_mask', 'desc-adaptiveGoodSignal_mask'),
    't2star map': ('t2sv', 'T2starmap'),
    's0 map': ('s0v', 'S0map'),
    'combined': ('ts_OC', 'desc-optcom_bold'),
    'z-scored PCA components': ('pca_components',
        'desc-PCA_stat-z_components'),
    'z-scored ICA components': ('betas_OC',
        'desc-ICA_stat-z_components'),
    'ICA accepted components': ('betas_hik_OC', 
        'desc-ICAAccepted_components'),
    'z-scored ICA accepted components': (
        'feats_OC2',
        'desc-ICAAccepted_stat-z_components'),
    'z-scored ICA components': ('ica_components',
        'desc-ICA_stat-z_components'),
    'denoised ts': ('dn_ts_OC', 'desc-optcomDenoised_bold'),
    'high kappa ts': ('hik_ts_OC', 'desc-optcomAccepted_bold'),
    'low kappa ts': ('lowk_ts_OC', 'desc-optcomRejected_bold'),
    # verbose outputs
    'full t2star map': ('t2svG', 'desc-full_T2starmap'),
    'full s0 map': ('s0vG', 'desc-full_S0map'),
    'whitened': ('ts_OC_whitened', 'desc-optcomPCAReduced_bold'),
    'echo weight PCA map split': ('e{0}_PCA_comp',
        'echo-{0}_desc-PCA_components'),
    'echo R2 PCA split': ('e{0}_PCA_R2',
        'echo-{0}_desc-PCAR2ModelPredictions_components'),
    'echo S0 PCA split': ('e{0}_PCA_S0',
        'echo-{0}_desc-PCAS0ModelPredictions_components'),
    'PCA component weights': ('pca_weights',
        'desc-PCAAveragingWeights_components'),
    'PCA reduced': ('oc_reduced', 'desc-optcomPCAReduced_bold'),
    'echo weight ICA map split': ('e{0}_ICA_comp',
        'echo-{0}_desc-ICA_components'),
    'echo R2 ICA split': ('e{0}_ICA_R2',
        'echo-{0}_desc-ICAR2ModelPredictions_components'),
    'echo S0 ICA split': ('e{0}_ICA_S0',
        'echo-{0}_desc-ICAS0ModelPredictions_components'),
    'ICA component weights': ('ica_weights',
        'desc-ICAAveragingWeights_components'),
    'high kappa ts split': ('hik_ts_e{0}',
        'echo-{0}_desc-Accepted_bold'),
    'low kappa ts split': ('lowk_ts_e{0}',
        'echo-{0}_desc-Rejected_bold'),
    'denoised ts split': ('dn_ts_e{0}', 'echo-{0}_desc-Denoised_bold'),
}


def gen_img_name(img_type, echo=0):
    """
    Generates an image file full path to simplify file output

    Parameters
    ----------
    img_type : str
        The description of the image. Must be a key in io.img_table
    echo : :obj: `int`
        The echo number of the image.
    
    Returns
    -------
    The full path for the image name

    Raises
    ------
    KeyError, if an invalid description is supplied
    RuntimeError, if io has no convention set
    ValueError, if an echo is supplied when it shouldn't be

    See Also
    --------
    io.img_table, a dict for translating various naming types
    """

    if convention == 'kundu':
        tuple_idx = 0
    elif convention == 'bids':
        tuple_idx = 1
    else:
        if convention:
            raise RuntimeError('Naming convention %s not supported' %
                               convention
            )
        else:
            raise RuntimeError('No naming convention given!')
    if echo:
        img_type += ' split'
    format_string = img_table[img_type][tuple_idx]
    if echo and not ('{' in format_string):
        raise ValueError('Echo supplied when not supported!')
    elif echo:
        basename = format_string.format(echo)
    else:
        basename = format_string
    return op.join(outdir, prefix + basename)
    

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
    fout = filewrite(ts_B, 'z-scored ICA components', ref_img)
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
        write_split_ts(catd[:, i_echo, :], mmix, mask, comptable, 
                ref_img, echo=(i_echo + 1)
        )


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
