"""
Functions to handle file input/output
"""
import re
import json
import logging
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.filename_parser import splitext_addext
from nilearn._utils import check_niimg
from nilearn.image import new_img_like

from tedana import model, utils

LGR = logging.getLogger(__name__)


def gen_fname(basefile, extension='_bold.nii.gz', **kwargs):
    """
    Generate BIDS derivatives-compatible filename from components.

    Parameters
    ----------
    basefile : :obj:`str`
        Name of file from which to derive BIDSRawBase prefix and datatype
        suffix.
    extension : :obj:`str`, optional
        Extension for file. Includes the datatype suffix. Default is
        "_bold.nii.gz".
    kwargs : :obj:`str`
        Additional keyword arguments are added to the filename in the order
        they appear.

    Returns
    -------
    out_file : :obj:`str`
        BIDS derivatives-compatible filename
    """
    if not all([isinstance(v, str) for k, v in kwargs.items()]):
        raise ValueError("All keyword arguments must be strings")

    bf_dir, bf_name = op.split(basefile)
    # Remove echo field from filename
    echo_regex = re.compile('_echo-[0-9+]_')
    temp = re.sub(echo_regex, '_', bf_name)

    # Remove duplicate keys
    for key in kwargs.keys():
        key_regex = re.compile('_{0}-[a-zA-Z0-9]+_'.format(key))
        temp = re.sub(key_regex, '_', temp)

    # Get prefix
    prefix = temp[:temp.rfind('_')]

    # Grab data type (should be "_bold")
    if not extension.startswith('_'):
        extension = '_' + extension

    # Create string with description-field pairs
    add_str = ''

    # Add echo first if provided
    if 'echo' in kwargs.keys():
        add_str += '_echo-{0}'.format(kwargs['echo'])
        del kwargs['echo']

    # Add rest of fields
    for k, v in kwargs.items():
        add_str += '_{0}-{1}'.format(k, v)

    out_file = op.join(bf_dir, prefix + add_str + extension)
    return out_file


def save_comptable(df, filename):
    """
    Save pandas DataFrame as a json file.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame to save to file.
    filename : :obj:`str`
        File to which to output DataFrame.
    """
    if 'component' in df.columns:
        data = df.set_index('component').to_dict(orient='index')
    else:
        data = df.to_dict(orient='index')

    with open(filename, 'w') as fo:
        json.dump(data, fo, sort_keys=True, indent=4)


def load_comptable(filename):
    """
    Load pandas DataFrame from json file.

    Parameters
    ----------
    filename : :obj:`str`
        File from which to load DataFrame.

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        DataFrame with contents from filename.
    """
    df = pd.read_json(filename, orient='index')
    df.index.name = 'component'
    return df


def split_ts(data, mmix, mask, acc):
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
    acc : :obj:`list`
        List of accepted components used to subset `mmix`

    Returns
    -------
    hikts : (S x T) :obj:`numpy.ndarray`
        Time series reconstructed using only components in `acc`
    rest : (S x T) :obj:`numpy.ndarray`
        Original data with `hikts` removed
    """

    cbetas = model.get_coeffs(data - data.mean(axis=-1, keepdims=True),
                              mmix, mask)
    betas = cbetas[mask]
    if len(acc) != 0:
        hikts = utils.unmask(betas[:, acc].dot(mmix.T[acc, :]), mask)
    else:
        hikts = None

    resid = data - hikts

    return hikts, resid


def write_split_ts(data, mmix, mask, acc, rej, midk, ref_img, bf, **kwargs):
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
    acc : :obj:`list`
        Indices of accepted (BOLD) components in `mmix`
    rej : :obj:`list`
        Indices of rejected (non-BOLD) components in `mmix`
    midk : :obj:`list`
        Indices of mid-K (questionable) components in `mmix`
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    bf : :obj:`str`
        Base filename for outputs.

    Returns
    -------
    varexpl : :obj:`float`
        Percent variance of data explained by extracted + retained components

    Notes
    -----
    This function writes out several files:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    hik_ts_[suffix].nii       High-Kappa time series.
    midk_ts_[suffix].nii      Mid-Kappa time series.
    low_ts_[suffix].nii       Low-Kappa time series.
    dn_ts_[suffix].nii        Denoised time series.
    ======================    =================================================
    """

    # mask and de-mean data
    mdata = data[mask]
    dmdata = mdata.T - mdata.T.mean(axis=0)

    # get variance explained by retained components
    betas = model.get_coeffs(dmdata.T, mmix, mask=None)
    varexpl = (1 - ((dmdata.T - betas.dot(mmix.T))**2.).sum() /
               (dmdata**2.).sum()) * 100
    LGR.info('Variance explained by ICA decomposition: '
             '{:.02f}%'.format(varexpl))

    # create component and de-noised time series and save to files
    hikts = betas[:, acc].dot(mmix.T[acc, :])
    midkts = betas[:, midk].dot(mmix.T[midk, :])
    lowkts = betas[:, rej].dot(mmix.T[rej, :])
    dnts = data[mask] - lowkts - midkts

    if 'desc' in kwargs.keys():
        prefix = 'optcom'
        del kwargs['desc']
    else:
        prefix = None

    if len(acc) != 0:
        if prefix:
            fout = gen_fname(bf, desc='{0}Accepted'.format(prefix))
        else:
            fout = gen_fname(bf, desc='accepted', **kwargs)
        filewrite(utils.unmask(hikts, mask), fout, ref_img)
        LGR.info('Writing high-Kappa time series: {}'.format(fout))

    if len(midk) != 0:
        if prefix:
            fout = gen_fname(bf, desc='{0}RejectedMidK'.format(prefix))
        else:
            fout = gen_fname(bf, desc='rejectedMidK', **kwargs)
        filewrite(utils.unmask(midkts, mask), fout, ref_img)
        LGR.info('Writing mid-Kappa time series: {}'.format(fout))

    if len(rej) != 0:
        if prefix:
            fout = gen_fname(bf, desc='{0}Rejected'.format(prefix))
        else:
            fout = gen_fname(bf, desc='rejected', **kwargs)
        filewrite(utils.unmask(lowkts, mask), fout, ref_img)
        LGR.info('Writing low-Kappa time series: {}'.format(fout))

    if prefix:
        fout = gen_fname(bf, desc='{0}Denoised'.format(prefix))
    else:
        fout = gen_fname(bf, desc='denoised', **kwargs)
    filewrite(utils.unmask(dnts, mask), fout, ref_img)
    LGR.info('Writing denoised time series: {}'.format(fout))

    return varexpl


def writefeats(data, mmix, mask, ref_img, bf):
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
    bf : :obj:`str`
        Base filename for outputs.

    Returns
    -------
    fname : :obj:`str`
        Filepath to saved file

    Notes
    -----
    This function writes out a file:

    ========================================    ===============================
    Filename                                    Content
    ========================================    ===============================
    *_desc-TEDICAAcceptedZ_components.nii.gz    Z-normalized spatial component
                                                maps.
    ========================================    ===============================
    """

    # write feature versions of components
    fname = gen_fname(bf, '_components.nii.gz', desc='TEDICAAcceptedZ')
    LGR.info('Writing Z-normalized spatial maps for accepted components: '
             '{}'.format(fname))
    feats = utils.unmask(model.computefeats2(data, mmix, mask), mask)
    filewrite(feats, fname, ref_img)

    return fname


def writeresults(ts, mask, comptable, mmix, n_vols, acc, rej, midk, empty,
                 ref_img, bf):
    """
    Denoises `ts` and saves all resulting files to disk

    Parameters
    ----------
    ts : (S x T) array_like
        Time series to denoise and save to disk
    mask : (S,) array_like
        Boolean mask array
    comptable : (N x 5) array_like
        Array with columns denoting (1) index of component, (2) Kappa score of
        component, (3) Rho score of component, (4) variance explained by
        component, and (5) normalized variance explained by component
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data`
    n_vols : :obj:`int`
        Number of volumes in original time series
    acc : :obj:`list`
        Indices of accepted (BOLD) components in `mmix`
    rej : :obj:`list`
        Indices of rejected (non-BOLD) components in `mmix`
    midk : :obj:`list`
        Indices of mid-K (questionable) components in `mmix`
    empty : :obj:`list`
        Indices of ignored components in `mmix`
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    bf : :obj:`str`
        Base filename for outputs.

    Notes
    -----
    This function writes out several files:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    ts_OC.nii                 Optimally combined 4D time series.
    hik_ts_OC.nii             High-Kappa time series. Generated by
                              :py:func:`tedana.utils.io.write_split_ts`.
    midk_ts_OC.nii            Mid-Kappa time series. Generated by
                              :py:func:`tedana.utils.io.write_split_ts`.
    low_ts_OC.nii             Low-Kappa time series. Generated by
                              :py:func:`tedana.utils.io.write_split_ts`.
    dn_ts_OC.nii              Denoised time series. Generated by
                              :py:func:`tedana.utils.io.write_split_ts`.
    betas_OC.nii              Full ICA coefficient feature set.
    betas_hik_OC.nii          Denoised ICA coefficient feature set.
    feats_OC2.nii             Z-normalized spatial component maps. Generated
                              by :py:func:`tedana.utils.io.writefeats`.
    comp_table.txt            Component table. Generated by
                              :py:func:`tedana.utils.io.writect`.
    ======================    =================================================
    """

    fout = gen_fname(bf, desc='optcom')
    filewrite(ts, fout, ref_img)
    LGR.info('Writing optimally-combined time series: {}'.format(fout))

    write_split_ts(ts, mmix, mask, acc, rej, midk, ref_img, bf, desc='optcom')

    ts_B = model.get_coeffs(ts, mmix, mask)
    fout = gen_fname(bf, '_components.nii.gz', desc='TEDICA')
    filewrite(ts_B, fout, ref_img)
    LGR.info('Writing full ICA coefficient feature set: {}'.format(fout))

    if len(acc) != 0:
        fout = gen_fname(bf, '_components.nii.gz', desc='TEDICAAccepted')
        filewrite(ts_B[:, acc], fout, ref_img)
        LGR.info('Writing denoised ICA coefficient feature set: {}'.format(fout))
        hikts = split_ts(ts, mmix, mask, acc)[0]
        writefeats(hikts, mmix[:, acc], mask, ref_img, bf)


def writeresults_echoes(catd, mmix, mask, acc, rej, midk, ref_img, bf):
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
    acc : :obj:`list`
        Indices of accepted (BOLD) components in `mmix`
    rej : :obj:`list`
        Indices of rejected (non-BOLD) components in `mmix`
    midk : :obj:`list`
        Indices of mid-K (questionable) components in `mmix`
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    bf : :obj:`str`
        Base filename for outputs.

    Notes
    -----
    This function writes out several files:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    hik_ts_e[echo].nii        High-Kappa timeseries for echo number ``echo``.
                              Generated by
                              :py:func:`tedana.utils.io.write_split_ts`.
    midk_ts_e[echo].nii       Mid-Kappa timeseries for echo number ``echo``.
                              Generated by
                              :py:func:`tedana.utils.io.write_split_ts`.
    lowk_ts_e[echo].nii       Low-Kappa timeseries for echo number ``echo``.
                              Generated by
                              :py:func:`tedana.utils.io.write_split_ts`.
    dn_ts_e[echo].nii         Denoised timeseries for echo number ``echo``.
                              Generated by
                              :py:func:`tedana.utils.io.write_split_ts`.
    ======================    =================================================
    """

    for i_echo in range(catd.shape[1]):
        LGR.info('Writing Kappa-filtered echo #{:01d} timeseries'.format(i_echo + 1))
        write_split_ts(catd[:, i_echo, :], mmix, mask, acc, rej, midk, ref_img,
                       bf, echo=str(i_echo + 1))


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


def filewrite(data, filename, ref_img, gzip=False, copy_header=True):
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
        if output dtype is NIFTI. Default: False
    copy_header : :obj:`bool`, optional
        Whether to copy header from `ref_img` to new image. Default: True

    Returns
    -------
    name : :obj:`str`
        Path of saved image (with added extensions, as appropriate)
    """

    # get reference image for comparison
    if isinstance(ref_img, list):
        ref_img = ref_img[0]

    if filename.endswith('gz'):
        gzip = True

    # generate out file for saving
    out = new_nii_like(ref_img, data, copy_header=copy_header)

    # FIXME: we only handle writing to nifti right now
    # get root of desired output file and save as nifti image
    root, ext, add = splitext_addext(filename)
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
    fdata = utils.load_image(img.get_data().reshape(nx, ny, nz, n_echos, -1, order='F'))
    # create reference image
    ref_img = img.__class__(np.zeros((nx, ny, nz, 1)), affine=img.affine,
                            header=img.header, extra=img.extra)
    ref_img.header.extensions = []
    ref_img.header.set_sform(ref_img.header.get_sform(), code=1)

    return fdata, ref_img
