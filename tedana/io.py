"""
Functions to handle file input/output
"""
import json
import logging
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.filename_parser import splitext_addext
from nilearn._utils import check_niimg
from nilearn.image import new_img_like

from tedana import utils
from tedana.stats import computefeats2, get_coeffs

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


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
    rest : (S x T) :obj:`numpy.ndarray`
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


def write_split_ts(data, mmix, mask, comptable, ref_img, out_dir='.', suffix=''):
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
    suffix : :obj:`str`, optional
        Appended to name of saved files (before extension). Default: ''

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
        fout = filewrite(utils.unmask(hikts, mask),
                         op.join(out_dir, 'hik_ts_{0}'.format(suffix)), ref_img)
        LGR.info('Writing high-Kappa time series: {}'.format(op.abspath(fout)))

    if len(rej) != 0:
        fout = filewrite(utils.unmask(lowkts, mask),
                         op.join(out_dir, 'lowk_ts_{0}'.format(suffix)), ref_img)
        LGR.info('Writing low-Kappa time series: {}'.format(op.abspath(fout)))

    fout = filewrite(utils.unmask(dnts, mask),
                     op.join(out_dir, 'dn_ts_{0}'.format(suffix)), ref_img)
    LGR.info('Writing denoised time series: {}'.format(op.abspath(fout)))
    return varexpl


def writefeats(data, mmix, mask, ref_img, out_dir='.', suffix=''):
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
    out_dir : :obj:`str`, optional
        Output directory.
    suffix : :obj:`str`, optional
        Appended to name of saved files (before extension). Default: ''

    Returns
    -------
    fname : :obj:`str`
        Filepath to saved file

    Notes
    -----
    This function writes out a file:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    feats_[suffix].nii        Z-normalized spatial component maps.
    ======================    =================================================
    """

    # write feature versions of components
    feats = utils.unmask(computefeats2(data, mmix, mask), mask)
    fname = filewrite(feats, op.join(out_dir, 'feats_{0}'.format(suffix)), ref_img)
    return fname


def writeresults(ts, mask, comptable, mmix, n_vols, ref_img, out_dir='.'):
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
    out_dir : :obj:`str`, optional
        Output directory.

    Notes
    -----
    This function writes out several files:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
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
    ts_OC.nii                 Optimally combined 4D time series.
    ======================    =================================================
    """
    acc = comptable[comptable.classification == 'accepted'].index.values

    fout = filewrite(ts, op.join(out_dir, 'ts_OC'), ref_img)
    LGR.info('Writing optimally combined time series: {}'.format(op.abspath(fout)))

    write_split_ts(ts, mmix, mask, comptable, ref_img, out_dir=out_dir, suffix='OC')

    ts_B = get_coeffs(ts, mmix, mask)
    fout = filewrite(ts_B, op.join(out_dir, 'betas_OC'), ref_img)
    LGR.info('Writing full ICA coefficient feature set: {}'.format(op.abspath(fout)))

    if len(acc) != 0:
        fout = filewrite(ts_B[:, acc], op.join(out_dir, 'betas_hik_OC'), ref_img)
        LGR.info('Writing denoised ICA coefficient feature set: {}'.format(op.abspath(fout)))
        fout = writefeats(split_ts(ts, mmix, mask, comptable)[0],
                          mmix[:, acc], mask, ref_img, out_dir=out_dir,
                          suffix='OC2')
        LGR.info('Writing Z-normalized spatial component maps: {}'.format(op.abspath(fout)))


def writeresults_echoes(catd, mmix, mask, comptable, ref_img, out_dir='.'):
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
    out_dir : :obj:`str`, optional
        Output directory.

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
        write_split_ts(catd[:, i_echo, :], mmix, mask, comptable, ref_img,
                       out_dir=out_dir, suffix='e%i' % (i_echo + 1))


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


def filewrite(data, filename, ref_img, gzip=True, copy_header=True):
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
    root = op.dirname(filename)
    base = op.basename(filename)
    base, ext, add = splitext_addext(base)
    root = op.join(root, base)
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


def _rem_column_prefix(name):
    """
    Remove column prefix
    """
    return int(name.split('_')[-1])


def _find_comp_rows(name):
    """
    Find component rows
    """
    is_valid = False
    temp = name.split('_')
    if len(temp) == 2 and temp[-1].isdigit():
        is_valid = True
    return is_valid


def save_comptable(df, filename, label='ica', metadata=None):
    """
    Save pandas DataFrame as a BIDS Derivatives-compatible json file.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        DataFrame to save to file.
    filename : :obj:`str`
        File to which to output DataFrame.
    label : :obj:`str`, optional
        Prefix to add to component names in json file. Generally either "ica"
        or "pca".
    metadata : :obj:`dict` or None, optional
        Additional top-level metadata (e.g., decomposition description) to add
        to json file. Default is None.
    """
    save_df = df.copy()

    if 'component' not in save_df.columns:
        save_df['component'] = save_df.index

    # Rename components
    max_value = save_df['component'].max()
    save_df['component'] = save_df['component'].apply(
        add_decomp_prefix, prefix=label, max_value=max_value)
    save_df = save_df.set_index('component')
    save_df = save_df.fillna('n/a')

    data = save_df.to_dict(orient='index')

    if metadata is not None:
        data = {**data, **metadata}

    with open(filename, 'w') as fo:
        json.dump(data, fo, sort_keys=True, indent=4)


def load_comptable(filename):
    """
    Load a BIDS Derivatives decomposition json file into a pandas DataFrame.

    Parameters
    ----------
    filename : :obj:`str`
        File from which to load DataFrame.

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        DataFrame with contents from filename.
    """
    with open(filename, 'r') as fo:
        data = json.load(fo)
    data = {d: data[d] for d in data.keys() if _find_comp_rows(d)}
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.replace('n/a', np.nan)  # our jsons store nans as 'n/a'
    df['component'] = df.index
    df['component'] = df['component'].apply(_rem_column_prefix)
    df = df.set_index('component', drop=True)
    df.index.name = 'component'
    return df
