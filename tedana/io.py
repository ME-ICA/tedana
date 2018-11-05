"""
Functions to handle file input/output
"""
import logging
import os.path as op

import nibabel as nib
import numpy as np
from nibabel.filename_parser import splitext_addext
from nilearn._utils import check_niimg
from nilearn.image import new_img_like
from numpy.linalg import lstsq

from tedana import model, utils

LGR = logging.getLogger(__name__)


def gscontrol_mmix(optcom_ts, mmix, mask, comptable, ref_img):
    """
    Perform global signal regression.

    Parameters
    ----------
    optcom_ts : (S x T) array_like
        Optimally combined time series data
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `optcom_ts`
    mask : (S,) array_like
        Boolean mask array
    acc : :obj:`list`
        Indices of accepted (BOLD) components in `mmix`
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk

    Notes
    -----
    This function writes out several files:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    sphis_hik.nii             T1-like effect.
    hik_ts_OC_T1c.nii         T1 corrected time series.
    dn_ts_OC_T1c.nii          Denoised version of T1 corrected time series
    betas_hik_OC_T1c.nii      T1-GS corrected components
    meica_mix_T1c.1D          T1-GS corrected mixing matrix
    ======================    =================================================
    """
    optcom_masked = optcom_ts[mask, :]
    optcom_mu = optcom_masked.mean(axis=-1)[:, np.newaxis]
    optcom_std = optcom_masked.std(axis=-1)[:, np.newaxis]

    """
    Compute temporal regression
    """
    data_norm = (optcom_masked - optcom_mu) / optcom_std
    cbetas = lstsq(mmix, data_norm.T, rcond=None)[0].T
    resid = data_norm - np.dot(cbetas, mmix.T)

    """
    Build BOLD time series without amplitudes, and save T1-like effect
    """
    acc = comptable.loc[comptable['classification'] == 'accepted', 'component']
    bold_ts = np.dot(cbetas[:, acc], mmix[:, acc].T)
    t1_map = bold_ts.min(axis=-1)
    t1_map -= t1_map.mean()
    filewrite(utils.unmask(t1_map, mask), 'sphis_hik', ref_img)
    t1_map = t1_map[:, np.newaxis]

    """
    Find the global signal based on the T1-like effect
    """
    glob_sig = lstsq(t1_map, data_norm, rcond=None)[0]

    """
    T1-correct time series by regression
    """
    bold_noT1gs = bold_ts - np.dot(lstsq(glob_sig.T, bold_ts.T,
                                         rcond=None)[0].T, glob_sig)
    filewrite(utils.unmask(bold_noT1gs * optcom_std, mask),
              'hik_ts_OC_T1c.nii', ref_img)

    """
    Make denoised version of T1-corrected time series
    """
    medn_ts = optcom_mu + ((bold_noT1gs + resid) * optcom_std)
    filewrite(utils.unmask(medn_ts, mask), 'dn_ts_OC_T1c', ref_img)

    """
    Orthogonalize mixing matrix w.r.t. T1-GS
    """
    mmixnogs = mmix.T - np.dot(lstsq(glob_sig.T, mmix, rcond=None)[0].T,
                               glob_sig)
    mmixnogs_mu = mmixnogs.mean(-1)[:, np.newaxis]
    mmixnogs_std = mmixnogs.std(-1)[:, np.newaxis]
    mmixnogs_norm = (mmixnogs - mmixnogs_mu) / mmixnogs_std
    mmixnogs_norm = np.vstack([np.atleast_2d(np.ones(max(glob_sig.shape))),
                               glob_sig, mmixnogs_norm])

    """
    Write T1-GS corrected components and mixing matrix
    """
    cbetas_norm = lstsq(mmixnogs_norm.T, data_norm.T, rcond=None)[0].T
    filewrite(utils.unmask(cbetas_norm[:, 2:], mask), 'betas_hik_OC_T1c',
              ref_img)
    np.savetxt('meica_mix_T1c.1D', mmixnogs)


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


def write_split_ts(data, mmix, mask, acc, rej, midk, ref_img, suffix=''):
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

    if len(acc) != 0:
        fout = filewrite(utils.unmask(hikts, mask),
                         'hik_ts_{0}'.format(suffix), ref_img)
        LGR.info('Writing high-Kappa time series: {}'.format(op.abspath(fout)))

    if len(midk) != 0:
        fout = filewrite(utils.unmask(midkts, mask),
                         'midk_ts_{0}'.format(suffix), ref_img)
        LGR.info('Writing mid-Kappa time series: {}'.format(op.abspath(fout)))

    if len(rej) != 0:
        fout = filewrite(utils.unmask(lowkts, mask),
                         'lowk_ts_{0}'.format(suffix), ref_img)
        LGR.info('Writing low-Kappa time series: {}'.format(op.abspath(fout)))

    fout = filewrite(utils.unmask(dnts, mask),
                     'dn_ts_{0}'.format(suffix), ref_img)
    LGR.info('Writing denoised time series: {}'.format(op.abspath(fout)))

    return varexpl


def writefeats(data, mmix, mask, ref_img, suffix=''):
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
    feats = utils.unmask(model.computefeats2(data, mmix, mask), mask)
    fname = filewrite(feats, 'feats_{0}'.format(suffix), ref_img)

    return fname


def writeresults(ts, mask, comptable, mmix, n_vols, fixed_seed,
                 acc, rej, midk, empty, ref_img):
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
    fixed_seed: :obj:`int`
        Integer value used in seeding ICA
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

    fout = filewrite(ts, 'ts_OC', ref_img)
    LGR.info('Writing optimally-combined time series: {}'.format(op.abspath(fout)))

    write_split_ts(ts, mmix, mask, acc, rej, midk, ref_img, suffix='OC')

    ts_B = model.get_coeffs(ts, mmix, mask)
    fout = filewrite(ts_B, 'betas_OC', ref_img)
    LGR.info('Writing full ICA coefficient feature set: {}'.format(op.abspath(fout)))

    if len(acc) != 0:
        fout = filewrite(ts_B[:, acc], 'betas_hik_OC', ref_img)
        LGR.info('Writing denoised ICA coefficient feature set: {}'.format(op.abspath(fout)))
        fout = writefeats(split_ts(ts, mmix, mask, acc)[0],
                          mmix[:, acc], mask, ref_img, suffix='OC2')
        LGR.info('Writing Z-normalized spatial component maps: {}'.format(op.abspath(fout)))


def writeresults_echoes(catd, mmix, mask, acc, rej, midk, ref_img):
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
        LGR.info('Writing Kappa-filtered echo #{:01d} timeseries'.format(i_echo+1))
        write_split_ts(catd[:, i_echo, :], mmix, mask, acc, rej, midk, ref_img,
                       suffix='e%i' % (i_echo+1))


def ctabsel(ctabfile):
    """
    Loads a pre-existing component table file

    Parameters
    ----------
    ctabfile : :obj:`str`
        Filepath to existing component table

    Returns
    -------
    ctab : (4,) :obj:`tuple` of :obj:`numpy.ndarray`
        Tuple containing arrays of (1) accepted, (2) rejected, (3) mid, and (4)
        ignored components
    """

    with open(ctabfile, 'r') as src:
        ctlines = src.readlines()
    class_tags = ['#ACC', '#REJ', '#MID', '#IGN']
    class_dict = {}
    for ii, ll in enumerate(ctlines):
        for kk in class_tags:
            if ll[:4] is kk and ll[4:].strip() is not '':
                class_dict[kk] = ll[4:].split('#')[0].split(',')
    return tuple([np.array(class_dict[kk], dtype=int) for kk in class_tags])


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
    ref_img = img.__class__(np.zeros((nx, ny, nz)), affine=img.affine,
                            header=img.header, extra=img.extra)
    ref_img.header.extensions = []
    ref_img.header.set_sform(ref_img.header.get_sform(), code=1)

    return fdata, ref_img
