"""
Functions to handle file input/output
"""
import re
import logging
import os.path as op

import numpy as np
from numpy.linalg import lstsq

from tedana import model, utils

LGR = logging.getLogger(__name__)


def generate_fname(basefile, extension='.nii.gz', **kwargs):
    """
    Generate BIDS derivatives-compatible filename from components.

    Parameters
    ----------
    basefile : :obj:`str`
        Name of file from which to derive BIDSRawBase prefix and datatype
        suffix.
    extension : :obj:`str`, optional
        Extension for file. Default is ".nii.gz".
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

    # Remove echo field from filename
    echo_regex = re.compile('_echo-[0-9+]_')
    temp = re.sub(echo_regex, '_', basefile)

    # Get prefix
    prefix = temp[:temp.rfind('_')]

    # Grab data type (should be "_bold")
    suffix = temp[temp.rfind('_'):]
    suffix = suffix.split('.')[0]

    # Check extension
    if not extension.startswith('.'):
        extension = '.' + extension

    # Create string with description-field pairs
    add_str = ''
    for k, v in kwargs.items():
        add_str += f'_{k}-{v}'

    out_file = prefix+add_str+suffix+extension
    return out_file


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
    utils.filewrite(utils.unmask(t1_map, mask), 'sphis_hik', ref_img)
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
    utils.filewrite(utils.unmask(bold_noT1gs * optcom_std, mask),
                    'hik_ts_OC_T1c.nii', ref_img)

    """
    Make denoised version of T1-corrected time series
    """
    medn_ts = optcom_mu + ((bold_noT1gs + resid) * optcom_std)
    utils.filewrite(utils.unmask(medn_ts, mask), 'dn_ts_OC_T1c', ref_img)

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
    utils.filewrite(utils.unmask(cbetas_norm[:, 2:], mask), 'betas_hik_OC_T1c',
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
        fout = utils.filewrite(utils.unmask(hikts, mask),
                               'hik_ts_{0}'.format(suffix), ref_img)
        LGR.info('Writing high-Kappa time series: {}'.format(op.abspath(fout)))

    if len(midk) != 0:
        fout = utils.filewrite(utils.unmask(midkts, mask),
                               'midk_ts_{0}'.format(suffix), ref_img)
        LGR.info('Writing mid-Kappa time series: {}'.format(op.abspath(fout)))

    if len(rej) != 0:
        fout = utils.filewrite(utils.unmask(lowkts, mask),
                               'lowk_ts_{0}'.format(suffix), ref_img)
        LGR.info('Writing low-Kappa time series: {}'.format(op.abspath(fout)))

    fout = utils.filewrite(utils.unmask(dnts, mask),
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
    fname = utils.filewrite(feats, 'feats_{0}'.format(suffix), ref_img)

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

    fout = utils.filewrite(ts, 'ts_OC', ref_img)
    LGR.info('Writing optimally-combined time series: {}'.format(op.abspath(fout)))

    write_split_ts(ts, mmix, mask, acc, rej, midk, ref_img, suffix='OC')

    ts_B = model.get_coeffs(ts, mmix, mask)
    fout = utils.filewrite(ts_B, 'betas_OC', ref_img)
    LGR.info('Writing full ICA coefficient feature set: {}'.format(op.abspath(fout)))

    if len(acc) != 0:
        fout = utils.filewrite(ts_B[:, acc], 'betas_hik_OC', ref_img)
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
