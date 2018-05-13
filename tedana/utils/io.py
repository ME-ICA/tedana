"""
Functions to handle file input/output
"""
import textwrap
import numpy as np
import os.path as op
from tedana import model, utils

import logging
logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
lgr = logging.getLogger(__name__)


def gscontrol_mmix(OCcatd, mmix, mask, acc, rej, midk, ref_img):
    """
    Parameters
    ----------
    OCcatd : (S x T) array_like
        Optimally-combined time series data
    mmix : (C x T) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `OCcatd`
    mask : (S,) array_like
        Boolean mask array
    acc : list
        Indices of accepted (BOLD) components in `mmix`
    rej : list
        Indices of rejected (non-BOLD) components in `mmix`
    midk : list
        Indices of mid-K (questionable) components in `mmix`
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    """

    Gmu = OCcatd.mean(axis=-1)
    Gstd = OCcatd.std(axis=-1)
    Gmask = (Gmu != 0)

    """
    Compute temporal regression
    """
    dat = (OCcatd[Gmask] - Gmu[Gmask][:, np.newaxis]) / Gstd[mask][:, np.newaxis]
    solG = np.linalg.lstsq(mmix, dat.T)[0]
    resid = dat - np.dot(solG.T, mmix.T)

    """
    Build BOLD time series without amplitudes, and save T1-like effect
    """
    bold_ts = np.dot(solG.T[:, acc], mmix[:, acc].T)
    sphis = bold_ts.min(axis=-1)
    sphis -= sphis.mean()
    lgr.info(sphis.shape)
    utils.utils.filewrite(utils.utils.unmask(sphis, mask), 'sphis_hik', ref_img)

    """
    Find the global signal based on the T1-like effect
    """
    sol = np.linalg.lstsq(np.atleast_2d(sphis).T, dat)
    glsig = sol[0]

    """
    T1 correct time series by regression
    """
    bold_noT1gs = bold_ts - np.dot(np.linalg.lstsq(glsig.T, bold_ts.T)[0].T, glsig)
    utils.utils.filewrite(utils.unmask(bold_noT1gs * Gstd[mask][:, np.newaxis], mask),
                          'hik_ts_OC_T1c.nii', ref_img)

    """
    Make medn version of T1 corrected time series
    """
    utils.filewrite(Gmu[..., np.newaxis] +
                    utils.unmask((bold_noT1gs+resid)*Gstd[mask][:, np.newaxis], mask),
                    'dn_ts_OC_T1c', ref_img)

    """
    Orthogonalize mixing matrix w.r.t. T1-GS
    """
    mmixnogs = mmix.T - np.dot(np.linalg.lstsq(glsig.T, mmix)[0].T, glsig)
    mmixnogs_mu = mmixnogs.mean(-1)
    mmixnogs_std = mmixnogs.std(-1)
    mmixnogs_norm = (mmixnogs - mmixnogs_mu[:, np.newaxis]) / mmixnogs_std[:, np.newaxis]
    mmixnogs_norm = np.vstack([np.atleast_2d(np.ones(max(glsig.shape))), glsig, mmixnogs_norm])

    """
    Write T1-GS corrected components and mixing matrix
    """
    sol = np.linalg.lstsq(mmixnogs_norm.T, dat.T)
    utils.filewrite(utils.unmask(sol[0].T[:, 2:], mask), 'betas_hik_OC_T1c', ref_img)
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
    acc : list
        List of accepted components used to subset `mmix`

    Returns
    -------
    hikts : (S x T) np.ndarray
        Time series reconstructed using only components in `acc`
    rest : (S x T) np.ndarray
        Original data with `hikts` removed
    """

    cbetas = model.get_coeffs(data - data.mean(axis=-1, keepdims=True), mask, mmix)
    betas = cbetas[mask]
    if len(acc) != 0:
        hikts = utils.unmask(betas[:, acc].dot(mmix.T[acc, :]), mask)
    else:
        hikts = None

    return hikts, data - hikts


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
    acc : list
        Indices of accepted (BOLD) components in `mmix`
    rej : list
        Indices of rejected (non-BOLD) components in `mmix`
    midk : list
        Indices of mid-K (questionable) components in `mmix`
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    suffix : str, optional
        Appended to name of saved files (before extension). Default: ''

    Returns
    -------
    varexpl : float
        Percent variance of data explained by extracted + retained components
    """

    # mask and de-mean data
    mdata = data[mask]
    dmdata = mdata.T - mdata.T.mean(axis=0)

    # get variance explained by retained components
    betas = model.get_coeffs(utils.unmask(dmdata.T, mask), mask, mmix)[mask]
    varexpl = (1 - ((dmdata.T - betas.dot(mmix.T))**2.).sum() / (dmdata**2.).sum()) * 100
    lgr.info('++ Variance explained: {:.02f}%'.format(varexpl))

    # create component and de-noised time series and save to files
    hikts = betas[:, acc].dot(mmix.T[acc, :])
    midkts = betas[:, midk].dot(mmix.T[midk, :])
    lowkts = betas[:, rej].dot(mmix.T[rej, :])
    dnts = data[mask] - lowkts - midkts
    if len(acc) != 0:
        utils.filewrite(utils.unmask(hikts, mask), 'hik_ts_{0}'.format(suffix), ref_img)
    if len(midk) != 0:
        utils.filewrite(utils.unmask(midkts, mask), 'midk_ts_{0}'.format(suffix), ref_img)
    if len(rej) != 0:
        utils.filewrite(utils.unmask(lowkts, mask), 'lowk_ts_{0}'.format(suffix), ref_img)
    utils.filewrite(utils.unmask(dnts, mask), 'dn_ts_{0}'.format(suffix), ref_img)

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
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    suffix : str, optional
        Appended to name of saved files (before extension). Default: ''

    Returns
    -------
    fname : str
        Filepath to saved file
    """

    # write feature versions of components
    feats = utils.unmask(model.computefeats2(data, mmix, mask), mask)
    fname = utils.filewrite(feats, 'feats_{0}'.format(suffix), ref_img)

    return fname


def writect(comptable, n_vols, acc, rej, midk, empty, ctname='comp_table.txt', varexpl='-1'):
    """
    Saves component table to disk

    Parameters
    ----------
    comptable : (N x 5) array_like
        Array with columns denoting (1) index of component, (2) Kappa score of
        component, (3) Rho score of component, (4) variance explained by
        component, and (5) normalized variance explained by component
    n_vols : int
        Number of volumes in original time series
    acc : list
        Indices of accepted (BOLD) components in `mmix`
    rej : list
        Indices of rejected (non-BOLD) components in `mmix`
    midk : list
        Indices of mid-K (questionable) components in `mmix`
    empty : list
        Indices of ignored components in `mmix`
    ctname : str, optional
        Filename to save comptable to disk. Default 'comp_table.txt'
    varexpl : str
        Variance explained by original data
    """

    n_components = comptable.shape[0]
    sortab = comptable[comptable[:, 1].argsort()[::-1], :]
    open('accepted.txt', 'w').write(','.join([str(int(cc)) for cc in acc]))
    open('rejected.txt', 'w').write(','.join([str(int(cc)) for cc in rej]))
    open('midk_rejected.txt',
         'w').write(','.join([str(int(cc)) for cc in midk]))

    _computed_vars = dict(file=op.abspath(op.curdir),
                          vex=varexpl,
                          n_components=n_components,
                          dfe=len(acc),
                          rjn=len(midk) + len(rej),
                          dfn=n_vols - len(midk) - len(rej),
                          acc=','.join([str(int(cc)) for cc in acc]),
                          rej=','.join([str(int(cc)) for cc in rej]),
                          mid=','.join([str(int(cc)) for cc in midk]),
                          ign=','.join([str(int(cc)) for cc in empty]))
    heading = textwrap.dedent("""\
        # ME-ICA Component statistics table for: {file} #
        # Dataset variance explained by ICA (VEx): {vex:.2f}
        # Total components generated by decomposition (TCo): {n_components}
        # No. accepted BOLD-like components, i.e. effective degrees
          of freedom for correlation (lower bound; DFe): {dfe}
        # Total number of rejected components (RJn): {rjn}
        # Nominal degress of freedom in denoised time series
          (..._medn.nii.gz; DFn): {dfn}
        # ACC {acc} \t# Accepted BOLD-like components
        # REJ {rej} \t# Rejected non-BOLD components
        # MID {mid} \t# Rejected R2*-weighted artifacts
        # IGN {ign} \t# Ignored components (kept in denoised time series)
        # VEx   TCo DFe RJn DFn
        # {vex:.2f} {n_components} {dfe} {rjn} {dfn}
        # comp    Kappa   Rho Var   Var(norm)
        """).format(**_computed_vars)

    with open(ctname, 'w') as f:
        f.write(heading)
        for i in range(n_components):
            f.write('%d\t%f\t%f\t%.2f\t%.2f\n' % (sortab[i, 0], sortab[i, 1],
                                                  sortab[i, 2], sortab[i, 3],
                                                  sortab[i, 4]))


def writeresults(ts, mask, comptable, mmix, n_vols, acc, rej, midk, empty, ref_img):
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
    acc : list
        Indices of accepted (BOLD) components in `mmix`
    rej : list
        Indices of rejected (non-BOLD) components in `mmix`
    midk : list
        Indices of mid-K (questionable) components in `mmix`
    empty : list
        Indices of ignored components in `mmix`
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    """

    lgr.info('++ Writing optimally combined time series')
    utils.filewrite(ts, 'ts_OC', ref_img)
    lgr.info('++ Writing Kappa-filtered optimally combined timeseries')
    varexpl = write_split_ts(ts, mmix, mask, acc, rej, midk, ref_img, suffix='OC')
    lgr.info('++ Writing signal versions of components')
    ts_B = model.get_coeffs(ts, mask, mmix)
    utils.filewrite(ts_B, 'betas_OC', ref_img)

    if len(acc) != 0:
        utils.filewrite(ts_B[:, acc], 'betas_hik_OC', ref_img)
        lgr.info('++ Writing optimally combined high-Kappa features')
        writefeats(split_ts(ts, mmix, mask, acc)[0],
                   mmix[:, acc], mask, ref_img, suffix='OC2')
    lgr.info('++ Writing component table')
    writect(comptable, n_vols, acc, rej, midk, empty, ctname='comp_table.txt',
            varexpl=varexpl)


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
    acc : list
        Indices of accepted (BOLD) components in `mmix`
    rej : list
        Indices of rejected (non-BOLD) components in `mmix`
    midk : list
        Indices of mid-K (questionable) components in `mmix`
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    """

    for i_echo in range(catd.shape[1]):
        lgr.info('++ Writing Kappa-filtered echo #{:01d} timeseries'.format(i_echo+1))
        write_split_ts(catd[:, i_echo, :], mmix, mask, acc, rej, midk, ref_img,
                       suffix='e%i' % (i_echo+1))


def ctabsel(ctabfile):
    """
    Loads a pre-existing component table file

    Parameters
    ----------
    ctabfile : str
        Filepath to existing component table

    Returns
    -------
    ctab : (4,) tuple-of-arrays
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
