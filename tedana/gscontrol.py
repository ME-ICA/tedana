"""
Global signal control methods
"""
import logging

import numpy as np
from numpy.linalg import lstsq
from scipy import stats
from scipy.special import lpmv

from tedana import io, utils

LGR = logging.getLogger(__name__)


def gscontrol_raw(catd, optcom, n_echos, ref_img, bf, dtrank=4):
    """
    Removes global signal from individual echo `catd` and `optcom` time series

    This function uses the spatial global signal estimation approach to
    to removal global signal out of individual echo time series datasets. The
    spatial global signal is estimated from the optimally combined data after
    detrending with a Legendre polynomial basis of `order = 0` and
    `degree = dtrank`.

    Parameters
    ----------
    catd : (S x E x T) array_like
        Input functional data
    optcom : (S x T) array_like
        Optimally combined functional data (i.e., the output of `make_optcom`)
    n_echos : :obj:`int`
        Number of echos in data. Should be the same as `E` dimension of `catd`
    ref_img : :obj:`str` or img_like
        Reference image to dictate how outputs are saved to disk
    bf : :obj:`str`
        Base filename for outputs.
    dtrank : :obj:`int`, optional
        Specifies degree of Legendre polynomial basis function for estimating
        spatial global signal. Default: 4

    Returns
    -------
    dm_catd : (S x E x T) array_like
        Input `catd` with global signal removed from time series
    dm_optcom : (S x T) array_like
        Input `optcom` with global signal removed from time series
    """
    LGR.info('Applying amplitude-based T1 equilibration correction')
    if catd.shape[0] != optcom.shape[0]:
        raise ValueError('First dimensions of catd ({0}) and optcom ({1}) do not '
                         'match'.format(catd.shape[0], optcom.shape[0]))
    elif catd.shape[1] != n_echos:
        raise ValueError('Second dimension of catd ({0}) does not match '
                         'n_echos ({1})'.format(catd.shape[1], n_echos))
    elif catd.shape[2] != optcom.shape[1]:
        raise ValueError('Third dimension of catd ({0}) does not match '
                         'second dimension of optcom '
                         '({1})'.format(catd.shape[2], optcom.shape[1]))

    # Legendre polynomial basis for denoising
    bounds = np.linspace(-1, 1, optcom.shape[-1])
    Lmix = np.column_stack([lpmv(0, vv, bounds) for vv in range(dtrank)])

    # compute mean, std, mask local to this function
    # inefficient, but makes this function a bit more modular
    Gmu = optcom.mean(axis=-1)  # temporal mean
    Gmask = Gmu != 0

    # find spatial global signal
    dat = optcom[Gmask] - Gmu[Gmask][:, np.newaxis]
    sol = np.linalg.lstsq(Lmix, dat.T, rcond=None)[0]  # Legendre basis for detrending
    detr = dat - np.dot(sol.T, Lmix.T)[0]
    sphis = (detr).min(axis=1)
    sphis -= sphis.mean()
    io.filewrite(utils.unmask(sphis, Gmask), io.gen_fname(bf, desc='T1gs'),
                 ref_img)

    # find time course ofc the spatial global signal
    # make basis with the Legendre basis
    glsig = np.linalg.lstsq(np.atleast_2d(sphis).T, dat, rcond=None)[0]
    glsig = stats.zscore(glsig, axis=None)
    np.savetxt(io.gen_fname(bf, '_regressors.tsv', desc='globalSignal'), glsig,
               delimiter='\t')
    glbase = np.hstack([Lmix, glsig.T])

    # Project global signal out of optimally combined data
    sol = np.linalg.lstsq(np.atleast_2d(glbase), dat.T, rcond=None)[0]
    tsoc_nogs = dat - np.dot(np.atleast_2d(sol[dtrank]).T,
                             np.atleast_2d(glbase.T[dtrank])) + Gmu[Gmask][:, np.newaxis]

    io.filewrite(optcom, io.gen_fname(bf, desc='optcomWithGlobalSignal'), ref_img)
    dm_optcom = utils.unmask(tsoc_nogs, Gmask)
    io.filewrite(dm_optcom, io.gen_fname(bf, desc='optcomNoGlobalSignal'), ref_img)

    # Project glbase out of each echo
    dm_catd = catd.copy()  # don't overwrite catd
    for echo in range(n_echos):
        dat = dm_catd[:, echo, :][Gmask]
        sol = np.linalg.lstsq(np.atleast_2d(glbase), dat.T, rcond=None)[0]
        e_nogs = dat - np.dot(np.atleast_2d(sol[dtrank]).T,
                              np.atleast_2d(glbase.T[dtrank]))
        dm_catd[:, echo, :] = utils.unmask(e_nogs, Gmask)

    return dm_catd, dm_optcom


def gscontrol_mmix(optcom_ts, mmix, mask, comptable, ref_img, bf):
    """
    Perform global signal regression.

    Parameters
    ----------
    optcom_ts : (S x T) array_like
        Optimally combined time series data
    mmix : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `optcom_ts`
    mask : (S,) array_like
        Boolean mask array
    comptable : :obj:`pandas.DataFrame`
        Component table with metrics and with classification (accepted,
        rejected, midk, or ignored)
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
    sphis_hik.nii             T1-like effect
    hik_ts_OC_T1c.nii         T1-corrected BOLD (high-Kappa) time series
    dn_ts_OC_T1c.nii          Denoised version of T1-corrected time series
    betas_hik_OC_T1c.nii      T1 global signal-corrected components
    meica_mix_T1c.1D          T1 global signal-corrected mixing matrix
    ======================    =================================================
    """
    all_comps = comptable['component'].values
    acc = comptable.loc[comptable['classification'] == 'accepted', 'component']
    ign = comptable.loc[comptable['classification'] == 'ignored', 'component']
    not_ign = sorted(np.setdiff1d(all_comps, ign))

    optcom_masked = optcom_ts[mask, :]
    optcom_mu = optcom_masked.mean(axis=-1)[:, np.newaxis]
    optcom_std = optcom_masked.std(axis=-1)[:, np.newaxis]

    """
    Compute temporal regression
    """
    data_norm = (optcom_masked - optcom_mu) / optcom_std
    cbetas = lstsq(mmix, data_norm.T, rcond=None)[0].T
    resid = data_norm - np.dot(cbetas[:, not_ign], mmix[:, not_ign].T)

    """
    Build BOLD time series without amplitudes, and save T1-like effect
    """
    bold_ts = np.dot(cbetas[:, acc], mmix[:, acc].T)
    t1_map = bold_ts.min(axis=-1)
    t1_map -= t1_map.mean()
    io.filewrite(utils.unmask(t1_map, mask),
                 io.gen_fname(bf, '_min.nii.gz', desc='optcomAccepted'), ref_img)
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
    hik_ts = bold_noT1gs * optcom_std
    io.filewrite(utils.unmask(hik_ts, mask),
                 io.gen_fname(bf, desc='optcomAcceptedT1cDenoised'), ref_img)

    """
    Make denoised version of T1-corrected time series
    """
    medn_ts = optcom_mu + ((bold_noT1gs + resid) * optcom_std)
    io.filewrite(utils.unmask(medn_ts, mask),
                 io.gen_fname(bf, desc='optcomT1cDenoised'), ref_img)

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
    io.filewrite(utils.unmask(cbetas_norm[:, 2:], mask),
                 io.gen_fname(bf, '_components.nii.gz',
                              desc='TEDICAAcceptedT1cDenoised'), ref_img)
    np.savetxt(io.gen_fname(bf, '_mixing.tsv', desc='TEDICAT1cDenoised'),
               mmixnogs, delimiter='\t')
