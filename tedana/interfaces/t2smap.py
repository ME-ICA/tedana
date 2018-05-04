import numpy as np
import nibabel as nib
from tedana.utils import (niwrite, cat2echos,
                          makeadmask, unmask, fmask)

import logging
lgr = logging.getLogger(__name__)


def t2sadmap(data, tes, mask, masksum, start_echo):
    """
    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E, ) list
        Echo times
    mask : (S, ) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    masksum : (S, ) array_like
        Valued array indicating number of echos that have sufficient signal in
        given sample
    start_echo : int
        First echo to consider

    Returns
    -------
    t2sa : (S x E x T) np.ndarray
        Limited T2* map
    s0va : (S x E x T) np.ndarray
        Limited S0 map
    t2ss : (S x E x T) np.ndarray
        ???
    s0vs : (S x E x T) np.ndarray
        ???
    t2saf : (S x E x T) np.ndarray
        Full T2* map
    s0vaf : (S x E x T) np.ndarray
        Full S0 map
    """

    n_samp, n_echos, n_vols = data.shape
    data = data[mask]
    t2ss, s0vs = np.zeros([n_samp, n_echos - 1]), np.zeros([n_samp, n_echos - 1])

    for echo in range(start_echo, n_echos + 1):
        # perform log linear fit of echo times against MR signal
        # make DV matrix: samples x (time series * echos)
        B = np.log((np.abs(data[:, :echo, :]) + 1).reshape(len(data), -1).T)
        # make IV matrix: intercept/TEs x (time series * echos)
        x = np.column_stack([np.ones(echo), [-te for te in tes[:echo]]])
        X = np.repeat(x, n_vols, axis=0)

        beta, res, rank, sing = np.linalg.lstsq(X, B)
        t2s = 1 / beta[1, :].T
        s0 = np.exp(beta[0, :]).T

        t2s[np.isinf(t2s)] = 500.  # why 500?
        s0[np.isnan(s0)] = 0.      # why 0?

        t2ss[..., echo - 2] = np.squeeze(unmask(t2s, mask))
        s0vs[..., echo - 2] = np.squeeze(unmask(s0, mask))

    # create limited T2* and S0 maps
    fl = np.zeros([n_samp, len(tes) - 1], dtype=bool)
    for echo in range(n_echos - 1):
        fl_ = np.squeeze(fl[..., echo])
        fl_[masksum == echo + 2] = True
        fl[..., echo] = fl_
    t2sa, s0va = unmask(t2ss[fl], masksum > 1), unmask(s0vs[fl], masksum > 1)
    # t2sa[masksum > 1], s0va[masksum > 1] = t2ss[fl], s0vs[fl]

    # create full T2* maps with S0 estimation errors
    t2saf, s0vaf = t2sa.copy(), s0va.copy()
    t2saf[masksum == 1] = t2ss[masksum == 1, 0]
    s0vaf[masksum == 1] = s0vs[masksum == 1, 0]

    return t2sa, s0va, t2ss, s0vs, t2saf, s0vaf


def optcom(data, t2, tes, mask, combmode):
    """
    Parameters
    ----------
    data : (S x E x T) array_like
    t2 : (S, ) array_like
    tes : (E, ) list
    combmode : str
        Must be in ['ste', 't2s']. Determines method for optimal combination.

    Returns
    -------
    comb_data : (S x T) np.ndarray
        Optimally combined data
    """

    n_samp, n_echos, n_vols = data.shape

    tes = np.array(tes)[np.newaxis]  # (1 x E) array_like
    t2s = t2[:, np.newaxis]  # (S x 1) array_like

    comb_data = np.zeros((data.shape[0], data.shape[-1]))
    mdata = data[mask]

    if combmode == 'ste':
        alpha = mdata.mean(axis=-1) * tes
    else:
        alpha = tes * np.exp(-tes / t2s[mask])

    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    comb_data[mask] = np.average(mdata, axis=1, weights=alpha)

    return comb_data


def main(options):
    """
    """
    if options.label is not None:
        suf = '_%s' % str(options.label)
    else:
        suf = ''

    tes = [float(te) for te in options.tes]
    ne = len(tes)

    catim = nib.load(options.data[0])
    head = catim.get_header()
    head.extensions = []
    head.set_sform(head.get_sform(), code=1)
    aff = catim.get_affine()
    catd = cat2echos(catim.get_data(), ne)
    nx, ny, nz, Ne, nt = catd.shape

    lgr.info('++ Computing Mask')
    mask, masksum = makeadmask(catd, minimum=False, getsum=True)

    lgr.info('++ Computing Adaptive T2* map')
    t2s, s0, t2ss, s0vs, t2saf, s0vaf = t2sadmap(catd, tes, mask, masksum, 2)
    niwrite(masksum, aff, 'masksum%s.nii' % suf)
    niwrite(t2ss, aff, 't2ss%s.nii' % suf)
    niwrite(s0vs, aff, 's0vs%s.nii' % suf)

    lgr.info('++ Computing optimal combination')
    tsoc = np.array(optcom(catd,
                           t2s,
                           tes,
                           mask,
                           options.combmode),
                    dtype=float)

    # Clean up numerical errors
    t2sm = t2s.copy()
    for n in (tsoc, s0, t2s, t2sm):
        np.nan_to_num(n, copy=False)

    s0[s0 < 0] = 0
    t2s[t2s < 0] = 0
    t2sm[t2sm < 0] = 0

    niwrite(tsoc, aff, 'ocv%s.nii' % suf)
    niwrite(s0, aff, 's0v%s.nii' % suf)
    niwrite(t2s, aff, 't2sv%s.nii' % suf)
    niwrite(t2sm, aff, 't2svm%s.nii' % suf)
