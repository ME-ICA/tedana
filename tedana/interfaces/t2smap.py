"""
"""
import numpy as np
import nibabel as nib
from tedana.utils import (niwrite, cat2echos, makeadmask, unmask, fmask)


def fit(data, mask, tes, masksum, start_echo):
    """
    Fit voxel- and timepoint-wise monoexponential decay models to estimate
    T2* and S0 timeseries.
    """
    nx, ny, nz, n_echoes, n_trs = data.shape
    echodata = fmask(data, mask)
    tes = np.array(tes)

    t2sa_ts = np.zeros([nx, ny, nz, n_trs])
    s0va_ts = np.zeros([nx, ny, nz, n_trs])
    t2saf_ts = np.zeros([nx, ny, nz, n_trs])
    s0vaf_ts = np.zeros([nx, ny, nz, n_trs])

    for vol in range(echodata.shape[-1]):
        t2ss = np.zeros([nx, ny, nz, n_echoes - 1])
        s0vs = t2ss.copy()
        # Fit monoexponential decay first for first echo only,
        # then first two echoes, etc.
        for i_echo in range(start_echo, n_echoes + 1):
            B = np.abs(echodata[:, :i_echo, vol]) + 1
            B = np.log(B).transpose()
            neg_tes = -1 * tes[:i_echo]

            # First row is constant, second is TEs for decay curve
            # Independent variables for least-squares model
            x = np.array([np.ones(i_echo), neg_tes])
            X = np.sort(x)[:, ::-1].transpose()

            beta, _, _, _ = np.linalg.lstsq(X, B)
            t2s = 1. / beta[1, :].transpose()
            s0 = np.exp(beta[0, :]).transpose()

            t2s[np.isinf(t2s)] = 500.
            s0[np.isnan(s0)] = 0.

            t2ss[:, :, :, i_echo-2] = np.squeeze(unmask(t2s, mask))
            s0vs[:, :, :, i_echo-2] = np.squeeze(unmask(s0, mask))

        # Limited T2* and S0 maps
        fl = np.zeros([nx, ny, nz, len(tes)-1], bool)
        for i_echo in range(n_echoes - 1):
            fl_ = np.squeeze(fl[:, :, :, i_echo])
            fl_[masksum == i_echo + 2] = True
            fl[:, :, :, i_echo] = fl_
        t2sa = np.squeeze(unmask(t2ss[fl], masksum > 1))
        s0va = np.squeeze(unmask(s0vs[fl], masksum > 1))

        # Full T2* maps with S0 estimation errors
        t2saf = t2sa.copy()
        s0vaf = s0va.copy()
        t2saf[masksum == 1] = t2ss[masksum == 1, 0]
        s0vaf[masksum == 1] = s0vs[masksum == 1, 0]

        t2sa_ts[:, :, :, vol] = t2sa
        s0va_ts[:, :, :, vol] = s0va
        t2saf_ts[:, :, :, vol] = t2saf
        s0vaf_ts[:, :, :, vol] = s0vaf

    return t2sa_ts, s0va_ts, t2saf_ts, s0vaf_ts


def t2sadmap(data, mask, tes, masksum, start_echo):
    """
    Fit voxelwise monoexponential decay models to estimate T2* and S0 maps.
    t2sadmap(data,mask,tes,masksum)

    Input:
    data : :obj:`numpy.ndarray`
        Concatenated BOLD data. Has shape (nx, ny, nz, n_echoes, n_trs)
    mask : :obj:`numpy.ndarray`
        Brain mask in 3D array. Has shape (nx, ny, nz)
    tes : :obj:`numpy.ndarray`
        Array of TEs, in milliseconds.
    masksum :
    """
    nx, ny, nz, n_echoes, n_trs = data.shape
    echodata = fmask(data, mask)
    n_voxels = echodata.shape[0]
    tes = np.array(tes)
    t2ss = np.zeros([nx, ny, nz, n_echoes - 1])
    s0vs = t2ss.copy()

    # Fit monoexponential decay first for first echo only,
    # then first two echoes, etc.
    for i_echo in range(start_echo, n_echoes + 1):
        # Do Log Linear fit
        B = np.reshape(np.abs(echodata[:, :i_echo, :]) + 1,
                       (n_voxels, i_echo*n_trs)).transpose()
        B = np.log(B)
        neg_tes = -1 * tes[:i_echo]

        # First row is constant, second is TEs for decay curve
        # Independent variables for least-squares model
        x = np.array([np.ones(i_echo), neg_tes])
        X = np.tile(x, (1, n_trs))
        X = np.sort(X)[:, ::-1].transpose()

        beta, _, _, _ = np.linalg.lstsq(X, B)
        t2s = 1. / beta[1, :].transpose()
        s0 = np.exp(beta[0, :]).transpose()

        t2s[np.isinf(t2s)] = 500.
        s0[np.isnan(s0)] = 0.

        t2ss[:, :, :, i_echo-2] = np.squeeze(unmask(t2s, mask))
        s0vs[:, :, :, i_echo-2] = np.squeeze(unmask(s0, mask))

    # Limited T2* and S0 maps
    fl = np.zeros([nx, ny, nz, len(tes)-1], bool)
    for i_echo in range(n_echoes - 1):
        fl_ = np.squeeze(fl[:, :, :, i_echo])
        fl_[masksum == i_echo + 2] = True
        fl[:, :, :, i_echo] = fl_
    t2sa = np.squeeze(unmask(t2ss[fl], masksum > 1))
    s0va = np.squeeze(unmask(s0vs[fl], masksum > 1))

    # Full T2* maps with S0 estimation errors
    t2saf = t2sa.copy()
    s0vaf = s0va.copy()
    t2saf[masksum == 1] = t2ss[masksum == 1, 0]
    s0vaf[masksum == 1] = s0vs[masksum == 1, 0]

    return t2sa, s0va, t2ss, s0vs, t2saf, s0vaf


def optcom(data, t2, tes, mask, combmode, useG=False):
    """
    Optimally combine BOLD data across TEs.

    out = optcom(data,t2s)

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Concatenated BOLD data. Has shape (nx, ny, nz, n_echoes, n_trs)
    t2 : :obj:`numpy.ndarray`
        3D map of estimated T2* values. Has shape (nx, ny, nz)
    tes : :obj:`numpy.ndarray`
        Array of TEs, in seconds.
    mask : :obj:`numpy.ndarray`
        Brain mask in 3D array. Has shape (nx, ny, nz)
    combmode : :obj:`str`
        How to combine data. Either 'ste' or 't2s'.
    useG : :obj:`bool`, optional
        Use G. Default is False.

    Returns
    -------
    out : :obj:`numpy.ndarray`
        Optimally combined data. Has shape (nx, ny, nz, n_trs)
    """
    _, _, _, _, n_trs = data.shape

    if useG:
        fdat = fmask(data, mask)
        ft2s = fmask(t2, mask)
    else:
        fdat = fmask(data, mask)
        ft2s = fmask(t2, mask)

    tes = np.array(tes)
    tes = tes[np.newaxis, :]

    if len(t2.shape) == 3:
        print('Optimally combining with voxel-wise T2 estimates')
        ft2s = ft2s[:, np.newaxis]
    else:
        print('Optimally combining with voxel- and volume-wise T2 estimates')
        ft2s = ft2s[:, :, np.newaxis]

    if combmode == 'ste':
        alpha = fdat.mean(-1) * tes
    else:
        alpha = tes * np.exp(-tes / ft2s)

    if len(t2.shape) == 3:
        alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_trs))
    else:
        alpha = np.swapaxes(alpha, 1, 2)
        ax0_idx, ax2_idx = np.where(np.all(alpha == 0, axis=1))
        alpha[ax0_idx, :, ax2_idx] = 1.

    fout = np.average(fdat, axis=1, weights=alpha)
    out = unmask(fout, mask)
    return out


def main(options):
    """
    Estimate T2 and S0, and optimally combine data across TEs.

    Parameters
    ----------
    options
        label
        tes
        data
    """
    if options.label is not None:
        suf = '_%s' % str(options.label)
    else:
        suf = ''

    tes = [float(te) for te in options.tes]
    n_echoes = len(tes)
    catim = nib.load(options.data[0])
    head = catim.get_header()
    head.extensions = []
    head.set_sform(head.get_sform(), code=1)
    aff = catim.get_affine()
    catd = cat2echos(catim.get_data(), n_echoes)
    nx, ny, nz, n_echoes, n_trs = catd.shape

    print("++ Computing Mask")
    mask, masksum = makeadmask(catd, minimum=False, getsum=True)
    niwrite(masksum, aff, 'masksum%s.nii' % suf)

    print("++ Computing Adaptive T2* map")
    t2s, s0, t2ss, s0vs, t2saf, s0vaf = t2sadmap(catd, mask, tes, masksum, 2)
    niwrite(t2ss, aff, 't2ss%s.nii' % suf)
    niwrite(s0vs, aff, 's0vs%s.nii' % suf)

    print("++ Computing optimal combination")
    tsoc = np.array(optcom(catd, t2s, tes, mask, options.combmode),
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
