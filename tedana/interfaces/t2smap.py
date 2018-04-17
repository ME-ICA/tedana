import numpy as np
import nibabel as nib
from tedana.utils import (niwrite, cat2echos,
                          makeadmask, unmask, fmask)


def t2sadmap(catd, mask, tes, masksum, start_echo):
    """
    t2sadmap(catd,mask,tes,masksum)

    Input:

    catd  has shape (nx,ny,nz,Ne,nt)
    mask  has shape (nx,ny,nz)
    tes   is a 1d numpy array
    masksum
    """
    nx, ny, nz, Ne, nt = catd.shape
    echodata = fmask(catd, mask)
    Nm = echodata.shape[0]

    t2ss = np.zeros([nx, ny, nz, Ne - 1])
    s0vs = t2ss.copy()

    for ne in range(start_echo, Ne + 1):

        # Do Log Linear fit
        B = np.reshape(np.abs(echodata[:, :ne]) + 1, (Nm, ne * nt)).transpose()
        B = np.log(B)
        neg_tes = [-1 * te for te in tes[:ne]]
        x = np.array([np.ones(ne), neg_tes])
        X = np.tile(x, (1, nt))
        X = np.sort(X)[:, ::-1].transpose()

        beta, res, rank, sing = np.linalg.lstsq(X, B)
        t2s = 1 / beta[1, :].transpose()
        s0 = np.exp(beta[0, :]).transpose()

        t2s[np.isinf(t2s)] = 500.
        s0[np.isnan(s0)] = 0.

        t2ss[:, :, :, ne - 2] = np.squeeze(unmask(t2s, mask))
        s0vs[:, :, :, ne - 2] = np.squeeze(unmask(s0, mask))

    # Limited T2* and S0 maps
    fl = np.zeros([nx, ny, nz, len(tes) - 2 + 1])
    for ne in range(Ne - 1):
        fl_ = np.squeeze(fl[:, :, :, ne])
        fl_[masksum == ne + 2] = True
        fl[:, :, :, ne] = fl_
    fl = np.array(fl, dtype=bool)
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
    out = optcom(data,t2s)


    Input:

    data.shape = (nx,ny,nz,Ne,Nt)
    t2s.shape  = (nx,ny,nz)
    tes.shape  = len(Ne)

    Output:

    out.shape = (nx,ny,nz,Nt)
    """
    nx, ny, nz, Ne, Nt = data.shape

    if useG:
        fdat = fmask(data, mask)
        ft2s = fmask(t2, mask)

    else:
        fdat = fmask(data, mask)
        ft2s = fmask(t2, mask)

    tes = np.array(tes)
    tes = tes[np.newaxis, :]
    ft2s = ft2s[:, np.newaxis]

    if combmode == 'ste':
        alpha = fdat.mean(-1) * tes
    else:
        alpha = tes * np.exp(-tes / ft2s)

    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, Nt))

    fout = np.average(fdat, axis=1, weights=alpha)
    out = unmask(fout, mask)
    print('Out shape is ', out.shape)
    return out


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

    print("++ Computing Mask")
    mask, masksum = makeadmask(catd, min=False, getsum=True)

    print("++ Computing Adaptive T2* map")
    t2s, s0, t2ss, s0vs, t2saf, s0vaf = t2sadmap(catd, mask, tes, masksum, 2)
    niwrite(masksum, aff, 'masksum%s.nii' % suf)
    niwrite(t2ss, aff, 't2ss%s.nii' % suf)
    niwrite(s0vs, aff, 's0vs%s.nii' % suf)

    print("++ Computing optimal combination")
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
