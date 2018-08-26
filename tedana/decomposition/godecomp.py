"""
Go Decomposition
"""
import logging

import numpy as np
from numpy.linalg import qr, lstsq

from tedana import utils
from tedana.decomposition._utils import dwtmat, idwtmat

LGR = logging.getLogger(__name__)


def wthresh(a, thresh):
    """
    Soft wavelet threshold
    """
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)


def godec(data, thresh=.03, rank=2, power=1, tol=1e-3, max_iter=100,
          random_seed=0, verbose=True):
    """
    Perform Go Decomposition

    Default threshold of .03 is assumed to be for input in the range 0-1...
    original matlab had 8 out of 255, which is about .03 scaled to 0-1 range

    Parameters
    ----------
    data : (M x T) array_like

    Returns
    -------
    L : array_like
        Low-rank components. Similar to global signals. Should be discarded,
        according to Power et al. (2018).
    S : array_like
        Sparse components. Should be retained, according to Power et al.
        (2018).
    G : array_like
        Residuals (i.e., data minus sparse and low-rank components)
    """
    LGR.info('Starting Go Decomposition')
    _, n_vols = data.shape
    L = data
    S = np.zeros(L.shape)
    itr = 0
    random_state = np.random.RandomState(random_seed)
    while True:
        Y2 = random_state.randn(n_vols, rank)
        for i in range(power+1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.T, Y1)
        Q, R = qr(Y2)
        L_new = np.dot(np.dot(L, Q), Q.T)
        T = L - L_new + S
        L = L_new
        S = wthresh(T, thresh)
        T -= S
        err = np.linalg.norm(T.ravel(), 2)
        if err < tol:
            if verbose:
                LGR.info('Successful convergence after {0} '
                         'iterations'.format(itr+1))
            break
        elif itr >= max_iter:
            if verbose:
                LGR.warning('Model failed to converge after {0} '
                            'iterations'.format(itr+1))
            break
        L += T
        itr += 1

    # Is this even useful in soft GoDec? May be a display issue...
    G = data - L - S
    return L, S, G


def _tedgodec(data, wavelet=False, rank=2, increment=2, power=2, tol=1e-3,
              thresh=10, max_iter=500, norm_mode='vn', random_seed=0,
              verbose=True):
    """
    Perform TE-dependent Go Decomposition

    Parameters
    ----------
    data : (M x T) array_like
    wavelet : :obj:`bool`, optional

    """
    if norm_mode == 'dm':
        # Demean
        data_mean = data.mean(-1)
        data_norm = data - data_mean[:, np.newaxis]
    elif norm_mode == 'vn':
        # What is this?
        data_mean = data.mean(-1)[:, np.newaxis]
        data_std = data.std(-1)[:, np.newaxis]
        data_norm = (data - data_mean) / data_std
    else:
        data_norm = data

    # GoDec
    if wavelet:
        data_wt, cal = dwtmat(data_norm)
        L, S, G = godec(data_wt, thresh=data_wt.std()*thresh, rank=rank,
                        power=power, tol=tol, max_iter=max_iter,
                        random_seed=random_seed, verbose=verbose)
        L = idwtmat(L, cal)
        S = idwtmat(S, cal)
        G = idwtmat(G, cal)
    else:
        L, S, G = godec(data_norm, thresh=thresh, rank=rank,
                        power=power, tol=tol, max_iter=max_iter,
                        random_seed=random_seed, verbose=verbose)

    if norm_mode == 'dm':
        # Remean
        L += data_mean
    elif norm_mode == 'vn':
        L = (L * data_std) + data_mean
        S *= data_std
        G *= data_std

    return L, S, G


def tedgodec(optcom_ts, mmix, mask, acc, ref_img, ranks=[2], wavelet=False,
             thresh=10, norm_mode='vn', power=2):
    """
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
    ranks : list of int
        Ranks of low-rank components to run
    """
    # Construct denoised data from optcom, mmix, and acc
    optcom_masked = optcom_ts[mask, :]
    optcom_mu = optcom_masked.mean(axis=-1)[:, np.newaxis]
    optcom_std = optcom_masked.std(axis=-1)[:, np.newaxis]
    data_norm = (optcom_masked - optcom_mu) / optcom_std
    cbetas = lstsq(mmix, data_norm.T, rcond=None)[0].T
    resid = data_norm - np.dot(cbetas, mmix.T)
    bold_ts = np.dot(cbetas[:, acc], mmix[:, acc].T)
    medn_ts = optcom_mu + ((bold_ts + resid) * optcom_std)

    for rank in ranks:
        L, S, G = _tedgodec(medn_ts, rank=rank, power=power, thresh=thresh,
                            max_iter=500, norm_mode=norm_mode)

        if norm_mode is None:
            name_norm_mode = ''
        else:
            name_norm_mode = 'n{0}'.format(norm_mode)

        if wavelet:
            name_norm_mode = 'w{0}'.format(name_norm_mode)

        suffix = '{0}r{1}p{2}t{3}'.format(name_norm_mode, rank, power, thresh)
        utils.filewrite(utils.unmask(L, mask),
                        'lowrank_{0}.nii'.format(suffix), ref_img)
        utils.filewrite(utils.unmask(S, mask),
                        'sparse_{0}.nii'.format(suffix), ref_img)
        utils.filewrite(utils.unmask(G, mask),
                        'noise_{0}.nii'.format(suffix), ref_img)
