"""
PCA based on Moving Average (stationary Gaussian) process
"""
import logging

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.linalg import svd
from scipy.signal import detrend, fftconvolve
from scipy.fftpack import fftshift, fftn

LGR = logging.getLogger(__name__)


def _autocorr(data):
    """
    Calculates the auto correlation of a given array.

    Parameters
    ----------
    data : array-like
        The array to calculate the autocorrelation of

    Returns
    -------
    u : ndarray
        The array of autocorrelations
    """
    u = np.correlate(data, data, mode='full')
    # Take upper half of correlation matrix
    return u[u.size // 2:]


def _check_order(order_in):
    """
    Checks the order passed to the window functions.

    Parameters
    ----------
    order_in : int
        The order to be passed to the window function

    Returns
    -------
    n_out : ndarray
        An integer order array
    w : list
        The window to be used
    trivialwin : boolean
        Whether the window is trivial (w in [0,1])
    """

    w = []
    trivialwin = False

    # Special case of negative orders:
    if order_in < 0:
        raise ValueError('Order cannot be less than zero.')

    order_out = np.round(order_in)
    if not np.array_equal(order_in, order_out):
        LGR.warning('Rounded order to nearest integer')

    # Special cases:
    if not order_out or order_out == 0:
        w = np.zeros((0, 1))  # Empty matrix: 0-by-1
        trivialwin = True
    elif order_out == 1:
        w = 1
        trivialwin = True

    return order_out, w, trivialwin


def _parzen_win(n_points):
    """
    Returns the N-point Parzen (de la Valle-Poussin) window in a column vector.

    Parameters
    ----------
    n_points : int
        Number of non-zero points the window must contain

    Returns
    -------
    parzen_w : 1D array
        The Parzen window

    Notes
    -----
    Maths are described in the following MATLAB documentation page:
    https://www.mathworks.com/help/signal/ref/parzenwin.html

    References
    ----------
    Harris, Fredric J. “On the Use of Windows for Harmonic Analysis
    with the Discrete Fourier Transform.” Proceedings of the IEEE.
    Vol. 66, January 1978, pp. 51–83.
    """

    # Check for valid window length (i.e., n < 0)
    n_points, parzen_w, trivialwin = _check_order(n_points)
    if trivialwin:
        return parzen_w

    # Index vectors
    k = np.arange(-(n_points - 1) / 2, ((n_points - 1) / 2) + 1)
    k1 = k[k < -(n_points - 1) / 4]
    k2 = k[abs(k) <= (n_points - 1) / 4]

    # Equation 37 of [1]: window defined in three sections
    parzen_w1 = 2 * (1 - abs(k1) / (n_points / 2))**3
    parzen_w2 = 1 - 6 * (abs(k2) / (n_points / 2))**2 + 6 * (abs(k2) / (n_points / 2))**3
    parzen_w = np.hstack((parzen_w1, parzen_w2, parzen_w1[::-1])).T

    return parzen_w


def ent_rate_sp(data, sm_window):
    """
    Calculate the entropy rate of a stationary Gaussian random process using
    spectrum estimation with smoothing window.

    Parameters
    ----------
    data : ndarray
        Data to calculate the entropy rate of and smooth
    sm_window : boolean
        Whether there is a Parzen window to use

    Returns
    -------
    ent_rate : float
        The entropy rate

    Notes
    -----
    This function attempts to calculate the entropy rate following

    References
    ----------
    Li, Y.O., Adalı, T. and Calhoun, V.D., (2007).
    Estimating the number of independent components for
    functional magnetic resonance imaging data.
    Human brain mapping, 28(11), pp.1251-1266.
    """

    dims = data.shape

    if data.ndim == 3 and min(dims) != 1:
        pass
    else:
        raise ValueError('Incorrect matrix dimensions.')

    # Normalize x_sb to be unit variance
    data_std = np.std(np.reshape(data, (-1, 1)))

    # Make sure we do not divide by zero
    if data_std == 0:
        raise ValueError('Divide by zero encountered.')
    data = data / data_std

    if sm_window:
        M = [int(i) for i in np.ceil(np.array(dims) / 10)]

        # Get Parzen window for each spatial direction
        parzen_w_3 = np.zeros((2 * dims[2] - 1, ))
        parzen_w_3[(dims[2] - M[2] - 1):(dims[2] + M[2])] = _parzen_win(2 * M[2] + 1)

        parzen_w_2 = np.zeros((2 * dims[1] - 1, ))
        parzen_w_2[(dims[1] - M[1] - 1):(dims[1] + M[1])] = _parzen_win(2 * M[1] + 1)

        parzen_w_1 = np.zeros((2 * dims[0] - 1, ))
        parzen_w_1[(dims[0] - M[0] - 1):(dims[0] + M[0])] = _parzen_win(2 * M[0] + 1)

    # Apply windows to 3D
    # TODO: replace correlate2d with 3d if possible
    data_corr = np.zeros((2 * dims[0] - 1, 2 * dims[1] - 1, 2 * dims[2] - 1))
    for m3 in range(dims[2] - 1):
        temp = np.zeros((2 * dims[0] - 1, 2 * dims[1] - 1))
        for k in range(dims[2] - m3):
            temp += fftconvolve(data[:, :, k + m3], data[::-1, ::-1, k])
            # default option:
            # computes raw correlations with NO normalization
            # -- Matlab help on xcorr
        data_corr[:, :, (dims[2] - 1) - m3] = temp
        data_corr[:, :, (dims[2] - 1) + m3] = temp

    # Create bias-correcting vectors
    v1 = np.hstack((np.arange(1, dims[0] + 1),
                    np.arange(dims[0] - 1, 0, -1)))[np.newaxis, :]
    v2 = np.hstack((np.arange(1, dims[1] + 1),
                    np.arange(dims[1] - 1, 0, -1)))[np.newaxis, :]
    v3 = np.arange(dims[2], 0, -1)

    vd = np.dot(v1.T, v2)
    vcu = np.zeros((2 * dims[0] - 1, 2 * dims[1] - 1, 2 * dims[2] - 1))
    for m3 in range(dims[2]):
        vcu[:, :, (dims[2] - 1) - m3] = vd * v3[m3]
        vcu[:, :, (dims[2] - 1) + m3] = vd * v3[m3]

    data_corr /= vcu

    # Scale Parzen windows
    parzen_window_2D = np.dot(parzen_w_1[np.newaxis, :].T,
                              parzen_w_2[np.newaxis, :])
    parzen_window_3D = np.zeros((2 * dims[0] - 1, 2 * dims[1] - 1, 2 * dims[2] - 1))
    for m3 in range(dims[2] - 1):
        parzen_window_3D[:, :, (dims[2] - 1) - m3] = np.dot(
            parzen_window_2D, parzen_w_3[dims[2] - 1 - m3])
        parzen_window_3D[:, :, (dims[2] - 1) + m3] = np.dot(
            parzen_window_2D, parzen_w_3[dims[2] - 1 + m3])

    # Apply 3D Parzen Window
    data_corr *= parzen_window_3D
    data_fft = abs(fftshift(fftn(data_corr)))
    data_fft[data_fft < 1e-4] = 1e-4

    # Estimation of the entropy rate
    ent_rate = 0.5 * np.log(2 * np.pi * np.exp(1)) + np.sum(np.log(abs(
        (data_fft)))[:]) / 2 / np.sum(abs(data_fft)[:])

    return ent_rate


def _est_indp_sp(data):
    """
    Estimate the effective number of independent samples based on the maximum
    entropy rate principle of stationary random process.

    Parameters
    ----------
    data : ndarray
        The data to have the number of samples estimated

    Returns
    -------
    n_iters : int
        Number of iterations required to estimate entropy rate
    ent_rate : float
        The entropy rate of the data

    Notes
    -----
    This function estimates the effective number of independent samples by omitting
    the least significant components with the subsampling scheme (Li et al., 2007)
    """

    dims = data.shape
    n_iters_0 = None

    for j in range(np.min(dims) - 1):
        data_sb = _subsampling(data, j + 1)
        ent_rate = ent_rate_sp(data_sb, 1)

        # Upper-bound.
        ent_ref = 1.41

        # If entropy rate of a subsampled Gaussian sequence reaches the upper bound
        # of the entropy rate, the subsampled sequence is an i.i.d. sequence.
        if ent_rate > ent_ref:
            n_iters_0 = j
            break

    if n_iters_0 is None:
        raise ValueError('Ill conditioned data, can not estimate '
                         'independent samples.')
    n_iters = n_iters_0
    LGR.debug('Estimated the entropy rate of the Gaussian component '
              'with subsampling depth {}'.format(j + 1))

    return n_iters, ent_rate


def _subsampling(data, sub_depth):
    """
    Subsampling the data evenly with space 'sub_depth'.

    Parameters
    ----------
    data : ndarray
        The data to be subsampled
    sub_depth : int
        The subsampling depth

    Returns
    -------
    out : ndarray
        Subsampled data
    """

    # First index from which to start subsampling for each dimension
    idx_0 = [0, 0, 0]
    ndims = data.shape

    if data.ndim == 3 and np.min(ndims) != 1:  # 3D
        out = data[np.arange(
            idx_0[0], ndims[0], sub_depth), :, :][:, np.arange(
                idx_0[1], ndims[1], sub_depth), :][:, :, np.arange(idx_0[2], ndims[2], sub_depth)]
    else:
        raise ValueError('Unrecognized matrix dimension! )'
                         'Input array must be 3D with min dimension > 1.')

    return out


def _kurtn(data):
    """
    Normalized kurtosis funtion so that for a Gaussian r.v. the kurtn(g) = 0.

    Parameters
    ----------
    data : ndarray
        The data to calculate the kurtosis of

    Returns
    -------
    kurt : (1:N) array-like
        The kurtosis of each vector in x along the second dimension. For
        tedana, this will be the kurtosis of each PCA component.
    """

    kurt = np.zeros((data.shape[1], 1))

    for i in range(data.shape[1]):
        data_norm = detrend(data[:, i], type='constant')
        data_norm /= np.std(data_norm)
        kurt[i] = np.mean(data_norm**4) - 3

    kurt[kurt < 0] = 0

    return kurt


def _icatb_svd(data, n_comps=None):
    """
    Run Singular Value Decomposition (SVD) on input data and extracts the
    given number of components (n_comps).

    Parameters
    ----------
    data : array
        The data to compute SVD for
    n_comps : int
        Number of PCA components to be kept

    Returns
    -------
    V : 2D array
        Eigenvectors from SVD
    Lambda : float
        Eigenvalues
    """

    if not n_comps:
        n_comps = np.min((data.shape[0], data.shape[1]))

    _, Lambda, vh = svd(data, full_matrices=False)

    # Sort eigen vectors in Ascending order
    V = vh.T
    Lambda = Lambda / np.sqrt(data.shape[0] - 1)  # Whitening (sklearn)
    inds = np.argsort(np.power(Lambda, 2))
    Lambda = np.power(Lambda, 2)[inds]
    V = V[:, inds]
    sumAll = np.sum(Lambda)

    # Return only the extracted components
    V = V[:, (V.shape[1] - n_comps):]
    Lambda = Lambda[Lambda.shape[0] - n_comps:]
    sumUsed = np.sum(Lambda)
    retained = (sumUsed / sumAll) * 100
    LGR.debug('{ret}% of non-zero components retained'.format(ret=retained))

    return V, Lambda


def _eigensp_adj(lam, n, p):
    """
    Eigen spectrum adjustment for EVD on finite samples.

    Parameters
    ----------
    lam : [Px1] array-like
        Component eigenvalues
    n : int
        Effective number of i.i.d. samples.
    p : int
        Number of eigen values.

    Returns
    -------
    lam_adj : (p,) array-like
              adjusted eigen values.

    Notes
    -----
    Adjusts the eigen spectrum to account for the finite samples
    after subsampling (Li et al., 2007)

    References
    ----------
    Li, Y.O., Adalı, T. and Calhoun, V.D., (2007).
    Estimating the number of independent components for
    functional magnetic resonance imaging data.
    Human brain mapping, 28(11), pp.1251-1266.
    """

    r = p / n
    bp = np.power((1 + np.sqrt(r)), 2)
    bm = np.power((1 - np.sqrt(r)), 2)
    vv_step = (bp - bm) / (5 * p - 1)
    vv = np.arange(bm, bp + vv_step, vv_step)
    gv = (1 / (2 * np.pi * r * vv)) * np.sqrt(abs((vv - bm) * (bp - vv)))
    gvd = np.zeros(gv.shape)
    for i in range(gv.shape[0]):
        gvd[i] = sum(gv[0:i])

    gvd /= np.max(gvd)

    lam_emp = np.zeros(lam.shape)
    for idx, i in enumerate(np.arange(1, p + 1)):
        i_norm = (i) / p
        minx = np.argmin(abs(i_norm - gvd))
        lam_emp[idx] = vv[minx]

    lam_emp = np.flip(lam_emp)

    lam_adj = lam / lam_emp

    return lam_adj


def ma_pca(data_nib, mask_nib, criteria='mdl'):
    """
    Run Singular Value Decomposition (SVD) on input data,
    automatically select components based on a Moving Average
    (stationary Gaussian) process. Finally perform PCA with
    selected number of components.

    Parameters
    ----------
    data_nib : 4D nibabel
               Unmasked data to compute the PCA on.
    mask_nib : 4D nibabel
               Mask to apply on data_nib.
    criteria : string in ['aic', 'kic', mdl']
               Criteria to select the number of components;
               default='mdl'.

    Returns
    -------
    u : (S [*E] x C) array-like
        Component weight map for each component.
    s : (C,) array-like
        Variance explained for each component.
    varex_norm : (n_components,) array-like
        Explained variance ratio.
    v : (T x C) array-like
        Component timeseries.

    Notes
    -----
    aic : Akaike Information Criterion. Least aggressive option.
    kic : Kullback-Leibler Information Criterion. Stands in the
          middle in terms of aggressiveness.
    mdl : Minimum Description Length. Most aggressive
          (and recommended) option.
    """

    data_nib = data_nib.get_data()
    mask_nib = mask_nib.get_data()
    [Nx, Ny, Nz, Nt] = data_nib.shape
    data_nib_V = np.reshape(data_nib, (Nx * Ny * Nz, Nt), order='F')
    maskvec = np.reshape(mask_nib, Nx * Ny * Nz, order='F')
    data_non_normalized = data_nib_V[maskvec == 1, :]
    scaler = StandardScaler(with_mean=True, with_std=True)
    # TODO: determine if tedana is already normalizing before this
    data = scaler.fit_transform(data_non_normalized)  # This was X_sc
    data = data_non_normalized

    LGR.info('Performing SVD on original OC data...')
    V, EigenValues = _icatb_svd(data, Nt)
    LGR.info('SVD done on original OC data')

    # Reordering of values
    EigenValues = EigenValues[::-1]
    dataN = np.dot(data, V[:, ::-1])
    # Potentially the small differences come from the different signs on V

    # Using 12 gaussian components from middle, top and bottom gaussian
    # components to determine the subsampling depth. Final subsampling depth is
    # determined using median
    kurtv1 = _kurtn(dataN)
    kurtv1[EigenValues > np.mean(EigenValues)] = 1000
    idx_gauss = np.where(
        ((kurtv1[:, 0] < 0.3) & (kurtv1[:, 0] > 0) & (EigenValues > np.finfo(float).eps)
         ) == 1)[0]  # DOUBT: make sure np.where is giving us just one tuple
    idx = np.array(idx_gauss[:]).T
    dfs = np.sum(EigenValues > np.finfo(float).eps)  # degrees of freedom
    minTp = 12

    if (len(idx) >= minTp):
        middle = int(np.round(len(idx) / 2))
        idx = np.hstack([idx[0:4], idx[middle - 1:middle + 3], idx[-4:]])
    else:
        minTp = np.min([minTp, dfs])
        idx = np.arange(dfs - minTp, dfs)

    idx = np.unique(idx)

    # Estimate the subsampling depth for effectively i.i.d. samples
    LGR.info('Estimating the subsampling depth for effective i.i.d samples...')
    mask_ND = np.reshape(maskvec, (Nx, Ny, Nz), order='F')
    sub_depth = len(idx)
    sub_iid_sp = np.zeros((sub_depth, ))
    for i in range(sub_depth):
        x_single = np.zeros(Nx * Ny * Nz)
        x_single[maskvec == 1] = dataN[:, idx[i]]
        x_single = np.reshape(x_single, (Nx, Ny, Nz), order='F')
        sub_iid_sp[i] = _est_indp_sp(x_single)[0] + 1
        if i > 6:
            tmp_sub_sp = sub_iid_sp[0:i]
            tmp_sub_median = np.round(np.median(tmp_sub_sp))
            if np.sum(tmp_sub_sp == tmp_sub_median) > 6:
                sub_iid_sp = tmp_sub_sp
                break
        dim_n = x_single.ndim

    sub_iid_sp_median = int(np.round(np.median(sub_iid_sp)))
    if np.floor(np.power(np.sum(maskvec) / Nt, 1 / dim_n)) < sub_iid_sp_median:
        sub_iid_sp_median = int(np.floor(np.power(np.sum(maskvec) / Nt, 1 / dim_n)))
    N = np.round(np.sum(maskvec) / np.power(sub_iid_sp_median, dim_n))

    if sub_iid_sp_median != 1:
        mask_s = _subsampling(mask_ND, sub_iid_sp_median)
        mask_s_1d = np.reshape(mask_s, np.prod(mask_s.shape), order='F')
        dat = np.zeros((int(np.sum(mask_s_1d)), Nt))
        LGR.info('Generating subsampled i.i.d. OC data...')
        for i in range(Nt):
            x_single = np.zeros((Nx * Ny * Nz, ))
            x_single[maskvec == 1] = data[:, i]
            x_single = np.reshape(x_single, (Nx, Ny, Nz), order='F')
            dat0 = _subsampling(x_single, sub_iid_sp_median)
            dat0 = np.reshape(dat0, np.prod(dat0.shape), order='F')
            dat[:, i] = dat0[mask_s_1d == 1]

        # Perform Variance Normalization
        dat = scaler.fit_transform(dat)

        # (completed)
        LGR.info('Performing SVD on subsampled i.i.d. OC data...')
        [V, EigenValues] = _icatb_svd(dat, Nt)
        LGR.info('SVD done on subsampled i.i.d. OC data')
        EigenValues = EigenValues[::-1]

    LGR.info('Effective number of i.i.d. samples %d' % N)

    # Make eigen spectrum adjustment
    LGR.info('Perform eigen spectrum adjustment ...')
    EigenValues = _eigensp_adj(EigenValues, N, EigenValues.shape[0])
    # (completed)
    if np.sum(np.imag(EigenValues)):
        raise ValueError('Invalid eigen value found for the subsampled data.')

    # Correction on the ill-conditioned results (when tdim is large,
    # some least significant eigenvalues become small negative numbers)
    if EigenValues[np.real(EigenValues) <= np.finfo(float).eps].shape[0] > 0:
        EigenValues[np.real(EigenValues) <= np.finfo(float).eps] = np.min(
            EigenValues[np.real(EigenValues) >= np.finfo(float).eps])
    LGR.info('Estimating the dimension ...')
    p = Nt
    aic = np.zeros(p - 1)
    kic = np.zeros(p - 1)
    mdl = np.zeros(p - 1)

    for k_idx, k in enumerate(np.arange(1, p)):
        LH = np.log(np.prod(np.power(EigenValues[k:], 1 / (p - k))) / np.mean(EigenValues[k:]))
        mlh = 0.5 * N * (p - k) * LH
        df = 1 + 0.5 * k * (2 * p - k + 1)
        aic[k_idx] = (-2 * mlh) + (2 * df)
        kic[k_idx] = (-2 * mlh) + (3 * df)
        mdl[k_idx] = -mlh + (0.5 * df * np.log(N))

    itc = np.row_stack([aic, kic, mdl])

    if criteria == 'aic':
        criteria_idx = 0
    elif criteria == 'kic':
        criteria_idx = 1
    elif criteria == 'mdl':
        criteria_idx = 2

    dlap = np.diff(itc[criteria_idx, :])
    a = np.where(dlap > 0)[0] + 1  # Plus 1 to
    if a.size == 0:
        comp_est = itc[criteria_idx, :].shape[0]
    else:
        comp_est = a[0]

    LGR.info('Estimated components is found out to be %d' % comp_est)

    # PCA with estimated number of components
    ppca = PCA(n_components=comp_est, svd_solver='full', copy=False)
    ppca.fit(data)
    v = ppca.components_.T
    s = ppca.explained_variance_
    u = np.dot(np.dot(data, v), np.diag(1. / s))
    varex_norm = ppca.explained_variance_ratio_

    return u, s, varex_norm, v
