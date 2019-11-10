import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.signal import detrend
from scipy.fftpack import fft, fftshift, fftn
from scipy.signal import correlate2d


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size / 2:]


def sumN(dat):
    # sum of the all elements of the dat matrix
    return np.sum(dat[:])


def checkOrder(n_in):
    # CHECKORDER Checks the order passed to the window functions.

    w = []
    trivalwin = 0

    # Special case of negative orders:
    if n_in < 0:
        print('ERROR!!!! Order cannot be less than zero.')

    # Check if order is already an integer or empty
    # If not, round to nearest integer.
    if not n_in or n_in == np.floor(n_in):
        n_out = n_in
    else:
        n_out = np.round(n_in)
        print('WARNING: Rounding order to nearest integer.')

    # Special cases:
    if not n_out or n_out == 0:
        w = np.zeros((0, 1))  # Empty matrix: 0-by-1
        trivalwin = 1
    elif n_out == 1:
        w = 1
        trivalwin = 1

    return n_out, w, trivalwin


def parzen_win(n):
    # PARZENWIN Parzen window.
    # PARZENWIN(N) returns the N-point Parzen (de la Valle-Poussin) window in
    # a column vector.

    # Check for valid window length (i.e., n < 0)
    n, w, trivialwin = checkOrder(n)
    if trivialwin:
        return w

    # Index vectors
    k = np.arange(-(n - 1) / 2, ((n - 1) / 2) + 1)
    k1 = k[k < -(n - 1) / 4]
    k2 = k[abs(k) <= (n - 1) / 4]

    # Equation 37 of [1]: window defined in three sections
    w1 = 2 * (1 - abs(k1) / (n / 2))**3
    w2 = 1 - 6 * (abs(k2) / (n / 2))**2 + 6 * (abs(k2) / (n / 2))**3
    w = np.hstack((w1, w2, w1[::-1])).T

    return w


def entrate_sp(x, sm_window):

    # Calculate the entropy rate of a stationary Gaussian random process using
    # spectrum estimation with smoothing window

    n = x.shape

    # Normalize x_sb to be unit variance
    x_std = np.std(np.reshape(x, (np.prod(n), 1)))
    if x_std < 1e-10:
        x_std = 1e-10
    x = x / x_std

    if (sm_window == 1):

        M = [int(i) for i in np.ceil(np.array(n) / 10)]

        if (x.ndim >= 3):
            parzen_w_3 = np.zeros((2 * n[2] - 1, ))
            parzen_w_3[(n[2] - M[2] - 1):(n[2] + M[2])] = parzen_win(2 * M[2] +
                                                                     1)

        if (x.ndim >= 2):
            parzen_w_2 = np.zeros((2 * n[1] - 1, ))
            parzen_w_2[(n[1] - M[1] - 1):(n[1] + M[1])] = parzen_win(2 * M[1] +
                                                                     1)

        if (x.ndim >= 1):
            parzen_w_1 = np.zeros((2 * n[0] - 1, ))
            parzen_w_1[(n[0] - M[0] - 1):(n[0] + M[0])] = parzen_win(2 * M[0] +
                                                                     1)

    if x.ndim == 2 and min(n) == 1:  # 1D
        xc = autocorr(x)
        xc = xc * parzen_w
        xf = fftshift(fft(xc))

    elif x.ndim == 2 and min(n) != 1:  # 2D
        xc = autocorr(x)  # default option: computes raw correlations with NO
        # normalization -- Matlab help on xcorr

        # Bias correction
        v1 = np.hstack((np.arange(1, n[0] + 1), np.arange(n[0] - 1, 0,
                                                          -1)))[np.newaxis, :]
        v2 = np.hstack((np.arange(1, n[1] + 1), np.arange(n[1] - 1, 0,
                                                          -1)))[np.newaxis, :]

        vd = np.dot(v1.T, v2)
        xc = xc / vd
        parzen_window_2D = np.dot(parzen_w_1, parzen_w_2.T)
        xc = xc * parzen_window_2D
        xf = fftshift(fft2(xc))

    elif x.ndim == 3 and min(n) != 1:  # 3D
        xc = np.zeros((2 * n[0] - 1, 2 * n[1] - 1, 2 * n[2] - 1))
        for m3 in range(n[2] - 1):
            temp = np.zeros((2 * n[0] - 1, 2 * n[1] - 1))
            for k in range(n[2] - m3):
                temp = temp + correlate2d(x[:, :, k + m3], x[:, :, k])
                # default option:
                # computes raw correlations with NO normalization
                # -- Matlab help on xcorr
            xc[:, :, (n[2] - 1) - m3] = temp
            xc[:, :, (n[2] - 1) + m3] = temp

        # Bias correction
        v1 = np.hstack((np.arange(1, n[0] + 1), np.arange(n[0] - 1, 0,
                                                          -1)))[np.newaxis, :]
        v2 = np.hstack((np.arange(1, n[1] + 1), np.arange(n[1] - 1, 0,
                                                          -1)))[np.newaxis, :]
        v3 = np.arange(n[2], 0, -1)

        vd = np.dot(v1.T, v2)
        vcu = np.zeros((2 * n[0] - 1, 2 * n[1] - 1, 2 * n[2] - 1))
        for m3 in range(n[2]):
            vcu[:, :, (n[2] - 1) - m3] = vd * v3[m3]
            vcu[:, :, (n[2] - 1) + m3] = vd * v3[m3]

        # Possible source of NAN values
        xc = xc / vcu

        parzen_window_2D = np.dot(parzen_w_1[np.newaxis, :].T,
                                  parzen_w_2[np.newaxis, :])
        parzen_window_3D = np.zeros((2 * n[0] - 1, 2 * n[1] - 1, 2 * n[2] - 1))
        for m3 in range(n[2] - 1):
            parzen_window_3D[:, :, (n[2] - 1) - m3] = np.dot(
                parzen_window_2D, parzen_w_3[n[2] - 1 - m3])
            parzen_window_3D[:, :, (n[2] - 1) + m3] = np.dot(
                parzen_window_2D, parzen_w_3[n[2] - 1 + m3])

        xc = xc * parzen_window_3D

        xf = fftshift(fftn(xc))

    else:

        print('ERROR!!!!! Unrecognized matrix dimension.')

    xf = abs(xf)
    xf[xf < 1e-4] = 1e-4
    out = 0.5 * np.log(2 * np.pi * np.exp(1)) + sumN(np.log(abs(
        (xf)))) / 2 / sumN(abs(xf))

    return out


def est_indp_sp(x):

    dimv = x.shape
    s0 = 0

    for j in range(np.min(dimv) - 1):
        x_sb = subsampling(x, j + 1, [0, 0, 0])
        if j == 0:
            print(
                'Estimating the entropy rate of the Gaussian component with subsampling depth {},'
                .format(j))
        else:
            print(' {},'.format(j))

        entrate_m = entrate_sp(x_sb, 1)

        ent_ref = 1.41
        if entrate_m > ent_ref:
            s0 = j
            break

    print(' Done;')
    if s0 == 0:
        print(
            'ERROR!!!!! Ill conditioned data, can not estimate independent samples.(est_indp_sp)'
        )
    else:
        s = s0

    return s, entrate_m


def subsampling(x, s, x0):
    # Subsampling the data evenly with space 's'

    n = x.shape

    if x.ndim == 2 and np.min(n) == 1:  # 1D
        out = x[np.arange(x0[0], np.max(n), s)]

    elif x.ndim == 2 and np.min(n) != 1:  # 2D
        out = x[np.arange(x0[0], n[0], s), :][:, np.arange(x0[1], n[1], s)]

    elif x.ndim == 3 and np.min(n) != 1:  # 3D
        out = x[np.arange(
            x0[0], n[0], s), :, :][:, np.arange(
                x0[1], n[1], s), :][:, :, np.arange(x0[2], n[2], s)]

    else:
        print('ERROR!!!! Unrecognized matrix dimension!(subsampling)')

    return out


def kurtn(x):

    kurt = np.zeros((x.shape[1], 1))

    for i in range(x.shape[1]):
        a = detrend(x[:, i], type='constant')
        a = a / np.std(a)
        kurt[i] = np.mean(a**4) - 3

    return kurt


def icatb_svd(data, numpc):
    _, Lambda, vh = svd(data)
    # Sort eigen vectors in Ascending order
    V = vh.T
    Lambda = Lambda / np.sqrt(data.shape[0] - 1)  # Whitening (sklearn)
    inds = np.argsort(np.power(Lambda, 2))
    Lambda = np.power(Lambda, 2)[inds]
    V = V[:, inds]
    sumAll = np.sum(Lambda)

    # Return only the extracted components
    V = V[:, (V.shape[1] - numpc):]
    Lambda = Lambda[Lambda.shape[0] - numpc:]
    sumUsed = np.sum(Lambda)
    retained = (sumUsed / sumAll) * 100
    print('{ret}% of non-zero eigenvalues retained'.format(ret=retained))

    return V, Lambda


def eigensp_adj(lam, n, p):
    r = p / n
    bp = np.power((1 + np.sqrt(r)), 2)
    bm = np.power((1 - np.sqrt(r)), 2)
    vv_step = (bp - bm) / (5 * p - 1)
    vv = np.arange(bm, bp + vv_step, vv_step)
    gv = (1 / (2 * np.pi * r * vv)) * np.sqrt(abs((vv - bm) * (bp - vv)))
    gvd = np.zeros(gv.shape)
    for i in range(gv.shape[0]):
        gvd[i] = sum(gv[0:i])

    gvd = gvd / np.max(gvd)

    lam_emp = np.zeros(lam.shape)
    for idx, i in enumerate(np.arange(1, p + 1)):
        i_norm = (i) / p
        minx = np.argmin(abs(i_norm - gvd))
        lam_emp[idx] = vv[minx]

    lam_emp = np.flip(lam_emp)

    lam_adj = lam / lam_emp

    return lam_adj


def run_gift_pca(data, criteria='mdl'):

    # [Nx, Ny, Nz, Nt] = data_nib.shape

    # scaler = StandardScaler(with_mean=True, with_std=True)
    # data = scaler.fit_transform(DATA)
    Nt = data.shape[1]

    V, EigenValues = icatb_svd(data, Nt)

    # Reordering of values
    EigenValues = EigenValues[::-1]
    dataN = np.dot(data, V[:, ::-1])
    # Potentially the small differences come from the different signs on V

    # Rename SVD results to be used later
    V_orig = V.copy()
    S_orig = EigenValues.copy()

    # Using 12 gaussian components from middle, top and bottom gaussian
    # components to determine the subsampling depth. Final subsampling depth is
    # determined using median
    kurtv1 = kurtn(dataN)
    kurtv1[EigenValues > np.mean(EigenValues)] = 1000
    idx_gauss = np.where(
        ((kurtv1[:, 0] < 0.3) * (EigenValues > np.finfo(float).eps)
         ) == 1)[0]  # DOUBT: make sure np.where is giving us just one tuple
    idx = np.array(idx_gauss[:]).T
    dfs = len(
        np.where(EigenValues > np.finfo(float).eps)[0])  # degrees of freedom
    minTp = 12
    if (len(idx) >= minTp):
        middle = int(np.round(len(idx) / 2))
        idx = np.hstack([idx[0:4], idx[middle - 1:middle + 3], idx[-4:]])
    elif not idx:
        minTp = np.min([minTp, dfs])
        idx = np.arange(dfs - minTp, dfs)

    idx = np.unique(idx)

    # Estimate the subsampling depth for effectively i.i.d. samples
    mask_ND = np.reshape(maskvec, (Nx, Ny, Nz), order='F')
    ms = len(idx)
    s = np.zeros((ms, ))
    for i in range(ms):
        x_single = np.zeros(Nx * Ny * Nz)
        x_single[maskvec == 1] = dataN[:, idx[i]]
        x_single = np.reshape(x_single, (Nx, Ny, Nz), order='F')
        s[i] = est_indp_sp(x_single)[0] + 1
        if i > 6:
            tmpS = s[0:i]
            tmpSMedian = np.round(np.median(tmpS))
            if np.sum(tmpS == tmpSMedian) > 6:
                s = tmpS
                break
        dim_n = x_single.ndim
    s1 = int(np.round(np.median(s)))
    if np.floor(np.power(np.sum(maskvec) / Nt, 1 / dim_n)) < s1:
        s1 = int(np.floor(np.power(np.sum(maskvec) / Nt, 1 / dim_n)))
    N = np.round(np.sum(maskvec) / np.power(s1, dim_n))

    # Use the subsampled dataset to calculate eigen values
    print('Perform EVD on the effectively i.i.d. samples ...')

    if s1 != 1:
        mask_s = subsampling(mask_ND, s1, [0, 0, 0])
        mask_s_1d = np.reshape(mask_s, np.prod(mask_s.shape), order='F')
        dat = np.zeros((int(np.sum(mask_s_1d)), Nt))
        for i in range(Nt):
            x_single = np.zeros((Nx * Ny * Nz, ))
            x_single[maskvec == 1] = data[:, i]
            x_single = np.reshape(x_single, (Nx, Ny, Nz), order='F')
            dat0 = subsampling(x_single, s1, [0, 0, 0])
            dat0 = np.reshape(dat0, np.prod(dat0.shape), order='F')
            dat[:, i] = dat0[mask_s_1d == 1]

        # Perform Variance Normalization
        dat = scaler.fit_transform(dat)

        # (completed)
        print('P2:')
        [V, EigenValues] = icatb_svd(dat, Nt)
        EigenValues = EigenValues[::-1]

    lam = EigenValues

    print(' Effective number of i.i.d. samples %d' % N)

    # Make eigen spectrum adjustment
    print('Perform eigen spectrum adjustment ...')
    lam = eigensp_adj(lam, N, lam.shape[0])
    # (completed)
    if np.sum(np.imag(lam)):
        print('ERROR: Invalid eigen value found for the subsampled data.')

    # Correction on the ill-conditioned results (when tdim is large,
    # some least significant eigenvalues become small negative numbers)
    if lam[np.real(lam) <= np.finfo(float).eps].shape[0] > 0:
        lam[np.real(lam) <= np.finfo(float).eps] = np.min(
            lam[np.real(lam) >= np.finfo(float).eps])
    print(' Estimating the dimension ...')
    p = Nt
    aic = np.zeros(p - 1)
    kic = np.zeros(p - 1)
    mdl = np.zeros(p - 1)

    for k_idx, k in enumerate(np.arange(1, p)):
        LH = np.log(np.prod(np.power(lam[k:], 1 / (p - k))) / np.mean(lam[k:]))
        mlh = 0.5 * N * (p - k) * LH
        df = 1 + 0.5 * k * (2 * p - k + 1)
        aic[k_idx] = (-2 * mlh) + (2 * df)
        kic[k_idx] = (-2 * mlh) + (3 * df)
        mdl[k_idx] = -mlh + (0.5 * df * np.log(N))
        #assert False

    itc = np.zeros((3, mdl.shape[0]))
    itc[0, :] = aic
    itc[1, :] = kic
    itc[2, :] = mdl

    if criteria == 'aic':
        criteria_idx = 0
    elif criteria == 'kic':
        criteria_idx = 1
    elif criteria == 'mdl':
        criteria_idx = 2

    dlap = np.diff(itc[criteria_idx, :])
    a = np.where(dlap > 0)[0] + 1  # Plus 1 to
    if a is None:
        comp_est = itc[criteria_idx, :].shape[0]
    else:
        comp_est = a[0]

    print('Estimated components is found out to be %d' % comp_est)

    # PCA with estimated number of components
    ppca = PCA(n_components=comp_est, svd_solver='full', copy=False)
    ppca.fit(data)
    v = ppca.components_.T
    s = ppca.explained_variance_
    u = np.dot(np.dot(data, v), np.diag(1. / s))
    varex_norm = ppca.explained_variance_ratio_

    return u, s, varex_norm, v