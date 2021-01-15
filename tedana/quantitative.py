"""Functions to estimate S0 and T2* from complex multi-echo data."""
import logging

import nibabel as nib
import numpy as np
from nilearn import image, masking
from nilearn._utils import load_niimg
from scipy import ndimage, optimize

from tedana import decay

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")


def t2star_fit(
    multiecho_magn,
    multiecho_phase,
    echo_times,
    compute_freq=True,
    smooth_freq=True,
    compute_corrected_fit=True,
    out_dir=".",
    fitting_method="curvefit",
):
    """Estimate T2* values for complex multi-echo data.

    Parameters
    ----------
    multiecho_magn : list of nibabel.nifti1.Nifti1Image or files
        List of magnitude images/time series. Each entry in the list is an echo.
    multiecho_phase : list of nibabel.nifti1.Nifti1Image or files
        List of phase images/time series. Each entry in the list is an echo.
    echo_times : array
        In milliseconds
    fitting_method : {'nlls', 'ols', 'gls', 'num'}, optional
        'nlls': Levenberg-Marquardt nonlinear fitting to exponential (default).
        'ols': Ordinary least squares linear fit of the log of S.
        'gls': Generalized least squares (=weighted least squares), to respect
        heteroscedasticity of the residual when taking the log of S.
        'num': Numerical approximation, based on the NumART2* method in
        [Hagberg, MRM 2002]. Tends to overestimate T2*.
    """
    params = {
        "mask_thresh": 500,  # intensity under which pixels are masked. Default=500.
        # threshold above which voxels are discarded for comuting the frequency map.
        # RMSE results from fitting the frequency slope on the phase data. Default=2.
        "rmse_thresh": 0.8,
        # 'gaussian' | 'box' | 'polyfit1d' | 'polyfit3d'. Default='polyfit3d'
        "smooth_type": "gaussian",
        "smooth_kernel": [27, 27, 7],  # only for 'gaussian' and 'box'
        "smooth_poly_order": 3,
        # 3D downsample frequency map to compute gradient along Z. Default=[2 2 2].
        "smooth_downsampling": [2, 2, 2],
        # minimum length of values along Z, below which values are not considered. Default=4.
        "min_length": 4,
        "dz": 1.25,  # slice thickness in mm. N.B. SHOULD INCLUDE GAP!
        # 0: Just use the initial freqGradZ value - which is acceptable if nicely computed
        "do_optimization": False,
        # in ms. threshold T2* map (for quantization purpose when saving in NIFTI).
        # Suggested value=1000.
        "threshold_t2star_max": 1000,
        "poly_fit_order": 4,
    }

    # Load data and select first volume from each
    # TODO: Support 4D data
    multiecho_magn = [load_niimg(img) for img in multiecho_magn]
    multiecho_phase = [load_niimg(img) for img in multiecho_phase]
    if multiecho_magn[0].ndim == 4:
        multiecho_magn = [image.index_img(img, 0) for img in multiecho_magn]
        multiecho_phase = [image.index_img(img, 0) for img in multiecho_phase]
    multiecho_magn = image.concat_imgs(multiecho_magn)
    multiecho_phase = image.concat_imgs(multiecho_phase)

    # Compute field map of frequencies from multi-echo phase data
    if compute_freq:
        freq_img, mask_img = compute_freq_map(
            multiecho_magn,
            multiecho_phase,
            echo_times,
            mask_thresh=params["mask_thresh"],
            rmse_thresh=params["rmse_thresh"],
        )

    # Smooth field map of frequencies
    if smooth_freq:
        freq_smooth = smooth_freq_map(
            freq_img,
            mask_img,
            echo_times,
            smooth_downsampling=None,
            smooth_type=params["smooth_type"],
            smooth_kernel=params["smooth_kernel"],
        )

    # Correct z gradients
    grad_z = compute_gradient_z(
        multiecho_magn,
        freq_smooth,
        mask_img,
        params["min_length"],
        params["poly_fit_order"],
        params["dz"],
    )

    # Estimate corrected T2*
    (
        t2star_unc,
        t2star_cor,
        rsquared_unc,
        rsquared_cor,
        n_iters,
        grad_z_final,
    ) = compute_corrected_fitting(
        multiecho_magn,
        multiecho_phase,
        fitting_method,
        grad_z,
        mask_img,
        echo_times,
        params["do_optimization"],
        params["threshold_t2star_max"],
    )
    return t2star_cor


def compute_freq_map(
    multiecho_magn, multiecho_phase, echo_times, mask_thresh, rmse_thresh
):
    """Compute field map of frequencies from multi echo phase data.

    Parameters
    ----------
    multiecho_magn, multiecho_phase : 4D niimg_like
        Magnitude and phase data. The fourth dimension is echo, not time.
    echo_times : array_like
        Echo times in milliseconds.
    mask_thresh : float
        Masking threshold.
    rmse_thresh : float
        Residual mean squared error threshold.
    """
    # run_4d = False  # Added by TS
    echo_times = np.array(echo_times)
    dims = multiecho_magn.shape
    n_x, n_y, n_z, n_e = dims
    assert n_e == len(echo_times) == multiecho_phase.shape[3]
    # convert echo times to seconds
    echo_times_s = echo_times / 1000.0

    freq_map_3d = np.zeros((n_x, n_y, n_z))
    mask_3d = np.zeros((n_x, n_y, n_z))

    multiecho_magn_data = multiecho_magn.get_fdata()
    multiecho_phase_data = multiecho_phase.get_fdata()

    # Create 3D frequency map
    for i_slice in range(n_z):
        LGR.info("Slice: {}".format(i_slice))
        magn_slice_data = multiecho_magn_data[:, :, i_slice, :]
        phase_slice_data = multiecho_phase_data[:, :, i_slice, :]

        # Create mask from magnitude data
        LGR.info("Create mask from first echo's magnitude data...")
        data_multiecho_magn_smooth_2d = ndimage.gaussian_filter(
            magn_slice_data[:, :, 0], sigma=(5, 5), mode="mirror", order=0
        )
        mask_2d = data_multiecho_magn_smooth_2d > mask_thresh
        n_mask_pixels = mask_2d.sum()
        LGR.info("\tNumber of pixels: {}".format(n_mask_pixels))
        mask_3d[:, :, i_slice] = mask_2d

        # convert to Radian [0,2pi), assuming max value is 4095
        LGR.info("\tConverting to Radian [0,2pi), assuming max value is 4095...")
        max_phase_rad = 2 * np.pi * (1 - (1.0 / 4096))
        phase_slice_data = (phase_slice_data / 4095.0) * max_phase_rad

        # This regression could be done in parallel (volume-wise or slice-wise)
        freq_map_1d = np.zeros((n_x * n_y))
        err_phase_1d = np.zeros((n_x * n_y))
        data_multiecho_phase_2d = np.reshape(phase_slice_data, (n_x * n_y, n_e))
        mask_1d = np.reshape(mask_2d, (n_x * n_y))
        X = np.concatenate((echo_times_s[:, None], np.ones((n_e, 1))), axis=1)
        mask_1d_idx = np.where(mask_1d)[0]
        for j_pix, mask_idx in enumerate(mask_1d_idx):
            data_phase_1d = data_multiecho_phase_2d[mask_idx, :]

            # unwrap phase
            data_phase_1d_unwrapped = np.unwrap(data_phase_1d)

            # Linear least square fitting of y = a.X + err
            phase_1d = data_phase_1d_unwrapped
            betas_unscaled, _, _, _ = np.linalg.lstsq(X, phase_1d, rcond=None)

            # scale phase signal
            phase_1d_scaled = phase_1d - np.min(phase_1d)
            phase_1d_scaled = phase_1d_scaled / np.max(phase_1d_scaled)
            # Linear least square fitting of scaled phase
            betas_scaled, _, _, _ = np.linalg.lstsq(X, phase_1d_scaled, rcond=None)

            err_phase_1d[mask_idx] = np.sqrt(
                np.sum(
                    (
                        phase_1d_scaled.T
                        - (betas_scaled[0] * echo_times_s + betas_scaled[1])
                    )
                    ** 2
                )
            )

            # Get frequency in Hertz
            freq_map_1d[mask_idx] = betas_unscaled[0] / (2 * np.pi)

        freq_map_2d = np.reshape(freq_map_1d, (n_x, n_y))
        err_phase_2d = np.reshape(err_phase_1d, (n_x, n_y))
        # Crease mask from RMSE map
        mask_freq = np.zeros((n_x, n_y))
        rmse_idx = np.where(err_phase_2d < rmse_thresh)[0]
        mask_freq[rmse_idx] = True  # unused
        freq_map_2d_masked = np.zeros((n_x, n_y))
        freq_map_2d_masked[rmse_idx] = freq_map_2d[rmse_idx]

        # fill 3D matrix
        freq_map_3d[:, :, i_slice] = freq_map_2d_masked

    freq_img = nib.Nifti1Image(freq_map_3d, multiecho_magn.affine, header=multiecho_magn.header)
    mask_img = nib.Nifti1Image(mask_3d, multiecho_magn.affine, header=multiecho_magn.header)
    freq_img.to_filename("freq.nii.gz")
    mask_img.to_filename("mask.nii.gz")
    return freq_img, mask_img


def smooth_freq_map(
    freq, mask, echo_times, smooth_downsampling, smooth_type, smooth_kernel
):
    """Smooth frequency map."""
    # Downsample field map
    LGR.info("Downsampling field map...")
    if smooth_downsampling is not None:
        new_shape = tuple(
            freq.shape[i] // smooth_downsampling[i] for i in range(len(freq.shape))
        )
    else:
        new_shape = freq.shape

    if new_shape != freq.shape:
        freq_img = image.resample_img(freq, target_shape=new_shape, interpolation="nearest")
    else:
        freq_img = freq

    # 3d smooth frequency map (zero values are ignored)
    LGR.info("3d smoothing frequency map using method: {}...".format(smooth_type))
    if smooth_type == "gaussian":
        freq_3d_smooth = image.smooth_img(freq_img, fwhm=smooth_kernel)
    elif smooth_type == "box":
        pass
    elif smooth_type == "polyfit1d":
        pass
    elif smooth_type == "polyfit3d":
        freqGradZ_i = 1
    else:
        raise ValueError(
            'Parameter "smooth_type" must be one of "gaussian", '
            '"box", "polyfit1d", "polyfit3d"'
        )
    #freqGradZ = freqGradZ_i

    # upsample data back to original resolution
    LGR.info("Upsampling data to native resolution (using nearest neighbor)...")
    if new_shape != freq.shape:
        freq_img = image.resample_img(
            freq_img, target_shape=freq.shape, interpolation="nearest"
        )

    # Load mask
    LGR.info("Loading magnitude mask...")

    # apply magnitude mask
    LGR.info("Applying magnitude mask...")
    freq_3d_smooth_masked = masking.unmask(
        masking.apply_mask(freq_3d_smooth, mask), mask
    )
    # it looks like freqGradZ_i is only defined when polyfit3d is used.
    #freqGradZ_masked = masking.unmask(masking.apply_mask(freqGradZ, mask), mask)

    # Save smoothed frequency map
    LGR.info("Saving smoothed frequency map...")
    freq_3d_smooth_masked.to_filename("freq_smooth.nii.gz")

    # Save gradient map
    #LGR.info("Saving gradient map...")
    #freqGradZ_masked.to_filename("freqGradZ.nii.gz")

    return freq_3d_smooth_masked


def compute_corrected_fitting(
    multiecho_magn,
    multiecho_phase,
    fitting_method,
    grad_z,
    mask,
    echo_times,
    do_optimization,
    threshold_t2star_max,
):
    """Fit T2* corrected for through-slice drop out.

    Parameters
    ----------
    multiecho_magn
    multiecho_phase
    fitting_method
    grad_z
    mask
    echo_times
    do_optimization
    threshold_t2star_max

    Returns
    -------
    t2star_unc
    t2star_cor
    rsquared_unc
    rsquared_cor
    n_iters
    grad_z_final
    """
    # Get dimensions of the data
    n_x, n_y, n_z, n_e = multiecho_magn.shape
    multiecho_magn_data = multiecho_magn.get_fdata()

    # Load gradient map

    # Load mask
    mask_data = mask.get_fdata()

    # Check echo time(s)

    # Split volumes (because of memory issue)

    # Loop across slices
    t2star_uncorr_3d = np.zeros((n_x, n_y, n_z))
    t2star_corr_3d = np.zeros((n_x, n_y, n_z))
    grad_z_final_3d = np.zeros((n_x, n_y, n_z))
    rsquared_uncorr_3d = np.zeros((n_x, n_y, n_z))
    rsquared_corr_3d = np.zeros((n_x, n_y, n_z))
    iter_3d = np.zeros((n_x, n_y, n_z))

    X = np.vstack((echo_times[:, None], np.ones((len(echo_times), 1))))
    for i_z in range(n_z):
        # load magnitude
        multiecho_magn_data_z = multiecho_magn_data[:, :, i_z, :]

        # get mask indices
        mask_idx = np.vstack(np.where(mask_data[:, :, i_z]))
        n_pixels = mask_idx.shape[1]

        # initialization
        t2star_uncorr_2d = np.zeros((n_x, n_y))
        t2star_corr_2d = np.zeros((n_x, n_y))
        rsquared_uncorr_2d = np.zeros((n_x, n_y))
        rsquared_corr_2d = np.zeros((n_x, n_y))
        iter_2d = np.zeros((n_x, n_y))

        # loop across pixels
        if do_optimization:
            grad_z_final_2d = np.zeros((n_x, n_y))
            freqGradZ_final_2d = np.zeros((n_x, n_y))

        for j_pix in range(n_pixels):
            j_x, j_y = mask_idx[:, j_pix]

            # Get data magnitude in 1D
            data_magn_1d = multiecho_magn_data_z[j_x, j_y, :]

            if np.any(data_magn_1d):
                # perform uncorrected T2* fit
                t2s_limited, s0_limited, t2s_full, s0_full, r_squared = decay.fit_decay(
                    data_magn_1d[None, :], echo_times, np.array([True]), np.array([n_e]), fitting_method,
                )
                rsquared_uncorr_2d[j_x, j_y] = r_squared
                t2star_uncorr_2d[j_x, j_y] = t2s_limited

                # get initial freqGradZ value from computed map
                freqGradZ_init = grad_z[j_x, j_y, i_z]

                # get final freqGradZ value
                if do_optimization:
                    # Minimization algorithm
                    res = optimize.minimize(
                        optimize_t2star,
                        x0=freqGradZ_init,
                        options={
                            "data_magn_1d": data_magn_1d,
                            "echo_times": echo_times,
                            "X": X,
                        },
                    )
                    freqGradZ_final = res.x
                    freqGradZ_final_2d[mask_idx[j_pix]] = freqGradZ_final

                else:
                    # Just use the initial freqGradZ value - which is acceptable if nicely computed
                    freqGradZ_final = freqGradZ_init

                # Correct signal by sinc function
                # N.B. echo time is in ms
                data_magn_1d_corr = data_magn_1d / np.abs(
                    np.sinc(freqGradZ_final * echo_times / 2000)
                )

                # perform T2* fit
                t2s_limited, s0_limited, t2s_full, s0_full, r_squared = decay.fit_decay(
                    data_magn_1d_corr[None, :], echo_times, np.array([True]), np.array([n_e]), fitting_method,
                )
                rsquared_corr_2d[j_x, j_y] = r_squared
                t2star_corr_2d[j_x, j_y] = t2s_limited

        # fill 3D T2* matrix
        t2star_uncorr_3d[:, :, i_z] = t2star_uncorr_2d
        t2star_corr_3d[:, :, i_z] = t2star_corr_2d
        if do_optimization:
            grad_z_final_3d[:, :, i_z] = grad_z_final_2d
        rsquared_uncorr_3d[:, :, i_z] = rsquared_uncorr_2d
        rsquared_corr_3d[:, :, i_z] = rsquared_corr_2d
        iter_3d[:, :, i_z] = iter_2d

    # threshold T2* map (for quantization purpose when saving in NIFTI).
    t2star_uncorr_3d[t2star_uncorr_3d > threshold_t2star_max] = threshold_t2star_max
    t2star_corr_3d = np.abs(t2star_corr_3d)
    t2star_corr_3d[t2star_corr_3d > threshold_t2star_max] = threshold_t2star_max
    return (
        t2star_uncorr_3d,
        t2star_corr_3d,
        rsquared_uncorr_3d,
        rsquared_corr_3d,
        iter_3d,
        grad_z_final_3d,
    )


def optimize_t2star(data_magn_1d, echo_times, delta_f, X):
    """Optimization function."""
    data_magn_1d_corr = data_magn_1d / np.sinc(delta_f * echo_times / 2)
    y = np.log(data_magn_1d_corr).T
    a = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    t2star_corr = -1 / a[0]
    # compute error
    Sfitted = np.exp(a[1] - echo_times / t2star_corr)
    err = Sfitted - data_magn_1d_corr
    sd_err = np.std(err)
    return sd_err


def compute_gradient_z(
    multiecho_magn, freq_smooth, mask, min_length, poly_fit_order, dz
):
    """Compute map of gradient frequencies along Z.

    Returns
    -------
    grad_z_3d_masked
    """
    # Get dimensions of the data...
    n_x, n_y, n_z, n_e = multiecho_magn.shape

    # Load frequency map
    freq_map_3d_smooth = freq_smooth.get_fdata()

    # Calculate frequency gradient in the slice direction (freqGradZ)
    grad_z_3d = np.zeros((n_x, n_y, n_z))
    grad_z_3d_masked = np.zeros((n_x, n_y, n_z))
    for i_x in range(n_x):
        for j_y in range(n_y):
            # initialize 1D gradient values
            grad_z = np.zeros((1, n_z))
            # get frequency along z (discard zero values)
            freq_z = np.squeeze(freq_map_3d_smooth[i_x, j_y, :])
            ind_nonzero = np.where(freq_z)[0]
            if len(ind_nonzero) >= min_length:
                # fit to polynomial function
                p = np.polyfit(ind_nonzero, freq_z[ind_nonzero], poly_fit_order)
                f = np.polyval(p, np.arange(n_z))
                # compute frequency gradient along Z
                grad_z = np.gradient(f, dz / 1000)
                # fill 3D gradient matrix
                grad_z_3d[i_x, j_y, :] = grad_z

    # Mask gradient map
    grad_z_3d_masked = grad_z_3d * mask.get_fdata()
    grad_z_3d_masked = nib.Nifti1Image(grad_z_3d_masked, mask.affine, mask.header)
    grad_z_3d_masked.to_filename("gradz_masked.nii.gz")
    return grad_z_3d_masked
