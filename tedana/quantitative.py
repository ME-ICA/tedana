"""
Functions to estimate S0 and T2* from complex multi-echo data.
"""
import logging
import scipy
import numpy as np
import nibabel as nib
from nilearn import image
from tedana import utils
from scipy import ndimage

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def t2star_fit(multiecho_magn, multiecho_phase, echo_times,
               compute_freq_map=True, smooth_freq_map=True,
               compute_corrected_fitting=True,
               out_dir='.', fitting_method='nlls'):
    """
    Estimate T2* values for complex multi-echo data.

    Parameters
    ----------
    multiecho_magn : list of nibabel.nifti1.Nifti1Image or files
        List of magnitude images/time series. Each entry in the list is an echo.
    multiecho_phase
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
        'mask_thresh': 500,  # intensity under which pixels are masked. Default=500.
        # threshold above which voxels are discarded for comuting the frequency map.
        # RMSE results from fitting the frequency slope on the phase data. Default=2.
        'rmse_thresh': 0.8,
        # 'gaussian' | 'box' | 'polyfit1d' | 'polyfit3d'. Default='polyfit3d'
        'smooth_type': 'polyfit3d',
        'smooth_kernel': [27, 27, 7],  # only for 'gaussian' and 'box'
        'smooth_poly_order': 3,
        # 3D downsample frequency map to compute gradient along Z. Default=[2 2 2].
        'smooth_downsampling': [2, 2, 2],
        # minimum length of values along Z, below which values are not considered. Default=4.
        'min_length': 6,
        'dz': 1.25,  # slice thickness in mm. N.B. SHOULD INCLUDE GAP!
        # 0: Just use the initial freqGradZ value - which is acceptable if nicely computed
        'do_optimization': False,
        # in ms. threshold T2* map (for quantization purpose when saving in NIFTI).
        # Suggested value=1000.
        'threshold_t2star_max': 1000,
    }
    # Compute field map of frequencies from multi-echo phase data
    freq_img, mask_img = t2star_computeFreqMap(
        multiecho_magn, multiecho_phase, echo_times,
        mask_thresh=params['mask_thresh'],
        rmse_thresh=params['rmse_thresh'])

    # Smooth field map of frequencies
    freq_smooth = t2star_smoothFreqMap(
        multiecho_magn, multiecho_phase, freq_img, mask_img, echo_times,
        mask_thresh=params['mask_thresh'], rmse_thresh=params['rmse_thresh'],
        smooth_downsampling=params['smooth_downsampling'],
        smooth_type=params['smooth_type'], smooth_kernel=params['smooth_kernel'])

    # Correct z gradients
    grad_z = t2star_computeGradientZ(
        multiecho_magn, freq_smooth, mask, grad_z,
        min_length, poly_fit_order, dz)

    # Estimate corrected T2*
    (t2star_unc, t2star_cor,
     rsquared_unc, rsquared_cor,
     n_iters, grad_z_final) = t2star_computeCorrectedFitting(
        multiecho_magn, multiecho_phase,
        fitting_method, gradZ_file, mask,
        echo_times, do_optimization,
        threshold_t2star_max)
    return t2star_cor


def t2star_computeFreqMap(multiecho_magn, multiecho_phase, echo_times,
                          mask_thresh, rmse_thresh):
    """
    Compute field map of frequencies from multi echo phase data.
    """
    run_4d = False
    echo_times = np.array(echo_times)
    first_img = nib.load(multiecho_magn[0])
    dims = first_img.shape
    n_e = len(multiecho_magn)
    n_x, n_y, n_z, n_t = dims
    assert n_e == len(echo_times) == len(multiecho_phase)
    # convert echo times to seconds
    echo_times_s = echo_times / 1000.
    LGR.info('Loading data')
    # multiecho_magn_imgs = [image.mean_img(img) for img in multiecho_magn]
    # multiecho_phase_imgs = [image.mean_img(img) for img in multiecho_phase]
    multiecho_magn_imgs = [image.index_img(img, 0) for img in multiecho_magn]
    multiecho_phase_imgs = [image.index_img(img, 0) for img in multiecho_phase]
    multiecho_magn_img = image.concat_imgs(multiecho_magn_imgs)
    multiecho_phase_img = image.concat_imgs(multiecho_phase_imgs)
    multiecho_magn_data = multiecho_magn_img.get_fdata()
    multiecho_phase_data = multiecho_phase_img.get_fdata()

    freq_map_3d = np.zeros((n_x, n_y, n_z))
    freq_map_3d_masked = np.zeros((n_x, n_y, n_z))
    grad_z_3d = np.zeros((n_x, n_y, n_z))
    mask_3d = np.zeros((n_x, n_y, n_z))

    # Create 3D frequency map
    for i_slice in range(n_z):
        LGR.info('Slice: {}'.format(i_slice))
        magn_slice_data = multiecho_magn_data[:, :, i_slice, :]
        phase_slice_data = multiecho_phase_data[:, :, i_slice, :]

        # Create mask from magnitude data
        LGR.info("Create mask from first echo's magnitude data...")
        data_multiecho_magn_smooth_2d = ndimage.gaussian_filter(
            magn_slice_data[:, :, 0], sigma=(5, 5), mode='mirror', order=0
        )
        mask_2d = data_multiecho_magn_smooth_2d > mask_thresh
        n_mask_pixels = mask_2d.sum()
        LGR.info("\tNumber of pixels: {}".format(n_mask_pixels))
        mask_3d[:, :, i_slice] = mask_2d

        # convert to Radian [0,2pi), assuming max value is 4095
        LGR.info('\tConverting to Radian [0,2pi), assuming max value is 4095...')
        max_phase_rad = 2 * np.pi * (1 - (1. / 4096))
        phase_slice_data = (phase_slice_data / 4095.) * max_phase_rad

        # This regression could be done in parallel (volume-wise or slice-wise)
        freq_map_1d = np.zeros((n_x * n_y))
        err_phase_1d = np.zeros((n_x * n_y))
        data_multiecho_magn_2d = np.reshape(magn_slice_data, (n_x*n_y, n_e))
        data_multiecho_phase_2d = np.reshape(phase_slice_data, (n_x*n_y, n_e))
        mask_1d = np.reshape(mask_2d, (n_x*n_y))
        X = np.concatenate((echo_times_s[:, None], np.ones((n_e, 1))), axis=1)
        mask_1d_idx = np.where(mask_1d)[0]
        for j_pix, mask_idx in enumerate(mask_1d_idx):
            data_magn_1d = data_multiecho_magn_2d[mask_idx, :]
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
                    (phase_1d_scaled.T - (betas_scaled[0] * echo_times_s + betas_scaled[1])) ** 2
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

    freq_img = nib.Nifti1Image(freq_map_3d, first_img.affine, header=first_img.header)
    mask_img = nib.Nifti1Image(mask_3d, first_img.affine, header=first_img.header)
    freq_img.to_filename('freq.nii.gz')
    mask_img.to_filename('mask.nii.gz')
    return freq_img, mask_img


def t2star_smoothFreqMap(multiecho_magn, multiecho_phase, freq, mask,
                         echo_times, mask_thresh, rmse_thresh,
                         smooth_downsampling, smooth_type, smooth_kernel):
    """
    Smooth frequency map.
    """
    # Downsample field map
    LGR.info('Downsampling field map...')
    new_shape = tuple(multiecho_magn.shape[i] // smooth_downsampling[i] for i
                      in range(len(multiecho_magn.shape)))
    if new_shape != multiecho_magn.shape:
        mag_img = image.resample(multiecho_magn, target_shape=new_shape,
                                 interpolation='nearest')
    else:
        mag_img = multiecho_magn.copy()

    # 3d smooth frequency map (zero values are ignored)
    LGR.info('3d smoothing frequency map using method: {}...'.format(smooth_type))
    if smooth_type == 'gaussian':
        mag_img = image.smooth_img(mag_img, fwhm=smooth_kernel)
    elif smooth_type == 'box':
        pass
    elif smooth_type == 'polyfit1d':
        pass
    elif smooth_type == 'polyfit3d':
        pass
    else:
        raise ValueError('Parameter "smooth_type" must be one of "gaussian", '
                         '"box", "polyfit1d", "polyfit3d"')

    # upsample data back to original resolution
    LGR.info('Upsampling data to native resolution (using nearest neighbor)...')
    if new_shape != multiecho_magn.shape:
        mag_img = image.resample(mag_img, target_shape=multiecho_magn.shape,
                                 interpolation='nearest')

    # Load mask
    LGR.info('Loading magnitude mask...')
    mask = mask

    # apply magnitude mask
    LGR.info('Applying magnitude mask...')
    freq_3d_smooth_masked = unmask(apply_mask(freq_3d_smooth, mask), mask)
    freqGradZ_masked = unmask(apply_mask(freqGradZ, mask), mask)

    # Save smoothed frequency map
    LGR.info('Saving smoothed frequency map...')
    freq_3d_smooth_masked.to_filename('freq_smooth.nii.gz')

    # Save gradient map
    LGR.info('Saving gradient map...')
    freqGradZ_masked.to_filename('freqGradZ.nii.gz')

    return freq_3d_smooth_masked


def t2star_computeCorrectedFitting(multiecho_magn, multiecho_phase,
                                   fitting_method, gradZ_file, mask,
                                   echo_times, do_optimization,
                                   threshold_t2star_max):
    """
    Fit T2* corrected for through-slice drop out.
    """
    return (t2star_unc, t2star_cor, rsquared_unc, rsquared_cor,
            n_iters, grad_z_final)


def func_t2star_optimization(data_magn_1d, echo_times, delta_f, X):
    """
    Optimization function.
    """
    return sd_err


def func_t2star_fit(S, TE, method, X, n_t):
    """
    Perform T2* fit.
    """
    return T2star, S0, Sfit, Rsquared, iter


def t2star_computeGradientZ(multiecho_magn, freq_smooth, mask, grad_z,
                            min_length, poly_fit_order, dz):
    """
    Compute map of gradient frequencies along Z.
    """
    return grad_z
