"""
Functions to optimally combine data across echoes.
"""
import logging
import numpy as np
from tedana.utils import unmask
from tedana.due import due, Doi

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


@due.dcite(Doi('10.1002/(SICI)1522-2594(199907)42:1<87::AID-MRM13>3.0.CO;2-O'),
           description='T2* method of combining data across echoes using '
                       'monoexponential equation.')
def _combine_t2s(data, tes, ft2s):
    """
    Combine data across echoes using weighted averaging according to voxel-
    (and sometimes volume-) wise estimates of T2*.

    Parameters
    ----------
    data : (M x E x T) array_like
        Masked data.
    tes : (1 x E) array_like
        Echo times in milliseconds.
    ft2s : (M [x T] X 1) array_like
        Either voxel-wise or voxel- and volume-wise estimates of T2*.

    Returns
    -------
    combined : (M x T) :obj:`numpy.ndarray`
        Data combined across echoes according to T2* estimates.
    """
    RepLGR.info("Multi-echo data were then optimally combined using the "
                "T2* combination method (Posse et al., 1999).")
    RefLGR.info("Posse, S., Wiese, S., Gembris, D., Mathiak, K., Kessler, "
                "C., Grosse‐Ruyken, M. L., ... & Kiselev, V. G. (1999). "
                "Enhancement of BOLD‐contrast sensitivity by single‐shot "
                "multi‐echo functional MR imaging. Magnetic Resonance in "
                "Medicine: An Official Journal of the International Society "
                "for Magnetic Resonance in Medicine, 42(1), 87-97.")
    n_vols = data.shape[-1]
    alpha = tes * np.exp(-tes / ft2s)
    if alpha.ndim == 2:
        # Voxel-wise T2 estimates
        alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    elif alpha.ndim == 3:
        # Voxel- and volume-wise T2 estimates
        # alpha is currently (S, T, E) but should be (S, E, T) like mdata
        alpha = np.swapaxes(alpha, 1, 2)

        # If all values across echos are 0, set to 1 to avoid
        # divide-by-zero errors
        ax0_idx, ax2_idx = np.where(np.all(alpha == 0, axis=1))
        alpha[ax0_idx, :, ax2_idx] = 1.
    combined = np.average(data, axis=1, weights=alpha)
    return combined


@due.dcite(Doi('10.1002/mrm.20900'),
           description='PAID method of combining data across echoes using just '
                       'SNR/signal and TE.')
def _combine_paid(data, tes):
    """
    Combine data across echoes using SNR/signal and TE via the
    parallel-acquired inhomogeneity desensitized (PAID) ME-fMRI combination
    method.

    Parameters
    ----------
    data : (M x E x T) array_like
        Masked data.
    tes : (1 x E) array_like
        Echo times in milliseconds.

    Returns
    -------
    combined : (M x T) :obj:`numpy.ndarray`
        Data combined across echoes according to SNR/signal.
    """
    RepLGR.info("Multi-echo data were then optimally combined using the "
                "parallel-acquired inhomogeneity desensitized (PAID) "
                "combination method.")
    RefLGR.info("Poser, B. A., Versluis, M. J., Hoogduin, J. M., & Norris, "
                "D. G. (2006). BOLD contrast sensitivity enhancement and "
                "artifact reduction with multiecho EPI: parallel‐acquired "
                "inhomogeneity‐desensitized fMRI. "
                "Magnetic Resonance in Medicine: An Official Journal of the "
                "International Society for Magnetic Resonance in Medicine, "
                "55(6), 1227-1235.")
    n_vols = data.shape[-1]
    snr = data.mean(axis=-1) / data.std(axis=-1)
    alpha = snr * tes
    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    combined = np.average(data, axis=1, weights=alpha)
    return combined


def make_optcom(data, tes, adaptive_mask, t2s=None, combmode='t2s', verbose=True):
    """
    Optimally combine BOLD data across TEs, using only those echos with reliable signal
    across at least three echos. If the number of echos providing reliable signal is greater
    than three but less than the total number of collected echos, we assume that later
    echos do not provided meaningful signal.

    Parameters
    ----------
    data : (S x E x T) :obj:`numpy.ndarray`
        Concatenated BOLD data.
    tes : (E,) :obj:`numpy.ndarray`
        Array of TEs, in seconds.
    adaptive_mask : (S,) :obj:`numpy.ndarray`
        Adaptive mask of the data indicating the number of echos with signal at each voxel
    t2s : (S [x T]) :obj:`numpy.ndarray` or None, optional
        Estimated T2* values. Only required if combmode = 't2s'.
        Default is None.
    combmode : {'t2s', 'paid'}, optional
        How to combine data. Either 'paid' or 't2s'. If 'paid', argument 't2s'
        is not required. Default is 't2s'.
    verbose : :obj:`bool`, optional
        Whether to print status updates. Default is True.

    Returns
    -------
    combined : (S x T) :obj:`numpy.ndarray`
        Optimally combined data.

    Notes
    -----
    1.  Estimate voxel- and TE-specific weights based on estimated
        :math:`T_2^*`:

            .. math::
                w(T_2^*)_n = \\frac{TE_n * exp(\\frac{-TE}\
                {T_{2(est)}^*})}{\\sum TE_n * exp(\\frac{-TE}{T_{2(est)}^*})}
    2.  Perform weighted average per voxel and TR across TEs based on weights
        estimated in the previous step.
    """
    if data.ndim != 3:
        raise ValueError('Input data must be 3D (S x E x T)')

    if len(tes) != data.shape[1]:
        raise ValueError('Number of echos provided does not match second '
                         'dimension of input data: {0} != '
                         '{1}'.format(len(tes), data.shape[1]))

    if adaptive_mask.ndim != 1:
        raise ValueError('Mask is not 1D')
    elif adaptive_mask.shape[0] != data.shape[0]:
        raise ValueError('Mask and data do not have same number of '
                         'voxels/samples: {0} != {1}'.format(
                             adaptive_mask.shape[0], data.shape[0]))

    if combmode not in ['t2s', 'paid']:
        raise ValueError("Argument 'combmode' must be either 't2s' or 'paid'")
    elif combmode == 't2s' and t2s is None:
        raise ValueError("Argument 't2s' must be supplied if 'combmode' is "
                         "set to 't2s'.")
    elif combmode == 'paid' and t2s is not None:
        LGR.warning("Argument 't2s' is not required if 'combmode' is 'paid'. "
                    "'t2s' array will not be used.")

    if combmode == 'paid':
        LGR.info('Optimally combining data with parallel-acquired '
                 'inhomogeneity desensitized (PAID) method')
    else:
        if t2s.ndim == 1:
            msg = 'Optimally combining data with voxel-wise T2* estimates'
        else:
            msg = ('Optimally combining data with voxel- and volume-wise T2* '
                   'estimates')
        LGR.info(msg)

    mask = adaptive_mask >= 3
    data = data[mask, :, :]  # mask out unstable voxels/samples
    tes = np.array(tes)[np.newaxis, ...]  # (1 x E) array_like
    combined = np.zeros((data.shape[0], data.shape[2]))
    for echo in np.unique(adaptive_mask[mask]):
        echo_idx = adaptive_mask[mask] == echo

        if combmode == 'paid':
            combined[echo_idx, :] = _combine_paid(data[echo_idx, :echo, :],
                                                  tes[:echo])
        else:
            t2s_ = t2s[mask, ..., np.newaxis]  # mask out empty voxels/samples

            combined[echo_idx, :] = _combine_t2s(
                data[echo_idx, :echo, :], tes[:, :echo], t2s_[echo_idx, ...])

    combined = unmask(combined, mask)
    return combined
