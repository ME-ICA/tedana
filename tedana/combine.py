"""
Functions to optimally combine data across echoes.
"""
import logging

import numpy as np

from tedana.due import Doi, due

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")


@due.dcite(
    Doi("10.1002/(SICI)1522-2594(199907)42:1<87::AID-MRM13>3.0.CO;2-O"),
    description="T2* method of combining data across echoes using monoexponential equation.",
)
def _combine_t2s(data, tes, ft2s, report=True):
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
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    combined : (M x T) :obj:`numpy.ndarray`
        Data combined across echoes according to T2* estimates.

    References
    ----------
    * Posse, S., Wiese, S., Gembris, D., Mathiak, K., Kessler,
      C., Grosse‐Ruyken, M. L., ... & Kiselev, V. G. (1999).
      Enhancement of BOLD‐contrast sensitivity by single‐shot
      multi‐echo functional MR imaging. Magnetic Resonance in
      Medicine: An Official Journal of the International Society
      for Magnetic Resonance in Medicine, 42(1), 87-97.
    """
    if report:
        RepLGR.info(
            "Multi-echo data were then optimally combined using the "
            "T2* combination method (Posse et al., 1999)."
        )
        RefLGR.info(
            "Posse, S., Wiese, S., Gembris, D., Mathiak, K., Kessler, "
            "C., Grosse‐Ruyken, M. L., ... & Kiselev, V. G. (1999). "
            "Enhancement of BOLD‐contrast sensitivity by single‐shot "
            "multi‐echo functional MR imaging. Magnetic Resonance in "
            "Medicine: An Official Journal of the International Society "
            "for Magnetic Resonance in Medicine, 42(1), 87-97."
        )
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
        alpha[ax0_idx, :, ax2_idx] = 1.0
    combined = np.average(data, axis=1, weights=alpha)
    return combined


@due.dcite(
    Doi("10.1002/mrm.20900"),
    description="PAID method of combining data across echoes using just SNR/signal and TE.",
)
def _combine_paid(data, tes, report=True):
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
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    combined : (M x T) :obj:`numpy.ndarray`
        Data combined across echoes according to SNR/signal.

    References
    ----------
    * Poser, B. A., Versluis, M. J., Hoogduin, J. M., & Norris,
      D. G. (2006). BOLD contrast sensitivity enhancement and
      artifact reduction with multiecho EPI: parallel‐acquired
      inhomogeneity‐desensitized fMRI.
      Magnetic Resonance in Medicine: An Official Journal of the
      International Society for Magnetic Resonance in Medicine,
      55(6), 1227-1235.
    """
    if report:
        RepLGR.info(
            "Multi-echo data were then optimally combined using the "
            "parallel-acquired inhomogeneity desensitized (PAID) "
            "combination method."
        )
        RefLGR.info(
            "Poser, B. A., Versluis, M. J., Hoogduin, J. M., & Norris, "
            "D. G. (2006). BOLD contrast sensitivity enhancement and "
            "artifact reduction with multiecho EPI: parallel‐acquired "
            "inhomogeneity‐desensitized fMRI. "
            "Magnetic Resonance in Medicine: An Official Journal of the "
            "International Society for Magnetic Resonance in Medicine, "
            "55(6), 1227-1235."
        )
    n_vols = data.shape[-1]
    snr = data.mean(axis=-1) / data.std(axis=-1)
    alpha = snr * tes
    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    combined = np.average(data, axis=1, weights=alpha)
    return combined


def make_optcom(data, tes, adaptive_mask, t2s=None, combmode="t2s", verbose=True):
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
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
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
    This function supports both the ``'t2s'`` method [1]_ and the ``'paid'``
    method [2]_. The ``'t2s'`` method operates according to the following
    logic:

    1.  Estimate voxel- and TE-specific weights based on estimated :math:`T_2^*`:

            .. math::
                w(T_2^*)_n = \\frac{TE_n * exp(\\frac{-TE}\
                {T_{2(est)}^*})}{\\sum TE_n * exp(\\frac{-TE}{T_{2(est)}^*})}
    2.  Perform weighted average per voxel and TR across TEs based on weights
        estimated in the previous step.

    References
    ----------
    .. [1] Posse, S., Wiese, S., Gembris, D., Mathiak, K., Kessler,
           C., Grosse‐Ruyken, M. L., ... & Kiselev, V. G. (1999).
           Enhancement of BOLD‐contrast sensitivity by single‐shot
           multi‐echo functional MR imaging. Magnetic Resonance in
           Medicine: An Official Journal of the International Society
           for Magnetic Resonance in Medicine, 42(1), 87-97.
    .. [2] Poser, B. A., Versluis, M. J., Hoogduin, J. M., & Norris,
           D. G. (2006). BOLD contrast sensitivity enhancement and
           artifact reduction with multiecho EPI: parallel‐acquired
           inhomogeneity‐desensitized fMRI.
           Magnetic Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine,
           55(6), 1227-1235.

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                              parameter.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be 3D (S x E x T)")

    if len(tes) != data.shape[1]:
        raise ValueError(
            "Number of echos provided does not match second "
            "dimension of input data: {0} != "
            "{1}".format(len(tes), data.shape[1])
        )

    if adaptive_mask.ndim != 1:
        raise ValueError("Mask is not 1D")
    elif adaptive_mask.shape[0] != data.shape[0]:
        raise ValueError(
            "Mask and data do not have same number of "
            "voxels/samples: {0} != {1}".format(adaptive_mask.shape[0], data.shape[0])
        )

    if combmode not in ["t2s", "paid"]:
        raise ValueError("Argument 'combmode' must be either 't2s' or 'paid'")
    elif combmode == "t2s" and t2s is None:
        raise ValueError("Argument 't2s' must be supplied if 'combmode' is set to 't2s'.")
    elif combmode == "paid" and t2s is not None:
        LGR.warning(
            "Argument 't2s' is not required if 'combmode' is 'paid'. "
            "'t2s' array will not be used."
        )

    if combmode == "paid":
        LGR.info(
            "Optimally combining data with parallel-acquired "
            "inhomogeneity desensitized (PAID) method"
        )
    else:
        if t2s.ndim == 1:
            msg = "Optimally combining data with voxel-wise T2* estimates"
        else:
            msg = "Optimally combining data with voxel- and volume-wise T2* estimates"
        LGR.info(msg)

    echos_to_run = np.unique(adaptive_mask)
    # When there is one good echo, use two
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    tes = np.array(tes)[np.newaxis, ...]  # (1 x E) array_like
    combined = np.zeros((data.shape[0], data.shape[2]))
    report = True
    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
            # Use the first two echoes for cases where there are
            # either one or two good echoes
            voxel_idx = np.where(np.logical_and(adaptive_mask > 0, adaptive_mask <= echo_num))[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        if combmode == "paid":
            combined[voxel_idx, :] = _combine_paid(
                data[voxel_idx, :echo_num, :], tes[:, :echo_num]
            )
        else:
            t2s_ = t2s[..., np.newaxis]  # add singleton

            combined[voxel_idx, :] = _combine_t2s(
                data[voxel_idx, :echo_num, :],
                tes[:, :echo_num],
                t2s_[voxel_idx, ...],
                report=report,
            )
        report = False

    return combined
