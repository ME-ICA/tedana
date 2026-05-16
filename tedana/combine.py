"""Functions to optimally combine data across echoes."""

import logging

import numpy as np

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def _combine_r2s(data, tes, fr2s, report=True):
    """Combine data across echoes using weighted averaging according to estimates of R2*.

    The R2* estimates may be voxel- or voxel- and volume-wise.

    This method was proposed in :footcite:t:`posse1999enhancement`.

    Parameters
    ----------
    data : (Mb x E x T) array_like
        Masked data.
    tes : (1 x E) array_like
        Echo times in seconds.
    fr2s : (Mb [x T] X 1) array_like
        Either voxel-wise or voxel- and volume-wise estimates of R2* in s⁻¹.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    combined : (Mb x T) :obj:`numpy.ndarray`
        Data combined across echoes according to R2* estimates.

    References
    ----------
    .. footbibliography::
    """
    if report:
        RepLGR.info(
            "Multi-echo data were then optimally combined using the "
            "R2* combination method \\citep{posse1999enhancement}."
        )

    n_vols = data.shape[-1]
    alpha = tes * np.exp(-tes * fr2s)
    if alpha.ndim == 2:
        # Voxel-wise R2* estimates
        alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    elif alpha.ndim == 3:
        # Voxel- and volume-wise R2* estimates
        # alpha is currently (S, T, E) but should be (S, E, T) like mdata
        alpha = np.swapaxes(alpha, 1, 2)

        # If all values across echos are 0, set to 1 to avoid
        # divide-by-zero errors
        ax0_idx, ax2_idx = np.where(np.all(alpha == 0, axis=1))
        alpha[ax0_idx, :, ax2_idx] = 1.0

    combined = np.average(data, axis=1, weights=alpha)
    return combined


def _combine_paid(data, tes, report=True):
    """Combine data across echoes using the PAID combination method.

    This method uses SNR/signal and TE via the parallel-acquired inhomogeneity desensitized (PAID)
    ME-fMRI combination method :footcite:t:`poser2006bold`.

    Parameters
    ----------
    data : (Mb x E x T) array_like
        Masked data, where `Mb` is samples in base mask, `E` is echos, and `T` is time.
    tes : (1 x E) array_like
        Echo times in seconds.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    combined : (Mb x T) :obj:`numpy.ndarray`
        Data combined across echoes according to SNR/signal.

    References
    ----------
    .. footbibliography::
    """
    if report:
        RepLGR.info(
            "Multi-echo data were then optimally combined using the "
            "parallel-acquired inhomogeneity desensitized (PAID) "
            "combination method \\citep{poser2006bold}."
        )

    n_vols = data.shape[-1]
    mean_signal = data.mean(axis=-1)
    signal_std = data.std(axis=-1)
    snr = np.zeros_like(mean_signal, dtype=float)
    np.divide(mean_signal, signal_std, out=snr, where=signal_std != 0)
    alpha = snr * tes
    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    combined = np.average(data, axis=1, weights=alpha)
    return combined


def make_optcom(data, tes, adaptive_mask, r2s=None, combmode="r2s"):
    r"""Optimally combine BOLD data across TEs.

    Optimally combine BOLD data across TEs, using only those echos with reliable signal
    across at least three echos. If the number of echos providing reliable signal is greater
    than three but less than the total number of collected echos, we assume that later
    echos do not provided meaningful signal.

    Parameters
    ----------
    data : (Mb x E x T) :obj:`numpy.ndarray`
        Concatenated BOLD data, where `Mb` is samples in base mask, `E` is echos, and `T` is time.
    tes : (E,) :obj:`numpy.ndarray`
        Array of TEs, in seconds.
    adaptive_mask : (Mb,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    r2s : (Mb [x T]) :obj:`numpy.ndarray` or None, optional
        Estimated R2* values in s⁻¹. Only required if combmode = 'r2s'.
        Default is None.
    combmode : {'r2s', 'paid'}, optional
        How to combine data. Either 'paid' or 'r2s'. If 'paid', argument 'r2s'
        is not required. Default is 'r2s'.

    Returns
    -------
    combined : (Mb x T) :obj:`numpy.ndarray`
        Optimally combined data.

    Notes
    -----
    This function supports both the ``'r2s'`` method :footcite:p:`posse1999enhancement`
    and the ``'paid'`` method :footcite:p:`poser2006bold`.
    The ``'r2s'`` method operates according to the following logic:

    1.  Estimate voxel- and TE-specific weights based on estimated :math:`R_2^*`:

            .. math::
                w(R_2^*)_n = \frac{TE_n * exp(-TE * R_{2(est)}^*)}{\sum TE_n * exp(-TE * R_{2(est)}^*)}
    2.  Perform weighted average per voxel and TR across TEs based on weights
        estimated in the previous step.

    References
    ----------
    .. footbibliography::

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
            f"dimension of input data: {len(tes)} != {data.shape[1]}"
        )

    if adaptive_mask.ndim != 1:
        raise ValueError("Mask is not 1D")
    elif adaptive_mask.shape[0] != data.shape[0]:
        raise ValueError(
            "Mask and data do not have same number of "
            f"voxels/samples: {adaptive_mask.shape[0]} != {data.shape[0]}"
        )

    if combmode not in ["r2s", "paid"]:
        raise ValueError("Argument 'combmode' must be either 'r2s' or 'paid'")
    elif combmode == "r2s" and r2s is None:
        raise ValueError("Argument 'r2s' must be supplied if 'combmode' is set to 'r2s'.")
    elif combmode == "paid" and r2s is not None:
        LGR.warning(
            "Argument 'r2s' is not required if 'combmode' is 'paid'. "
            "'r2s' array will not be used."
        )

    if combmode == "paid":
        LGR.info(
            "Optimally combining data with parallel-acquired "
            "inhomogeneity desensitized (PAID) method"
        )
    else:
        if r2s.shape[0] != data.shape[0]:
            raise ValueError(
                "R2* estimates and data do not have same number of "
                f"voxels/samples: {r2s.shape[0]} != {data.shape[0]}"
            )

        if r2s.ndim == 1:
            msg = "Optimally combining data with voxel-wise R2* estimates"
        else:
            msg = "Optimally combining data with voxel- and volume-wise R2* estimates"
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
            r2s_ = r2s[..., np.newaxis]  # add singleton

            combined[voxel_idx, :] = _combine_r2s(
                data[voxel_idx, :echo_num, :],
                tes[:, :echo_num],
                r2s_[voxel_idx, ...],
                report=report,
            )
        report = False

    return combined
