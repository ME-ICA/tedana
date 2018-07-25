"""
Functions to optimally combine data across echoes.
"""
import logging
import numpy as np
from tedana import utils

LGR = logging.getLogger(__name__)


def make_optcom(data, tes, mask, t2s=None, combmode='t2s', verbose=True):
    """
    Optimally combine BOLD data across TEs.

    Parameters
    ----------
    data : (S x E x T) :obj:`numpy.ndarray`
        Concatenated BOLD data.
    tes : :obj:`numpy.ndarray`
        Array of TEs, in seconds.
    mask : (S,) :obj:`numpy.ndarray`
        Brain mask in 3D array.
    t2s : (S,) or (S x T) :obj:`numpy.ndarray` or None, optional
        Estimated T2* values. Only required if combmode = 't2s'.
        Default is None.
    combmode : {'t2s', 'ste'}
        How to combine data. Either 'ste' or 't2s'. If 'ste', argument 't2s' is
        not required.
    verbose : :obj:`bool`, optional
        Whether to print status updates

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
                {T_{2(est)}^*})}{\sum TE_n * exp(\\frac{-TE}{T_{2(est)}^*})}
    2.  Perform weighted average per voxel and TR across TEs based on weights
        estimated in the previous step.
    """
    if combmode == 't2s' and t2s is None:
        raise ValueError("Argument 't2s' must be supplied if 'combmode' is "
                         "set to 't2s'.")
    elif combmode == 'ste' and t2s is not None:
        LGR.warning("Argument 't2s' is not required if 'combmode' is 'ste'.")

    _, _, n_vols = data.shape
    mdata = data[mask, :, :]  # mask out empty voxels/samples
    tes = np.array(tes)[np.newaxis]  # (1 x E) array_like

    if t2s.ndim == 1:
        msg = 'Optimally combining data with voxel-wise T2 estimates'
        ft2s = t2s[mask, np.newaxis]
    else:
        msg = 'Optimally combining data with voxel- and volume-wise T2 estimates'
        ft2s = t2s[mask, :, np.newaxis]

    if verbose:
        LGR.info(msg)

    if combmode == 'ste':
        alpha = mdata.mean(axis=-1) * tes
        alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    else:
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

    combined = np.average(mdata, axis=1, weights=alpha)
    combined = utils.unmask(combined, mask)

    return combined
