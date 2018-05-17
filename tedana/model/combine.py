"""
Functions to optimally combine data across echoes.
"""
import logging
import numpy as np
from tedana import utils

LGR = logging.getLogger(__name__)


def make_optcom(data, t2s, tes, mask, combmode, verbose=True):
    """
    Optimally combine BOLD data across TEs.

    Parameters
    ----------
    data : (S x E x T) :obj:`numpy.ndarray`
        Concatenated BOLD data.
    t2 : (S,) :obj:`numpy.ndarray`
        Estimated T2* values.
    tes : :obj:`numpy.ndarray`
        Array of TEs, in seconds.
    mask : (S,) :obj:`numpy.ndarray`
        Brain mask in 3D array.
    combmode : :obj:`str`
        How to combine data. Either 'ste' or 't2s'.
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

    _, _, n_vols = data.shape
    mdata = data[mask]
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
    else:
        alpha = tes * np.exp(-tes / ft2s)

    if t2s.ndim == 1:
        alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    else:
        alpha = np.swapaxes(alpha, 1, 2)
        ax0_idx, ax2_idx = np.where(np.all(alpha == 0, axis=1))
        alpha[ax0_idx, :, ax2_idx] = 1.

    combined = np.average(mdata, axis=1, weights=alpha)
    combined = utils.unmask(combined, mask)

    return combined
