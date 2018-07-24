"""
Functions to optimally combine data across echoes.
"""
import logging
import numpy as np
from tedana.utils import unmask
from tedana.due import due, BibTeX, Doi

LGR = logging.getLogger(__name__)


@due.dcite(BibTeX('@article{doi:10.1002/(SICI)1522-2594(199907)42:1<87::AID-'
                  'MRM13>3.0.CO;2-O,'
                  'author = {Posse Stefan and Wiese Stefan and Gembris Daniel '
                  'and Mathiak Klaus and Kessler Christoph and Grosse-Ruyken '
                  'Maria-Liisa and Elghahwagi Barbara and Richards Todd and '
                  'Dager Stephen R. and Kiselev Valerij G.},'
                  'title = {Enhancement of BOLD-contrast sensitivity by '
                  'single-shot multi-echo functional MR imaging},'
                  'journal = {Magnetic Resonance in Medicine},'
                  'volume = {42},'
                  'number = {1},'
                  'pages = {87-97},'
                  'doi = {10.1002/(SICI)1522-2594(199907)42:1<87::AID-MRM13>'
                  '3.0.CO;2-O},'
                  'url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/'
                  '%28SICI%291522-2594%28199907%2942%3A1%3C87%3A%3AAID-'
                  'MRM13%3E3.0.CO%3B2-O}}'),
           description='T2* method of combining data across echoes using '
                       'monoexponential equation.')
def _combine_t2s(mdata, ft2s, tes, n_vols):
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
    return combined


@due.cite(Doi('10.1002/mrm.20900'),
         description='STE method of combining data across echoes using just '
                     'SNR/signal and TE.')
def _combine_ste(mdata, tes, n_vols):
    alpha = mdata.mean(axis=-1) * tes
    alpha = np.tile(alpha[:, :, np.newaxis], (1, 1, n_vols))
    combined = np.average(mdata, axis=1, weights=alpha)
    return combined


def make_optcom(data, tes, mask, t2s=None, combmode='t2s', verbose=True):
    """
    Optimally combine BOLD data across TEs.

    Parameters
    ----------
    data : (S x E x T) :obj:`numpy.ndarray`
        Concatenated BOLD data.
    tes : (E,) :obj:`numpy.ndarray`
        Array of TEs, in seconds.
    mask : (S,) :obj:`numpy.ndarray`
        Brain mask in 3D array.
    t2s : (S [x T]) :obj:`numpy.ndarray` or None, optional
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
    if data.ndim != 3:
        raise ValueError('Input data must be 3D (S x E x T)')

    if len(tes) != data.shape[1]:
        raise ValueError('Number of echos provided does not match second '
                         'dimension of input data: {0} != '
                         '{1}'.format(len(tes), data.shape[1]))

    if mask.ndim != 1:
        raise ValueError('Mask is not 1D')
    elif mask.shape[0] != data.shape[0]:
        raise ValueError('Mask and data do not have same number of '
                         'voxels/samples: {0} != {1}'.format(mask.shape[0],
                                                             data.shape[0]))

    if combmode not in ['t2s', 'ste']:
        raise ValueError("Argument 'combmode' must be either 't2s' or 'ste'")
    elif combmode == 't2s' and t2s is None:
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
        msg = ('Optimally combining data with voxel- and volume-wise T2 '
               'estimates')
        ft2s = t2s[mask, :, np.newaxis]

    if verbose:
        LGR.info(msg)

    if combmode == 'ste':
        combined = _combine_ste(mdata, tes, n_vols)
    else:
        combined = _combine_t2s(mdata, ft2s, tes, n_vols)

    combined = unmask(combined, mask)
    return combined
