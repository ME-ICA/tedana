"""
Functions to identify TE-dependent and TE-independent components.
"""
import logging
import numpy as np

from tedana import utils
from tedana.selection._utils import (getelbow_cons, getelbow)

LGR = logging.getLogger(__name__)

F_MAX = 500


def kundu_tedpca(comptable, n_echos, kdaw, rdaw, stabilize=False):
    """
    Select PCA components using Kundu's decision tree approach.

    Parameters
    ----------
    comptable : :obj:`pandas.DataFrame`
        Component table with relevant metrics: kappa, rho, and normalized
        variance explained. Component number should be the index.
    n_echos : :obj:`int`
        Number of echoes in dataset.
    kdaw : :obj:`float`
        Kappa dimensionality augmentation weight. Must be a non-negative float,
        or -1 (a special value).
    rdaw : :obj:`float`
        Rho dimensionality augmentation weight. Must be a non-negative float,
        or -1 (a special value).
    stabilize : :obj:`bool`, optional
        Whether to stabilize convergence by reducing dimensionality, for low
        quality data. Default is False.

    Returns
    -------
    comptable : :obj:`pandas.DataFrame`
        Component table with components classified as 'accepted', 'rejected',
        or 'ignored'.
    """
    eigenvalue_elbow = getelbow(comptable['normalized variance explained'],
                                return_val=True)

    diff_varex_norm = np.abs(np.diff(comptable['normalized variance explained']))
    lower_diff_varex_norm = diff_varex_norm[(len(diff_varex_norm) // 2):]
    varex_norm_thr = np.mean([lower_diff_varex_norm.max(),
                              diff_varex_norm.min()])
    varex_norm_min = comptable['normalized variance explained'][
        (len(diff_varex_norm) // 2) +
        np.arange(len(lower_diff_varex_norm))[lower_diff_varex_norm >= varex_norm_thr][0] + 1]
    varex_norm_cum = np.cumsum(comptable['normalized variance explained'])

    fmin, fmid, fmax = utils.getfbounds(n_echos)
    if int(kdaw) == -1:
        lim_idx = utils.andb([comptable['kappa'] < fmid,
                              comptable['kappa'] > fmin]) == 2
        kappa_lim = comptable.loc[lim_idx, 'kappa'].values
        kappa_thr = kappa_lim[getelbow(kappa_lim)]

        lim_idx = utils.andb([comptable['rho'] < fmid, comptable['rho'] > fmin]) == 2
        rho_lim = comptable.loc[lim_idx, 'rho'].values
        rho_thr = rho_lim[getelbow(rho_lim)]
        stabilize = True
        LGR.info('kdaw set to -1. Switching TEDPCA method to '
                 'kundu-stabilize')
    elif int(rdaw) == -1:
        lim_idx = utils.andb([comptable['rho'] < fmid, comptable['rho'] > fmin]) == 2
        rho_lim = comptable.loc[lim_idx, 'rho'].values
        rho_thr = rho_lim[getelbow(rho_lim)]
    else:
        kappa_thr = np.average(
            sorted([fmin, (getelbow(comptable['kappa'], return_val=True) / 2), fmid]),
            weights=[kdaw, 1, 1])
        rho_thr = np.average(
            sorted([fmin, (getelbow_cons(comptable['rho'], return_val=True) / 2), fmid]),
            weights=[rdaw, 1, 1])

    # Reject if low Kappa, Rho, and variance explained
    is_lowk = comptable['kappa'] <= kappa_thr
    is_lowr = comptable['rho'] <= rho_thr
    is_lowe = comptable['normalized variance explained'] <= eigenvalue_elbow
    is_lowkre = is_lowk & is_lowr & is_lowe
    comptable.loc[is_lowkre, 'classification'] = 'rejected'
    comptable.loc[is_lowkre, 'rationale'] += 'P001;'

    # Reject if low variance explained
    is_lows = comptable['normalized variance explained'] <= varex_norm_min
    comptable.loc[is_lows, 'classification'] = 'rejected'
    comptable.loc[is_lows, 'rationale'] += 'P002;'

    # Reject if Kappa over limit
    is_fmax1 = comptable['kappa'] == F_MAX
    comptable.loc[is_fmax1, 'classification'] = 'rejected'
    comptable.loc[is_fmax1, 'rationale'] += 'P003;'

    # Reject if Rho over limit
    is_fmax2 = comptable['rho'] == F_MAX
    comptable.loc[is_fmax2, 'classification'] = 'rejected'
    comptable.loc[is_fmax2, 'rationale'] += 'P004;'

    if stabilize:
        temp7 = varex_norm_cum >= 0.95
        comptable.loc[temp7, 'classification'] = 'rejected'
        comptable.loc[temp7, 'rationale'] += 'P005;'
        under_fmin1 = comptable['kappa'] <= fmin
        comptable.loc[under_fmin1, 'classification'] = 'rejected'
        comptable.loc[under_fmin1, 'rationale'] += 'P006;'
        under_fmin2 = comptable['rho'] <= fmin
        comptable.loc[under_fmin2, 'classification'] = 'rejected'
        comptable.loc[under_fmin2, 'rationale'] += 'P007;'

    n_components = comptable.loc[comptable['classification'] == 'accepted'].shape[0]
    LGR.info('Selected {0} components with Kappa threshold: {1:.02f}, Rho '
             'threshold: {2:.02f}'.format(n_components, kappa_thr, rho_thr))

    # Move decision columns to end
    cols_at_end = ['classification', 'rationale']
    comptable = comptable[[c for c in comptable if c not in cols_at_end] +
                          [c for c in cols_at_end if c in comptable]]
    comptable['rationale'] = comptable['rationale'].str.rstrip(';')
    return comptable
