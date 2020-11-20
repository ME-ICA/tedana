"""
Handle metric dependence and calculations
"""
import logging

import numpy as np
from scipy import stats

from tedana import io, utils
from tedana.stats import computefeats2, get_coeffs, t_to_z


LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


import numpy as np

class Metric:
    table = {}
    def __init__(self, name, dependencies=[]):
        self.name = name
        self.calculate = eval(name)
        self.dependencies = dependencies
        self.value = None
        Metric.table[name] = self
    def get(self):
        if self.value is not None:
            return self.value
        else:
            Metric.resolve_dependencies(self.name)
            self.value = self.calculate()
            return self.value
    def calculate(self):
        self.calculate()
    def fill(data_optcom, mixing, mask, data_cat, adaptive_mask, tes,
             ref_img):
        fill_metrics()
        Metric.data_optcom = data_optcomb[mask, :]
        Metric.mixing = mixing.copy()
	Metric.data_cat = data_cat[mask, ...]
	Metric.adaptive_mask = adaptive_mask[mask]
        Metric.tes = tes
        Metric.ref_img = ref_img
        Metric.n_components = mixing.shape[1]
        Metric.comptable = pd.DataFrame(index=np.arange(n_components,
                                        dtype=int))
    def resolve_dependencies(metric):
        dependencies = Metric.str2obj(metric).dependencies
        for d in dependencies:
            Metric.str2obj(d).get()
    def str2obj(metric):
        return Metric.table[metric]
    def get_metric(metric):
        return Metric.table[metric]
    def get_metric_value(metric):
        return Metric.table[metric].get()


def fill_metrics():
    Metric('kappa', ['map_FT2', 'map_Z'])
    Metric('rho', ['map_FS0', 'map_Z'])
    Metric('countnoise', ['map_Z', 'map_Z_clusterized'])
    Metric('countsigFT2', ['map_FT2_clusterized'])
    Metric('countsigFS0', ['map_FS0_clusterized'])
    Metric('dice_FT2', ['map_beta T2_clusterized', 'map_FT2_clusterized'])
    Metric('dice_FS0', ['map_beta S0_clusterized', 'map_FS0_clusterized'])
    Metric('signal_noise_t', ['map_Z', 'map_Z_clusterized', 'map_FT2'])
    Metric('variance_explained', ['map_optcom betas'])
    Metric('normalized_variance explained', ['map_weight'])
    Metric('d_table_score', ['kappa', 'dice_FT2', 'signal_noise_t',
    			     'countnoise', 'countsigFT2'])
    Metric('map_FT2', ['map_Z', 'mixing', 'tes', 'data_cat', 
			'adaptive_mask'])
    Metric('map_FS0', ['map_Z', 'mixing', 'tes', 'data_cat',
		       'adaptive_mask'])
    Metric('map_Z', ['map_weight'])
    Metric('map_weight', ['data_optcom', 'mixing'])
    Metric('map_optcom_betas', ['data_optcom', 'mixing'])
    Metric('map_percent_signal_change', ['data_optcom', 'map_optcom_betas'])
    Metric('map_Z_clusterized', ['map_Z', 'mask', 'ref_img', 'tes'])
    Metric('map_FT2_clusterized', ['map_FT2', 'mask', 'ref_img', 'tes'])
    Metric('map_FS0_clusterized', ['map_FS0', 'mask', 'ref_img', 'tes'])
    Metric('map_beta T2_clusterized', ['map_FT2_clusterized', 
				       'map_optcom_betas','countsigFT2',
				       'mask', 'ref_img', 'tes'])
    Metric('map_beta_S0_clusterized', ['map_FS0_clusterized',
				       'map_optcom_betas',
				       'countsigFS0', 'mask', 'ref_img',
                                       'tes'])


def map_weights():
    """
    Calculate standardized parameter estimates between data, mixing matrix.

    Returns
    _------
    weights : (M x C) array_like
        Standardized parameter estimates for optimally combined data against
        the mixing matrix.
    """

    mixing_z = stats.zscore(Metric.mixing, axis=0)
    # compute un-normalized weight dataset (features)
    weights = computefeats2(Metric.data_optcom, mixing_z, normalize=False)
    signs = determine_signs(weights, axis=0)
    weights, mixing = flip_components(weights, mixing, signs=signs)
    Metric.mixing = mixing.copy()
    return weights


def map_optcom_betas():
    """ 
    Calculate unstandardized parameter estimates between data and mixing
    matrix.

    Parameters
    ----------
    data_optcom : (M x T) array_like
        Optimally combined data
    mixing : (T x C) array_like
        Mixing matrix

    Returns
    -------
    betas : (M x C) array_like
        Unstandardized parameter estimates
    """
    data_optcom = Metric.data_optcom
    mixing = Metric.mixing
    assert data_optcom.shape[1] == mixing.shape[0]
    # demean optimal combination
    data_optcom_dm = data_optcom - data_optcom.mean(axis=-1, keepdims=True)
    # compute PSC dataset - shouldn't have to refit data
    betas = get_coeffs(data_optcom_dm, mixing)
    return betas

def map_percent_signal_change():
    """ 
    Calculate percent signal change maps for components against optimally
    combined data.

    Returns
    -------
    psc : (M x C) array_like
        Component-wise percent signal change maps.
    """
    data_optcom = Metric.data_optcom
    optcom_betas = Metric.get('optcom_betas')
    assert data_optcom.shape[0] == optcom_betas.shape[0]
    psc = 100 * optcom_betas / data_optcom.mean(axis=-1, keepdims=True)
    return psc


def map_optcom_betas():
    """
    Calculate unstandardized parameter estimates between data and mixing
    matrix.

    Returns
    -------
    betas : (M x C) array_like
        Unstandardized parameter estimates
    """
    data_optcom = Metric.data_optcom
    mixing = Metric.mixing
    assert data_optcom.shape[1] == mixing.shape[0]
    # demean optimal combination
    data_optcom_dm = data_optcom - data_optcom.mean(axis=-1, keepdims=True)
    # compute PSC dataset - shouldn't have to refit data
    betas = get_coeffs(data_optcom_dm, mixing)
    return betas


def map_percent_signal_change():
    """
    Calculate percent signal change maps for components against optimally
    combined data.

    Returns
    -------
    psc : (M x C) array_like
        Component-wise percent signal change maps.
    """
    data_optcom = Metric.data_optcom
    optcom_betas = Metric.get('optcom_betas')
    assert data_optcom.shape[0] == optcom_betas.shape[0]
    psc = 100 * optcom_betas / data_optcom.mean(axis=-1, keepdims=True)
    return psc


def map_Z():
    """
    Calculate z-statistic maps by z-scoring standardized parameter estimate
    maps and cropping extreme values.

    Returns
    -------
    Z_maps : (M x C) array_like
        Z-statistic maps for components, reflecting voxel-wise component loadings.
    """
    z_max = 8
    Z_maps = stats.zscore(weights, axis=0)
    extreme_idx = np.abs(Z_maps) > z_max
    Z_maps[extreme_idx] = z_max * np.sign(Z_maps[extreme_idx])
    return Z_maps
