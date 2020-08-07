"""
Collect metrics.
"""
import logging

import numpy as np
import pandas as pd

from . import dependence
from ._utils import (determine_signs, flip_components, sort_df, apply_sort,
                     dependency_resolver)
from tedana.stats import getfbounds


LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def generate_metrics(data_cat, data_optcom, mixing, mask, adaptive_mask,
                     tes, ref_img,
                     metrics=None, sort_by='kappa', ascending=False):
    """
    Fit TE-dependence and -independence models to components.

    Parameters
    ----------
    data_cat : (S x E x T) array_like
        Input data, where `S` is samples, `E` is echos, and `T` is time
    data_optcom : (S x T) array_like
        Optimally combined data
    mixing : (T x C) array_like
        Mixing matrix for converting input data to component space, where `C`
        is components and `T` is the same as in `data_cat`
    mask : (S) array_like
        Boolean mask
    adaptive_mask : (S) array_like
        Adaptive mask, where each voxel's value is the number of echoes with
        "good signal".
    tes : list
        List of echo times associated with `data_cat`, in milliseconds
    ref_img : str or img_like
        Reference image to dictate how outputs are saved to disk
    metrics : list
        List of metrics to return
    sort_by : str, optional
        Metric to sort component table by. Default is 'kappa'.
    ascending : bool, optional
        Whether to sort the table in ascending or descending order.
        Default is False.

    Returns
    -------
    comptable : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for
        each metric. The index is the component number.
    mixing : :obj:`numpy.ndarray`
        Mixing matrix after sign flipping and sorting.
    """
    if metrics is None:
        metrics = ['map weight']
    RepLGR.info('The following metrics were calculated: {}.'.format(', '.join(metrics)))

    if not (data_cat.shape[0] == data_optcom.shape[0] == adaptive_mask.shape[0] ==
            mask.shape[0]):
        raise ValueError('First dimensions (number of samples) of data_cat ({0}), '
                         'data_optcom ({1}), adaptive_mask ({2}), and mask ({3}) do not '
                         'match'.format(data_cat.shape[0], data_optcom.shape[0],
                                        adaptive_mask.shape[0], mask.shape[0]))
    elif data_cat.shape[1] != len(tes):
        raise ValueError('Second dimension of data_cat ({0}) does not match '
                         'number of echoes provided (tes; '
                         '{1})'.format(data_cat.shape[1], len(tes)))
    elif not (data_cat.shape[2] == data_optcom.shape[1] == mixing.shape[0]):
        raise ValueError('Number of volumes in data_cat ({0}), '
                         'data_optcom ({1}), and mixing ({2}) do not '
                         'match.'.format(data_cat.shape[2], data_optcom.shape[1],
                                         mixing.shape[0]))

    INPUTS = ['data_cat', 'data_optcom', 'mixing', 'adaptive_mask',
              'mask', 'tes', 'ref_img']
    METRIC_DEPENDENCIES = {
        'kappa': ['map FT2', 'map Z'],
        'rho': ['map FS0', 'map Z'],
        'countnoise': ['map Z', 'map Z clusterized'],
        'countsigFT2': ['map FT2 clusterized'],
        'countsigFS0': ['map FS0 clusterized'],
        'dice_FT2': ['map beta T2 clusterized', 'map FT2 clusterized'],
        'dice_FS0': ['map beta S0 clusterized', 'map FS0 clusterized'],
        'signal-noise_t': ['map Z', 'map Z clusterized', 'map FT2'],
        'variance explained': ['map optcom betas'],
        'normalized variance explained': ['map weight'],
        'd_table_score': ['kappa', 'dice_FT2', 'signal-noise_t',
                          'countnoise', 'countsigFT2'],
        'map FT2': ['map Z', 'mixing', 'tes', 'data_cat', 'adaptive_mask'],
        'map FS0': ['map Z', 'mixing', 'tes', 'data_cat', 'adaptive_mask'],
        'map Z': ['map weight'],
        'map weight': ['data_optcom', 'mixing'],
        'map optcom betas': ['data_optcom', 'mixing'],
        'map percent signal change': ['data_optcom', 'map optcom betas'],
        'map Z clusterized': ['map Z', 'mask', 'ref_img', 'tes'],
        'map FT2 clusterized': ['map FT2', 'mask', 'ref_img', 'tes'],
        'map FS0 clusterized': ['map FS0', 'mask', 'ref_img', 'tes'],
        'map beta T2 clusterized': ['map FT2 clusterized', 'map optcom betas',
                                    'countsigFT2', 'mask', 'ref_img', 'tes'],
        'map beta S0 clusterized': ['map FS0 clusterized', 'map optcom betas',
                                    'countsigFS0', 'mask', 'ref_img', 'tes'],
    }
    # Apply masks before anything else
    data_cat = data_cat[mask, ...]
    data_optcom = data_optcom[mask, :]
    adaptive_mask = adaptive_mask[mask]

    required_metrics = dependency_resolver(METRIC_DEPENDENCIES, metrics, INPUTS)

    # Use copy to avoid changing the original variable outside of this function
    mixing = mixing.copy()

    # Generate the component table, which will be filled out, column by column,
    # throughout this function
    n_components = mixing.shape[1]
    comptable = pd.DataFrame(index=np.arange(n_components, dtype=int))

    # Metric maps
    # Maps will be stored as arrays in an easily-indexable dictionary
    metric_maps = {}
    if 'map weight' in required_metrics:
        LGR.info('Calculating weight maps')
        metric_maps['map weight'] = dependence.calculate_weights(data_optcom, mixing)
        signs = determine_signs(metric_maps['map weight'], axis=0)
        metric_maps['map weight'], mixing = flip_components(
            metric_maps['map weight'], mixing, signs=signs)

    if 'map optcom betas' in required_metrics:
        LGR.info('Calculating parameter estimate maps for optimally combined data')
        metric_maps['map optcom betas'] = dependence.calculate_betas(data_optcom, mixing)

    if 'map percent signal change' in required_metrics:
        LGR.info('Calculating percent signal change maps')
        # used in kundu v3.2 tree
        metric_maps['map percent signal change'] = dependence.calculate_psc(
            data_optcom,
            metric_maps['map optcom betas'])

    if 'map Z' in required_metrics:
        LGR.info('Calculating z-statistic maps')
        metric_maps['map Z'] = dependence.calculate_z_maps(metric_maps['map weight'])

    if ('map FT2' in required_metrics) or ('map FS0' in required_metrics):
        LGR.info('Calculating F-statistic maps')
        metric_maps['map FT2'], metric_maps['map FS0'] = dependence.calculate_f_maps(
            data_cat, metric_maps['map Z'], mixing, adaptive_mask, tes)

    if 'map Z clusterized' in required_metrics:
        LGR.info('Thresholding z-statistic maps')
        z_thresh = 1.95
        metric_maps['map Z clusterized'] = dependence.threshold_map(
            metric_maps['map Z'], mask, ref_img, z_thresh)

    if 'map FT2 clusterized' in required_metrics:
        LGR.info('Calculating T2* F-statistic maps')
        f_thresh, _, _ = getfbounds(len(tes))
        metric_maps['map FT2 clusterized'] = dependence.threshold_map(
            metric_maps['map FT2'], mask, ref_img, f_thresh)

    if 'map FS0 clusterized' in required_metrics:
        LGR.info('Calculating S0 F-statistic maps')
        f_thresh, _, _ = getfbounds(len(tes))
        metric_maps['map FS0 clusterized'] = dependence.threshold_map(
            metric_maps['map FS0'], mask, ref_img, f_thresh)

    # Intermediate metrics
    if 'countsigFT2' in required_metrics:
        LGR.info('Counting significant voxels in T2* F-statistic maps')
        comptable['countsigFT2'] = dependence.compute_countsignal(
            metric_maps['map FT2 clusterized'])

    if 'countsigFS0' in required_metrics:
        LGR.info('Counting significant voxels in S0 F-statistic maps')
        comptable['countsigFS0'] = dependence.compute_countsignal(
            metric_maps['map FS0 clusterized'])

    # Back to maps
    if 'map beta T2 clusterized' in required_metrics:
        LGR.info('Thresholding optimal combination beta maps to match T2* F-statistic maps')
        metric_maps['map beta T2 clusterized'] = dependence.threshold_to_match(
            metric_maps['map optcom betas'],
            comptable['countsigFT2'],
            mask, ref_img)

    if 'map beta S0 clusterized' in required_metrics:
        LGR.info('Thresholding optimal combination beta maps to match S0 F-statistic maps')
        metric_maps['map beta S0 clusterized'] = dependence.threshold_to_match(
            metric_maps['map optcom betas'],
            comptable['countsigFS0'],
            mask, ref_img)

    # Dependence metrics
    if ('kappa' in required_metrics) or ('rho' in required_metrics):
        LGR.info('Calculating kappa and rho')
        comptable['kappa'], comptable['rho'] = dependence.calculate_dependence_metrics(
            F_T2_maps=metric_maps['map FT2'],
            F_S0_maps=metric_maps['map FS0'],
            Z_maps=metric_maps['map Z'])

    # Generic metrics
    if 'variance explained' in required_metrics:
        LGR.info('Calculating variance explained')
        comptable['variance explained'] = dependence.calculate_varex(
            metric_maps['map optcom betas'])

    if 'normalized variance explained' in required_metrics:
        LGR.info('Calculating normalized variance explained')
        comptable['normalized variance explained'] = dependence.calculate_varex_norm(
            metric_maps['map weight'])

    # Spatial metrics
    if 'dice_FT2' in required_metrics:
        LGR.info('Calculating DSI between thresholded T2* F-statistic and '
                 'optimal combination beta maps')
        comptable['dice_FT2'] = dependence.compute_dice(
            metric_maps['map beta T2 clusterized'],
            metric_maps['map FT2 clusterized'], axis=0)

    if 'dice_FS0' in required_metrics:
        LGR.info('Calculating DSI between thresholded S0 F-statistic and '
                 'optimal combination beta maps')
        comptable['dice_FS0'] = dependence.compute_dice(
            metric_maps['map beta S0 clusterized'],
            metric_maps['map FS0 clusterized'], axis=0)

    if 'signal-noise_t' in required_metrics:
        LGR.info('Calculating signal-noise t-statistics')
        RepLGR.info('A t-test was performed between the distributions of T2*-model '
                    'F-statistics associated with clusters (i.e., signal) and '
                    'non-cluster voxels (i.e., noise) to generate a t-statistic '
                    '(metric signal-noise_z) and p-value (metric signal-noise_p) '
                    'measuring relative association of the component to signal '
                    'over noise.')
        (comptable['signal-noise_t'],
         comptable['signal-noise_p']) = dependence.compute_signal_minus_noise_t(
            Z_maps=metric_maps['map Z'],
            Z_clmaps=metric_maps['map Z clusterized'],
            F_T2_maps=metric_maps['map FT2'])

    if 'signal-noise_z' in required_metrics:
        LGR.info('Calculating signal-noise z-statistics')
        RepLGR.info('A t-test was performed between the distributions of T2*-model '
                    'F-statistics associated with clusters (i.e., signal) and '
                    'non-cluster voxels (i.e., noise) to generate a z-statistic '
                    '(metric signal-noise_z) and p-value (metric signal-noise_p) '
                    'measuring relative association of the component to signal '
                    'over noise.')
        (comptable['signal-noise_z'],
         comptable['signal-noise_p']) = dependence.compute_signal_minus_noise_z(
            Z_maps=metric_maps['map Z'],
            Z_clmaps=metric_maps['map Z clusterized'],
            F_T2_maps=metric_maps['map FT2'])

    if 'countnoise' in required_metrics:
        LGR.info('Counting significant noise voxels from z-statistic maps')
        RepLGR.info('The number of significant voxels not from clusters was '
                    'calculated for each component.')
        comptable['countnoise'] = dependence.compute_countnoise(
            metric_maps['map Z'],
            metric_maps['map Z clusterized'])

    # Composite metrics
    if 'd_table_score' in required_metrics:
        LGR.info('Calculating decision table score')
        comptable['d_table_score'] = dependence.generate_decision_table_score(
            comptable['kappa'],
            comptable['dice_FT2'],
            comptable['signal-noise_t'],
            comptable['countnoise'],
            comptable['countsigFT2'])

    # Sort the component table and mixing matrix
    comptable, sort_idx = sort_df(comptable, by=sort_by, ascending=ascending)
    mixing, = apply_sort(mixing, sort_idx=sort_idx, axis=1)
    return comptable, mixing
