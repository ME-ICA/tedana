import logging

import sys
import os.path as op
import argparse
import scipy.stats as sct
import numpy as np
import pandas as pd
from tedana.workflows.parser_utils import is_valid_file

# import sys

LGR = logging.getLogger(__name__)


def _get_parser():
    """
    Parses command line inputs for this function

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser()
    # Argument parser follow template provided by RalphyZ, also used by tedana.
    # https://stackoverflow.com/a/43456577
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-td', '--ted-dir',
                          dest='tedana_dir',
                          type=str,
                          help=('Output directory of tedana, i.e. where '
                                'the mixing matrix and the component'
                                'table are located.'),
                          required=True)
    required.add_argument('-dmf', '--dmat_dir',
                          dest='dmat_dir',
                          type=str,
                          help=('Directory containing the design matrices.'),
                          required=True)
    optional.add_argument('-gts', '--good_ts',
                          dest='good_ts',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('File containing timeseries that should'
                                'be accepted. It can be a 1D array or a'
                                'matrix, where columns are the timeseries.'
                                'Specify with path.'),
                          default=None)
    optional.add_argument('-bts', '--bad_ts',
                          dest='bad_ts',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('File containing timeseries that should'
                                'be rejected. It can be a 1D array or a'
                                'matrix, where columns are the timeseries'
                                'Specify with path.'),
                          default=None)
    optional.add_argument('-t', '--thr',
                          dest='thr',
                          type=float,
                          help='Threshold to apply for component selection',
                          default=0.6)
    parser._action_groups.append(optional)
    return parser


def import_file(file_name):
    """
    Importing files and normalising them for Normalised Cross-Correlation
    """
    timeseries = np.genfromtxt(file_name)
    norm_ts = sct.zscore(timeseries, axis=0)
    return norm_ts


def check_dimensionality(ts, mixmat):
    """
    Just checking ts is rightly oriented
    """
    ntr, _ = mixmat.shape
    ts_shape = ts.shape
    # Add a general check on matching at least one dimension
    if ts_shape[0] != ntr:
        if np.size(ts.shape) == 1 or ts_shape[1] != ntr:
            LGR.info('Input matrix has wrong dimensionality.')
            sys.exit()
        else:
            LGR.info('Wrong orientation, transposing the design matrix.')
            ts = ts.T

    return ts


def correlate_ts(ts, mixmat, thr=0.6):
    """
    Run (normalised) cross correlation of the signals
    And then selects maximum NCC to decide which one to flag.
    Also threshold components and then save their position
    """
    _, ncomp = mixmat.shape

    if np.size(ts.shape) != 1:
        _, nts = ts.shape
    else:
        nts = 1

    corr_mtx = np.empty((ncomp, nts))

    for ts_num in range(0, nts):
        for comp_num in range(0, ncomp):
            ncc = np.correlate(mixmat[:, comp_num], ts[:, ts_num], mode='same')
            corr_mtx[comp_num, ts_num] = np.max(ncc)

    thr_mtx = corr_mtx > thr
    mix_mask = thr_mtx.any(axis=1)
    selcomp = [i for i, x in enumerate(mix_mask) if x]

    return selcomp


def modify_comp_table(ctab_fullpath, selcomp, flag='accepted'):
    """
    This function requires as input the component table,
    the index of the selected components and a flag that will overwrite the one in the table.

    """

    comptable = pd.read_csv(ctab_fullpath, sep='\t', index_col='component')

    comptable['original_classification'] = comptable['classification']
    comptable['original_rationale'] = comptable['rationale']
    LGR.info('Overriding original classifications with correlation with'
             'input timeseries (' + flag + ' components)')
    comptable.loc[selcomp, 'classification'] = flag
    if flag == 'accepted':
        comptable.loc[selcomp, 'rationale'] = 'I099'
    else:
        comptable.loc[selcomp, 'rationale'] = 'I098'

    LGR.info('Overwriting original component table.')
    comptable.to_csv(ctab_fullpath, sep='\t', index_label='component')


def check_task_corr(mixmat, ctab_fullpath, ts, thr, flag='accepted'):
    """
    This function is the workflow of the component selection
    """
    norm_ts = import_file(ts)
    checked_ts = check_dimensionality(norm_ts, mixmat)
    selcomp = correlate_ts(checked_ts, mixmat, thr)
    modify_comp_table(ctab_fullpath, selcomp, flag)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    ctab_fullpath = op.join(options.tedana_dir, 'comp_table_ica.txt')
    mixm_fullpath = op.join(options.tedana_dir, '__meica_mix.1D')

    mixmat = import_file(mixm_fullpath)

    if options.bad_ts is not None:
        check_task_corr(mixmat, ctab_fullpath, options.bad_ts, options.thr, flag='rejected')

    if options.good_ts is not None:
        check_task_corr(mixmat, ctab_fullpath, options.good_ts, options.thr, flag='accepted')


if __name__ == '__main__':
    _main()
