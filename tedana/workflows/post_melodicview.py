"""
Quick hack for tedana, to prepare a ready-to-read FSLeyes folder.

"""

import logging

import sys
import argparse

import pandas as pd

import os.path as op

LGR = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    optional.add_argument('-o', '--out-dir',
                          dest='out_dir',
                          type=str,
                          help=('Output folder, i.e. where'
                                'the meica.ica folder should be.'),
                          default=None)
    parser._action_groups.append(optional)
    return parser


def check_dimensionality(ts, mixmat):
    """
    Just checking that ts is rightly oriented

    Parameters
    ----------
    ts: T x N or N x T numpy.array
        Array of unkown dimensionality to be compared with mixmat.
        Its first dimension length should match mixmat.
    mixmat: T x N numpy.array
        Array of known dimensionality, to which ts is compared to.

    Returns
    -------
    ts: T x N numpy.array
        Array of known dimensionality. Its first dimension is the same as mixmat.

    Raise
    -----
    sys.exit if ts and mixmat don't have any dimension of same length.

    """
    ntr, _ = mixmat.shape
    ts_shape = ts.shape
    # Add a general check on matching at least one dimension
    if ts_shape[0] != ntr:
        if np.size(ts.shape) == 1 or ts_shape[1] != ntr:
            LGR.error('Input matrix has wrong dimensionality.')
            sys.exit()
        else:
            LGR.warning('Wrong orientation, transposing the design matrix.')
            ts = ts.T

    return ts


def correlate_ts(ts, mixmat, thr=0.6):
    """
    Run (normalised) cross correlation of the signals
    And then selects maximum NCC to decide which one to flag.
    Also threshold components and then save their index.

    Parameters
    ----------
    ts: T x N numpy.array
        Timeseries to compare components to.
    mixmat: T x N numpy.array
        Timeseries of MEICA components.
    thr: float, Optional
        Threshold for NCC value considered significant. Default = 0.6.

    Returns
    -------
    selcomp: list
        List of indexes of components that are significantly similar to any of the N
        timeseries of ts.

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
    Function to change flag of components in the components table.

    Parameters
    ----------
    ctab_fullpath: string
        Full path to components table as of tedana output.
        !!! It will be overwritten !!!
    selcomp: list
        List of indexes of components to be flagged with flag.
    flag: string, Optional
        Flag to change the label of the components with index
        selcomp, in the components table.

    Note
    ----
    The file output is the same ctab_fullpath

    """

    comptable = pd.read_csv(ctab_fullpath, sep='\t', index_col='component')

    comptable['original_classification'] = comptable['classification']
    comptable['original_rationale'] = comptable['rationale']
    LGR.info('Overriding original classifications with correlation with'
             'input timeseries (%s components)', flag)
    comptable.loc[selcomp, 'classification'] = flag
    if flag == 'accepted':
        comptable.loc[selcomp, 'rationale'] = 'I099'
    else:
        comptable.loc[selcomp, 'rationale'] = 'I098'

    LGR.info('Overwriting original component table.')
    comptable.to_csv(ctab_fullpath, sep='\t', index_label='component')


def check_task_corr(mixmat, ctab_fullpath, ts, thr, flag='accepted'):
    """
    This function is the workflow of the component selection.
    """
    norm_ts = import_file(ts)
    checked_ts = check_dimensionality(norm_ts, mixmat)
    selcomp = correlate_ts(checked_ts, mixmat, thr)
    if selcomp:
        modify_comp_table(ctab_fullpath, selcomp, flag)
    else:
        LGR.info('No component to flag as %s', flag)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    ctab_fullpath = op.join(options.tedana_dir, 'comp_table_ica.txt')
    mixm_fullpath = op.join(options.tedana_dir, 'meica_mix.1D')

    mixmat = import_file(mixm_fullpath)

    if options.bad_ts is not None:
        check_task_corr(mixmat, ctab_fullpath, options.bad_ts, options.thr, flag='rejected')

    if options.good_ts is not None:
        check_task_corr(mixmat, ctab_fullpath, options.good_ts, options.thr, flag='accepted')


if __name__ == '__main__':
    _main()
