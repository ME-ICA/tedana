"""
Call tedana from the command line.
"""
import os.path as op

import argparse

from tedana import workflows

import logging
logging.basicConfig(format='[%(levelname)s]: ++ %(message)s', level=logging.INFO)


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error('The file {0} does not exist!'.format(arg))

    return arg


def get_parser():
    """
    Parses command line inputs for tedana

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        dest='data',
                        nargs='+',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x),
                        help=('Multi-echo dataset for analysis. May be a '
                              'single file with spatially concatenated data '
                              'or a set of echo-specific files, in the same '
                              'order as the TEs are listed in the -e '
                              'argument.'),
                        required=True)
    parser.add_argument('-e',
                        dest='tes',
                        nargs='+',
                        metavar='TE',
                        type=float,
                        help='Echo times (in ms). E.g., 15.0 39.0 63.0',
                        required=True)
    parser.add_argument('--mix',
                        dest='mixm',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x),
                        help=('File containing mixing matrix. If not '
                              'provided, ME-PCA & ME-ICA is done.'),
                        default=None)
    parser.add_argument('--ctab',
                        dest='ctab',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x),
                        help=('File containing a component table from which '
                              'to extract pre-computed classifications.'),
                        default=None)
    parser.add_argument('--manacc',
                        dest='manacc',
                        help=('Comma separated list of manually '
                              'accepted components'),
                        default=None)
    parser.add_argument('--kdaw',
                        dest='kdaw',
                        type=float,
                        help=('Dimensionality augmentation weight (Kappa). '
                              'Default=10. -1 for low-dimensional ICA'),
                        default=10.)
    parser.add_argument('--rdaw',
                        dest='rdaw',
                        type=float,
                        help=('Dimensionality augmentation weight (Rho). '
                              'Default=1. -1 for low-dimensional ICA'),
                        default=1.)
    parser.add_argument('--conv',
                        dest='conv',
                        type=float,
                        help='Convergence limit. Default 2.5e-5',
                        default='2.5e-5')
    parser.add_argument('--sourceTEs',
                        dest='ste',
                        type=str,
                        help=('Source TEs for models. E.g., 0 for all, '
                              '-1 for opt. com., and 1,2 for just TEs 1 and '
                              '2. Default=-1.'),
                        default=-1)
    parser.add_argument('--combmode',
                        dest='combmode',
                        action='store',
                        choices=['t2s', 'ste'],
                        help=('Combination scheme for TEs: '
                              't2s (Posse 1999, default), ste (Poser)'),
                        default='t2s')
    parser.add_argument('--initcost',
                        dest='initcost',
                        action='store',
                        choices=['tanh', 'pow3', 'gaus', 'skew'],
                        help=('Initial cost function for ICA.'),
                        default='tanh')
    parser.add_argument('--finalcost',
                        dest='finalcost',
                        action='store',
                        choices=['tanh', 'pow3', 'gaus', 'skew'],
                        help=('Final cost function for ICA. Same options as '
                              'initcost.'),
                        default='tanh')
    parser.add_argument('--denoiseTEs',
                        dest='dne',
                        action='store_true',
                        help='Denoise each TE dataset separately.',
                        default=False)
    parser.add_argument('--strict',
                        dest='strict',
                        action='store_true',
                        help='Ignore low-variance ambiguous components',
                        default=False)
    parser.add_argument('--no_gscontrol',
                        dest='gscontrol',
                        action='store_false',
                        help='Disable global signal regression.',
                        default=True)
    parser.add_argument('--stabilize',
                        dest='stabilize',
                        action='store_true',
                        help=('Stabilize convergence by reducing '
                              'dimensionality, for low quality data'),
                        default=False)
    parser.add_argument('--fout',
                        dest='fout',
                        help='Output TE-dependence Kappa/Rho SPMs',
                        action='store_true',
                        default=False)
    parser.add_argument('--filecsdata',
                        dest='filecsdata',
                        help='Save component selection data',
                        action='store_true',
                        default=False)
    parser.add_argument('--label',
                        dest='label',
                        type=str,
                        help='Label for output directory.',
                        default=None)
    parser.add_argument('--seed',
                        dest='fixed_seed',
                        type=int,
                        help='Seeded value for ICA, for reproducibility.',
                        default=42)
    parser.add_argument('--debug',
                        dest='debug',
                        help=argparse.SUPPRESS,
                        action='store_true',
                        default=False)
    parser.add_argument('--quiet',
                        dest='quiet',
                        help=argparse.SUPPRESS,
                        action='store_true',
                        default=False)
    return parser


def main(argv=None):
    """Tedana entry point"""
    options = get_parser().parse_args(argv)
    if options.debug and not options.quiet:
        logging.getLogger().setLevel(logging.DEBUG)
    elif options.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    workflows.tedana(**vars(options))


def run_t2smap(argv=None):
    """T2smap entry point"""
    options = get_parser().parse_args(argv)
    if options.debug and not options.quiet:
        logging.getLogger().setLevel(logging.DEBUG)
    elif options.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    workflows.t2smap(**vars(options))


if __name__ == '__main__':
    main()
