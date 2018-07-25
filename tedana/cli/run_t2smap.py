"""
Call tedana from the command line.
"""
import os.path as op

import argparse

from tedana import workflows

import logging
logging.basicConfig(format='[%(levelname)s]: ++ %(message)s',
                    level=logging.INFO)


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
    parser.add_argument('--fitmode',
                        dest='fitmode',
                        action='store',
                        choices=['all', 'ts'],
                        help=('Monoexponential model fitting scheme. '
                              '"all" means that the model is fit, per voxel, '
                              'across all timepoints. '
                              '"ts" means that the model is fit, per voxel '
                              'and per timepoint.'),
                        default='all')
    parser.add_argument('--combmode',
                        dest='combmode',
                        action='store',
                        choices=['t2s', 'ste'],
                        help=('Combination scheme for TEs: '
                              't2s (Posse 1999, default), ste (Poser)'),
                        default='t2s')
    parser.add_argument('--label',
                        dest='label',
                        type=str,
                        help='Label for output directory.',
                        default=None)
    return parser


def main(argv=None):
    """T2smap entry point"""
    options = get_parser().parse_args(argv)
    if options.debug and not options.quiet:
        logging.getLogger().setLevel(logging.DEBUG)
    elif options.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    workflows.t2smap(**vars(options))


if __name__ == '__main__':
    main()
