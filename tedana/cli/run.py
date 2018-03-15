import argparse
from tedana.interfaces import tedana


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
                        help='Spatially Concatenated Multi-Echo Dataset',
                        required=True)
    parser.add_argument('-e',
                        dest='tes',
                        nargs='+',
                        help='Echo times (in ms) ex: 15,39,63',
                        required=True)
    parser.add_argument('--mix',
                        dest='mixm',
                        help='Mixing matrix. If not provided, ' +
                             'ME-PCA & ME-ICA is done.',
                        default=None)
    parser.add_argument('--ctab',
                        dest='ctab',
                        help='Component table extract pre-computed ' +
                             'classifications from.',
                        default=None)
    parser.add_argument('--manacc',
                        dest='manacc',
                        help='Comma separated list of manually ' +
                             'accepted components',
                        default=None)
    parser.add_argument('--strict',
                        dest='strict',
                        action='store_true',
                        help='Ignore low-variance ambiguous components',
                        default=False)
    parser.add_argument('--no_gscontrol',
                        dest='no_gscontrol',
                        action='store_true',
                        help='Control global signal using spatial approach',
                        default=False)
    parser.add_argument('--kdaw',
                        dest='kdaw',
                        help='Dimensionality augmentation weight ' +
                             '(Kappa). Default 10. -1 for low-dimensional ICA',
                        default=10.)
    parser.add_argument('--rdaw',
                        dest='rdaw',
                        help='Dimensionality augmentation weight (Rho). ' +
                             'Default 1. -1 for low-dimensional ICA',
                        default=1.)
    parser.add_argument('--conv',
                        dest='conv',
                        help='Convergence limit. Default 2.5e-5',
                        default='2.5e-5')
    parser.add_argument('--sourceTEs',
                        dest='ste',
                        help='Source TEs for models. ex: -ste 0 for all, ' +
                             '-1 for opt. com. Default -1.',
                        default=-1)
    parser.add_argument('--combmode',
                        dest='combmode',
                        help='Combination scheme for TEs: t2s ' +
                             '(Posse 1999, default),ste(Poser)',
                        default='t2s')
    parser.add_argument('--denoiseTEs',
                        dest='dne',
                        action='store_true',
                        help='Denoise each TE dataset separately',
                        default=False)
    parser.add_argument('--initcost',
                        dest='initcost',
                        help='Initial cost func. for ICA: pow3, ' +
                             'tanh(default), gaus, skew',
                        default='tanh')
    parser.add_argument('--finalcost',
                        dest='finalcost',
                        help='Final cost func, same opts. as initial',
                        default='tanh')
    parser.add_argument('--stabilize',
                        dest='stabilize',
                        action='store_true',
                        help='Stabilize convergence by reducing ' +
                             'dimensionality, for low quality data',
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
                        help='Label for output directory.',
                        default=None)
    parser.add_argument('--seed',
                        dest='fixed_seed',
                        help='Seeded value for ICA, for reproducibility.',
                        default=42)
    return parser


def main(argv=None):
    """Entry point"""
    options = get_parser().parse_args(argv)
    tedana.main(options)


if __name__ == '__main__':
    main()
