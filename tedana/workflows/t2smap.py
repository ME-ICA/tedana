"""
Estimate T2 and S0, and optimally combine data across TEs.
"""
import os
import os.path as op
import logging

import argparse
import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from tedana import (combine, decay, io, utils)
from tedana.workflows.parser_utils import is_valid_file

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def _get_parser():
    """
    Parses command line inputs for tedana

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser()
    # Argument parser follow templtate provided by RalphyZ
    # https://stackoverflow.com/a/43456577
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-d',
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
    required.add_argument('-e',
                          dest='tes',
                          nargs='+',
                          metavar='TE',
                          type=float,
                          help='Echo times (in ms). E.g., 15.0 39.0 63.0',
                          required=True)
    optional.add_argument('--out-dir',
                          dest='out_dir',
                          type=str,
                          metavar='PATH',
                          help='Output directory.',
                          default='.')
    optional.add_argument('--mask',
                          dest='mask',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('Binary mask of voxels to include in TE '
                                'Dependent ANAlysis. Must be in the same '
                                'space as `data`.'),
                          default=None)
    optional.add_argument('--fittype',
                          dest='fittype',
                          action='store',
                          choices=['loglin', 'curvefit'],
                          help='Desired Fitting Method'
                               '"loglin" means that a linear model is fit'
                               ' to the log of the data, default'
                               '"curvefit" means that a more computationally'
                               'demanding monoexponential model is fit'
                               'to the raw data',
                          default='loglin')
    optional.add_argument('--fitmode',
                          dest='fitmode',
                          action='store',
                          choices=['all', 'ts'],
                          help=('Monoexponential model fitting scheme. '
                                '"all" means that the model is fit, per voxel, '
                                'across all timepoints. '
                                '"ts" means that the model is fit, per voxel '
                                'and per timepoint.'),
                          default='all')
    optional.add_argument('--combmode',
                          dest='combmode',
                          action='store',
                          choices=['t2s', 'paid'],
                          help=('Combination scheme for TEs: '
                                't2s (Posse 1999, default), paid (Poser)'),
                          default='t2s')
    optional.add_argument('--n-threads',
                          dest='n_threads',
                          type=int,
                          action='store',
                          help=('Number of threads to use. Used by '
                                'threadpoolctl to set the parameter outside '
                                'of the workflow function. Higher numbers of '
                                'threads tend to slow down performance on '
                                'typical datasets. Default is 1.'),
                          default=1)
    optional.add_argument('--debug',
                          dest='debug',
                          help=argparse.SUPPRESS,
                          action='store_true',
                          default=False)
    optional.add_argument('--quiet',
                          dest='quiet',
                          help=argparse.SUPPRESS,
                          action='store_true',
                          default=False)
    parser._action_groups.append(optional)
    return parser


def t2smap_workflow(data, tes, out_dir='.', mask=None,
                    fittype='loglin', fitmode='all', combmode='t2s',
                    debug=False, quiet=False):
    """
    Estimate T2 and S0, and optimally combine data across TEs.

    Parameters
    ----------
    data : :obj:`str` or :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list or str) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.
    out_dir : :obj:`str`, optional
        Output directory.
    mask : :obj:`str`, optional
        Binary mask of voxels to include in TE Dependent ANAlysis. Must be spatially
        aligned with `data`.
    fittype : {'loglin', 'curvefit'}, optional
        Monoexponential fitting method.
        'loglin' means to use the the default linear fit to the log of
        the data.
        'curvefit' means to use a monoexponential fit to the raw data,
        which is slightly slower but may be more accurate.
    fitmode : {'all', 'ts'}, optional
        Monoexponential model fitting scheme.
        'all' means that the model is fit, per voxel, across all timepoints.
        'ts' means that the model is fit, per voxel and per timepoint.
        Default is 'all'.
    combmode : {'t2s', 'paid'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'paid' (Poser).

    Other Parameters
    ----------------
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppress logging/printing of messages. Default is False.


    Notes
    -----
    This workflow writes out several files, which are described below:

    ==========================    =================================================
    Filename                      Content
    ==========================    =================================================
    T2starmap.nii.gz              Limited estimated T2* 3D map or 4D timeseries.
                                  Will be a 3D map if ``fitmode`` is 'all' and a
                                  4D timeseries if it is 'ts'.
    S0map.nii.gz                  Limited S0 3D map or 4D timeseries.
    desc-full_T2starmap.nii.gz    Full T2* map/timeseries. The difference between
                                  the limited and full maps is that, for voxels
                                  affected by dropout where only one echo contains
                                  good data, the full map uses the single echo's
                                  value while the limited map has a NaN.
    desc-full_S0map.nii.gz        Full S0 map/timeseries.
    desc-optcom_bold.nii.gz       Optimally combined timeseries.
    ==========================    =================================================
    """
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    if debug and not quiet:
        logging.basicConfig(level=logging.DEBUG)
    elif quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    LGR.info('Using output directory: {}'.format(out_dir))

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    if isinstance(data, str):
        data = [data]

    LGR.info('Loading input data: {}'.format([f for f in data]))
    catd, ref_img = io.load_data(data, n_echos=n_echos)
    n_samp, n_echos, n_vols = catd.shape
    LGR.debug('Resulting data shape: {}'.format(catd.shape))

    if mask is None:
        LGR.info('Computing adaptive mask')
    else:
        LGR.info('Using user-defined mask')
    mask, masksum = utils.make_adaptive_mask(catd, mask=mask, getsum=True)

    LGR.info('Computing adaptive T2* map')
    if fitmode == 'all':
        (t2s_limited, s0_limited,
         t2s_full, s0_full) = decay.fit_decay(catd, tes, mask, masksum,
                                              fittype)
    else:
        (t2s_limited, s0_limited,
         t2s_full, s0_full) = decay.fit_decay_ts(catd, tes, mask, masksum,
                                                 fittype)

    # set a hard cap for the T2* map/timeseries
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s_limited.flatten(), 99.5,
                                      interpolation_method='lower')
    cap_t2s_sec = utils.millisec2sec(cap_t2s * 10.)
    LGR.debug('Setting cap on T2* map at {:.5f}s'.format(cap_t2s_sec))
    t2s_limited[t2s_limited > cap_t2s * 10] = cap_t2s

    LGR.info('Computing optimal combination')
    # optimally combine data
    OCcatd = combine.make_optcom(catd, tes, masksum, t2s=t2s_full,
                                 combmode=combmode)

    # clean up numerical errors
    for arr in (OCcatd, s0_limited, t2s_limited):
        np.nan_to_num(arr, copy=False)

    s0_limited[s0_limited < 0] = 0
    t2s_limited[t2s_limited < 0] = 0

    io.filewrite(utils.millisec2sec(t2s_limited),
                 op.join(out_dir, 'T2starmap.nii.gz'), ref_img)
    io.filewrite(s0_limited, op.join(out_dir, 'S0map.nii.gz'), ref_img)
    io.filewrite(utils.millisec2sec(t2s_full),
                 op.join(out_dir, 'desc-full_T2starmap.nii.gz'), ref_img)
    io.filewrite(s0_full, op.join(out_dir, 'desc-full_S0map.nii.gz'), ref_img)
    io.filewrite(OCcatd, op.join(out_dir, 'desc-optcom_bold.nii.gz'), ref_img)


def _main(argv=None):
    """T2smap entry point"""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    n_threads = kwargs.pop('n_threads')
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        t2smap_workflow(**kwargs)


if __name__ == '__main__':
    _main()
