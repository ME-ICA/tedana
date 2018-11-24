"""
Estimate T2 and S0, and optimally combine data across TEs.
"""
import os
import os.path as op
import logging

import argparse
import numpy as np
from scipy import stats

from tedana import (combine, decay, io, utils)
from tedana.workflows.parser_utils import is_valid_file

LGR = logging.getLogger(__name__)


def _get_parser():
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
    parser.add_argument('--mask',
                        dest='mask',
                        metavar='FILE',
                        type=lambda x: is_valid_file(parser, x),
                        help=('Binary mask of voxels to include in TE '
                              'Dependent ANAlysis. Must be in the same '
                              'space as `data`.'),
                        default=None)
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


def t2smap_workflow(data, tes, mask=None, fitmode='all', combmode='t2s',
                    label=None, debug=False, quiet=False):
    """
    Estimate T2 and S0, and optimally combine data across TEs.

    Parameters
    ----------
    data : :obj:`str` or :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list or str) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.
    mask : :obj:`str`, optional
        Binary mask of voxels to include in TE Dependent ANAlysis. Must be spatially
        aligned with `data`.
    fitmode : {'all', 'ts'}, optional
        Monoexponential model fitting scheme.
        'all' means that the model is fit, per voxel, across all timepoints.
        'ts' means that the model is fit, per voxel and per timepoint.
        Default is 'all'.
    combmode : {'t2s', 'ste'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'ste' (Poser).
    label : :obj:`str` or :obj:`None`, optional
        Label for output directory. Default is None.

    Other Parameters
    ----------------
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppresses logging/printing of messages. Default is False.


    Notes
    -----
    This workflow writes out several files, which are written out to a folder
    named TED.[ref_label].[label] if ``label`` is provided and TED.[ref_label]
    if not. ``ref_label`` is determined based on the name of the first ``data``
    file.

    Files are listed below:

    ======================    =================================================
    Filename                  Content
    ======================    =================================================
    t2sv.nii                  Limited estimated T2* 3D map or 4D timeseries.
                              Will be a 3D map if ``fitmode`` is 'all' and a
                              4D timeseries if it is 'ts'.
    s0v.nii                   Limited S0 3D map or 4D timeseries.
    t2svG.nii                 Full T2* map/timeseries. The difference between
                              the limited and full maps is that, for voxels
                              affected by dropout where only one echo contains
                              good data, the full map uses the single echo's
                              value while the limited map has a NaN.
    s0vG.nii                  Full S0 map/timeseries.
    ts_OC.nii                 Optimally combined timeseries.
    ======================    =================================================
    """
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

    try:
        ref_label = os.path.basename(ref_img).split('.')[0]
    except (TypeError, AttributeError):
        ref_label = os.path.basename(str(data[0])).split('.')[0]

    if label is not None:
        out_dir = 'TED.{0}.{1}'.format(ref_label, label)
    else:
        out_dir = 'TED.{0}'.format(ref_label)
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        LGR.info('Creating output directory: {}'.format(out_dir))
        os.mkdir(out_dir)
    else:
        LGR.info('Using output directory: {}'.format(out_dir))

    if mask is None:
        LGR.info('Computing adaptive mask')
    else:
        LGR.info('Using user-defined mask')
    mask, masksum = utils.make_adaptive_mask(catd, minimum=False, getsum=True)

    LGR.info('Computing adaptive T2* map')
    if fitmode == 'all':
        (t2s_limited, s0_limited,
         t2ss, s0s,
         t2s_full, s0_full) = decay.fit_decay(catd, tes, mask, masksum)
    else:
        (t2s_limited, s0_limited,
         t2s_full, s0_full) = decay.fit_decay_ts(catd, tes, mask, masksum)

    # set a hard cap for the T2* map/timeseries
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s_limited.flatten(), 99.5,
                                      interpolation_method='lower')
    LGR.debug('Setting cap on T2* map at {:.5f}'.format(cap_t2s * 10))
    t2s_limited[t2s_limited > cap_t2s * 10] = cap_t2s

    LGR.info('Computing optimal combination')
    # optimally combine data
    OCcatd = combine.make_optcom(catd, tes, mask, t2s=t2s_full,
                                 combmode=combmode)

    # clean up numerical errors
    for arr in (OCcatd, s0_limited, t2s_limited):
        np.nan_to_num(arr, copy=False)

    s0_limited[s0_limited < 0] = 0
    t2s_limited[t2s_limited < 0] = 0

    io.filewrite(t2s_limited, op.join(out_dir, 't2sv.nii'), ref_img)
    io.filewrite(s0_limited, op.join(out_dir, 's0v.nii'), ref_img)
    io.filewrite(t2s_full, op.join(out_dir, 't2svG.nii'), ref_img)
    io.filewrite(s0_full, op.join(out_dir, 's0vG.nii'), ref_img)
    io.filewrite(OCcatd, op.join(out_dir, 'ts_OC.nii'), ref_img)


def _main(argv=None):
    """T2smap entry point"""
    options = _get_parser().parse_args(argv)
    if options.debug and not options.quiet:
        logging.basicConfig(level=logging.DEBUG)
    elif options.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    t2smap_workflow(**vars(options))


if __name__ == '__main__':
    _main()
