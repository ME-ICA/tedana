"""
Run the "canonical" TE-Dependent ANAlysis workflow.
"""
import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import os.path as op
import shutil
import logging
from datetime import datetime

import argparse
import numpy as np
import pandas as pd
from scipy import stats

from tedana.workflows.parser_utils import is_valid_file
from tedana import decay, combine, decomposition, io, model, selection, utils

LGR = logging.getLogger(__name__)


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
    optional.add_argument('--mask',
                          dest='mask',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('Binary mask of voxels to include in TE '
                                'Dependent ANAlysis. Must be in the same '
                                'space as `data`.'),
                          default=None)
    optional.add_argument('--mix',
                          dest='mixm',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('File containing mixing matrix. If not '
                                'provided, ME-PCA & ME-ICA is done.'),
                          default=None)
    optional.add_argument('--ctab',
                          dest='ctab',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('File containing a component table from which '
                                'to extract pre-computed classifications.'),
                          default=None)
    optional.add_argument('--manacc',
                          dest='manacc',
                          help=('Comma separated list of manually '
                                'accepted components'),
                          default=None)
    optional.add_argument('--sourceTEs',
                          dest='ste',
                          type=str,
                          help=('Source TEs for models. E.g., 0 for all, '
                                '-1 for opt. com., and 1,2 for just TEs 1 and '
                                '2. Default=-1.'),
                          default=-1)
    optional.add_argument('--combmode',
                          dest='combmode',
                          action='store',
                          choices=['t2s', 'ste'],
                          help=('Combination scheme for TEs: '
                                't2s (Posse 1999, default), ste (Poser)'),
                          default='t2s')
    optional.add_argument('--verbose',
                          dest='verbose',
                          action='store_true',
                          help='Generate intermediate and additional files.',
                          default=False)
    optional.add_argument('--tedort',
                          dest='tedort',
                          action='store_true',
                          help=('Orthogonalize rejected components w.r.t. '
                                'accepted components prior to denoising.'),
                          default=False)
    optional.add_argument('--gscontrol',
                          dest='gscontrol',
                          required=False,
                          action='store',
                          nargs='+',
                          help=('Perform additional denoising to remove '
                                'spatially diffuse noise. Default is None. '
                                'This argument can be single value or a space '
                                'delimited list'),
                          choices=['t1c', 'gsr'],
                          default=None)
    optional.add_argument('--wvpca',
                          dest='wvpca',
                          help='Perform PCA on wavelet-transformed data',
                          action='store_true',
                          default=False)
    optional.add_argument('--tedpca',
                          dest='tedpca',
                          help='Method with which to select components in TEDPCA',
                          choices=['mle', 'kundu', 'kundu-stabilize'],
                          default='mle')
    optional.add_argument('--out-dir',
                          dest='out_dir',
                          type=str,
                          help='Output directory.',
                          default='.')
    optional.add_argument('--seed',
                          dest='fixed_seed',
                          type=int,
                          help=('Value passed to repr(mdp.numx_rand.seed()) '
                                'Set to an integer value for reproducible ICA results; '
                                'otherwise, set to -1 for varying results across calls.'),
                          default=42)
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


def tedana_workflow(data, tes, mask=None, mixm=None, ctab=None, manacc=None,
                    tedort=False, gscontrol=None, tedpca='mle',
                    ste=-1, combmode='t2s', verbose=False, stabilize=False,
                    wvpca=False, out_dir='.', fixed_seed=42, debug=False,
                    quiet=False):
    """
    Run the "canonical" TE-Dependent ANAlysis workflow.

    Parameters
    ----------
    data : :obj:`str` or :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list or str) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.
    mask : :obj:`str`, optional
        Binary mask of voxels to include in TE Dependent ANAlysis. Must be
        spatially aligned with `data`.
    mixm : :obj:`str`, optional
        File containing mixing matrix. If not provided, ME-PCA and ME-ICA are
        done.
    ctab : :obj:`str`, optional
        File containing component table from which to extract pre-computed
        classifications.
    manacc : :obj:`str`, optional
        Comma separated list of manually accepted components in string form.
        Default is None.
    tedort : :obj:`bool`, optional
        Orthogonalize rejected components w.r.t. accepted ones prior to
        denoising. Default is False.
    gscontrol : {None, 't1c', 'gsr'} or :obj:`list`, optional
        Perform additional denoising to remove spatially diffuse noise. Default
        is None.
    tedpca : {'mle', 'kundu', 'kundu-stabilize'}, optional
        Method with which to select components in TEDPCA. Default is 'mle'.
    ste : :obj:`int`, optional
        Source TEs for models. 0 for all, -1 for optimal combination.
        Default is -1.
    combmode : {'t2s', 'ste'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'ste' (Poser).
    verbose : :obj:`bool`, optional
        Generate intermediate and additional files. Default is False.
    wvpca : :obj:`bool`, optional
        Whether or not to perform PCA on wavelet-transformed data.
        Default is False.
    out_dir : :obj:`str`, optional
        Output directory.

    Other Parameters
    ----------------
    fixed_seed : :obj:`int`, optional
        Value passed to ``mdp.numx_rand.seed()``.
        Set to a positive integer value for reproducible ICA results;
        otherwise, set to -1 for varying results across calls.
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppresses logging/printing of messages. Default is False.

    Notes
    -----
    This workflow writes out several files. For a complete list of the files
    generated by this workflow, please visit
    https://tedana.readthedocs.io/en/latest/outputs.html
    """
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    if debug and not quiet:
        formatter = logging.Formatter(
                    '%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S')
        fh = logging.FileHandler(op.join(
            out_dir,
            'runlog-{0}.tsv'.format(datetime.now().isoformat().replace(':', '.'))))
        fh.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[fh, logging.StreamHandler()])
    elif quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    LGR.info('Using output directory: {}'.format(out_dir))

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # Coerce gscontrol to list
    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]

    # coerce data to samples x echos x time array
    if isinstance(data, str):
        data = [data]

    LGR.info('Loading input data: {}'.format([f for f in data]))
    catd, ref_img = io.load_data(data, n_echos=n_echos)
    n_samp, n_echos, n_vols = catd.shape
    LGR.debug('Resulting data shape: {}'.format(catd.shape))

    if mixm is not None and op.isfile(mixm):
        shutil.copyfile(mixm, op.join(out_dir, 'meica_mix.1D'))
        shutil.copyfile(mixm, op.join(out_dir, op.basename(mixm)))
    elif mixm is not None:
        raise IOError('Argument "mixm" must be an existing file.')

    if ctab is not None and op.isfile(ctab):
        shutil.copyfile(ctab, op.join(out_dir, 'comp_table_ica.txt'))
        shutil.copyfile(ctab, op.join(out_dir, op.basename(ctab)))
    elif ctab is not None:
        raise IOError('Argument "ctab" must be an existing file.')

    if mask is None:
        LGR.info('Computing adaptive mask')
    else:
        # TODO: add affine check
        LGR.info('Using user-defined mask')

    mask, masksum = utils.make_adaptive_mask(catd, mask=mask,
                                             minimum=False, getsum=True)
    LGR.debug('Retaining {}/{} samples'.format(mask.sum(), n_samp))
    if verbose:
        io.filewrite(masksum, op.join(out_dir, 'adaptive_mask.nii'), ref_img)

    os.chdir(out_dir)

    LGR.info('Computing T2* map')
    t2s, s0, t2ss, s0s, t2sG, s0G = decay.fit_decay(catd, tes, mask, masksum)

    # set a hard cap for the T2* map
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s.flatten(), 99.5,
                                      interpolation_method='lower')
    LGR.debug('Setting cap on T2* map at {:.5f}'.format(cap_t2s * 10))
    t2s[t2s > cap_t2s * 10] = cap_t2s
    io.filewrite(t2s, op.join(out_dir, 't2sv.nii'), ref_img)
    io.filewrite(s0, op.join(out_dir, 's0v.nii'), ref_img)

    if verbose:
        io.filewrite(t2ss, op.join(out_dir, 't2ss.nii'), ref_img)
        io.filewrite(s0s, op.join(out_dir, 's0vs.nii'), ref_img)
        io.filewrite(t2sG, op.join(out_dir, 't2svG.nii'), ref_img)
        io.filewrite(s0G, op.join(out_dir, 's0vG.nii'), ref_img)

    # optimally combine data
    data_oc = combine.make_optcom(catd, tes, mask, t2s=t2sG, combmode=combmode)

    # regress out global signal unless explicitly not desired
    if 'gsr' in gscontrol:
        catd, data_oc = model.gscontrol_raw(catd, data_oc, n_echos, ref_img)

    if mixm is None:
        # Identify and remove thermal noise from data
        n_components, dd = decomposition.tedpca(catd, data_oc, combmode, mask,
                                                t2s, t2sG, ref_img,
                                                tes=tes, method=tedpca, ste=ste,
                                                kdaw=10., rdaw=1., wvpca=wvpca,
                                                verbose=verbose)
        mmix_orig, fixed_seed = decomposition.tedica(n_components, dd,
                                                     fixed_seed)

        if verbose:
            np.savetxt(op.join(out_dir, '__meica_mix.1D'), mmix_orig)
            if ste == -1:
                io.filewrite(utils.unmask(dd, mask),
                             op.join(out_dir, 'ts_OC_whitened.nii'), ref_img)

        LGR.info('Making second component selection guess from ICA results')
        # Estimate betas and compute selection metrics for mixing matrix
        # generated from dimensionally reduced data using full data (i.e., data
        # with thermal noise)
        seldict, comptable, betas, mmix = model.fitmodels_direct(
                    catd, mmix_orig, mask, t2s, t2sG, tes, combmode,
                    ref_img, reindex=True, label='meica_', out_dir=out_dir,
                    verbose=verbose)
        np.savetxt(op.join(out_dir, 'meica_mix.1D'), mmix)

        comptable = selection.selcomps(seldict, comptable, mmix, manacc,
                                       n_echos)
    else:
        LGR.info('Using supplied mixing matrix from ICA')
        mmix_orig = np.loadtxt(op.join(out_dir, 'meica_mix.1D'))
        seldict, comptable, betas, mmix = model.fitmodels_direct(
                    catd, mmix_orig, mask, t2s, t2sG, tes, combmode,
                    ref_img, label='meica_', out_dir=out_dir,
                    verbose=verbose)
        if ctab is None:
            comptable = selection.selcomps(seldict, comptable, mmix, manacc,
                                           n_echos)
        else:
            comptable = pd.read_csv(ctab, sep='\t', index_col='component')

    comptable.to_csv(op.join(out_dir, 'comp_table_ica.txt'), sep='\t',
                     index=True, index_label='component', float_format='%.6f')
    if 'component' not in comptable.columns:
        comptable['component'] = comptable.index
    acc = comptable.loc[comptable['classification'] == 'accepted', 'component']
    rej = comptable.loc[comptable['classification'] == 'rejected', 'component']
    midk = comptable.loc[comptable['classification'] == 'midk', 'component']
    ign = comptable.loc[comptable['classification'] == 'ignored', 'component']
    if len(acc) == 0:
        LGR.warning('No BOLD components detected! Please check data and '
                    'results!')

    if tedort:
        acc_idx = comptable.loc[
            ~comptable['classification'].str.contains('rejected'),
            'component']
        rej_idx = comptable.loc[
            comptable['classification'].str.contains('rejected'),
            'component']
        acc_ts = mmix[:, acc_idx]
        rej_ts = mmix[:, rej_idx]
        betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
        pred_rej_ts = np.dot(acc_ts, betas)
        resid = rej_ts - pred_rej_ts
        mmix[:, rej_idx] = resid
        np.savetxt(op.join(out_dir, 'meica_mix_orth.1D'), mmix)

    io.writeresults(data_oc, mask=mask, comptable=comptable, mmix=mmix,
                    n_vols=n_vols, fixed_seed=fixed_seed,
                    acc=acc, rej=rej, midk=midk, empty=ign,
                    ref_img=ref_img)

    if 't1c' in gscontrol:
        LGR.info('Performing T1c global signal regression to remove spatially '
                 'diffuse noise')
        io.gscontrol_mmix(data_oc, mmix, mask, comptable, ref_img)

    if verbose:
        io.writeresults_echoes(catd, mmix, mask, acc, rej, midk, ref_img)

    LGR.info('Workflow completed')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def _main(argv=None):
    """Tedana entry point"""
    options = _get_parser().parse_args(argv)
    tedana_workflow(**vars(options))


if __name__ == '__main__':
    _main()
