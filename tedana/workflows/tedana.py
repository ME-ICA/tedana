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
import glob
import shutil
import logging

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from nilearn.masking import compute_epi_mask

from tedana import (decay, combine, decomposition, io, metrics, selection, utils,
                    viz)
import tedana.gscontrol as gsc
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
                          help=("Binary mask of voxels to include in TE "
                                "Dependent ANAlysis. Must be in the same "
                                "space as `data`. If an explicit mask is not "
                                "provided, then Nilearn's compute_epi_mask "
                                "function will be used to derive a mask "
                                "from the first echo's data."),
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
                          dest='source_tes',
                          type=str,
                          help=('Source TEs for models. E.g., 0 for all, '
                                '-1 for opt. com., and 1,2 for just TEs 1 and '
                                '2. Default=-1.'),
                          default=-1)
    optional.add_argument('--combmode',
                          dest='combmode',
                          action='store',
                          choices=['t2s'],
                          help=('Combination scheme for TEs: '
                                't2s (Posse 1999, default)'),
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
                          help=('Value passed to repr(mdp.numx_rand.seed()). '
                                'Set to an integer value for reproducible ICA results. '
                                'Set to -1 for varying results across ICA calls. '
                                'Default=42.'),
                          default=42)
    optional.add_argument('--png',
                          dest='png',
                          action='store_true',
                          help=('Creates a figures folder with static component '
                                'maps, timecourse plots and other diagnostic '
                                'images'),
                          default=False)
    optional.add_argument('--png-cmap',
                          dest='png_cmap',
                          type=str,
                          help=('Colormap for figures'),
                          default='coolwarm')
    optional.add_argument('--maxit',
                          dest='maxit',
                          type=int,
                          help=('Maximum number of iterations for ICA.'),
                          default=500)
    optional.add_argument('--maxrestart',
                          dest='maxrestart',
                          type=int,
                          help=('Maximum number of attempts for ICA. If ICA '
                                'fails to converge, the fixed seed will be '
                                'updated and ICA will be run again. If '
                                'convergence is achieved before maxrestart '
                                'attempts, ICA will finish early.'),
                          default=10)
    optional.add_argument('--TR',
                          dest='user_tr',
                          type=float,
                          help=('A TR in seconds that you supply if you '
                                'suspect your header reflects a TR of 0. '
                                'Will cause a warning to be thrown if it '
                                'mismatches a nonzero TR in the header.'),
                          default=0.0
                          )
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
                    source_tes=-1, combmode='t2s', verbose=False, stabilize=False,
                    out_dir='.', fixed_seed=42, maxit=500, maxrestart=10,
                    debug=False, quiet=False,
                    png=False, png_cmap='coolwarm', user_tr=0.0
                    ):
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
        spatially aligned with `data`. If an explicit mask is not provided,
        then Nilearn's compute_epi_mask function will be used to derive a mask
        from the first echo's data.
    mixm : :obj:`str`, optional
        File containing mixing matrix. If not provided, ME-PCA and ME-ICA are
        done.
    ctab : :obj:`str`, optional
        File containing component table from which to extract pre-computed
        classifications.
    manacc : :obj:`list`, :obj:`str`, or None, optional
        List of manually accepted components. Can be a list of the components,
        a comma-separated string with component numbers, or None. Default is
        None.
    tedort : :obj:`bool`, optional
        Orthogonalize rejected components w.r.t. accepted ones prior to
        denoising. Default is False.
    gscontrol : {None, 't1c', 'gsr'} or :obj:`list`, optional
        Perform additional denoising to remove spatially diffuse noise. Default
        is None.
    tedpca : {'mle', 'kundu', 'kundu-stabilize'}, optional
        Method with which to select components in TEDPCA. Default is 'mle'.
    source_tes : :obj:`int`, optional
        Source TEs for models. 0 for all, -1 for optimal combination.
        Default is -1.
    combmode : {'t2s'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default).
    verbose : :obj:`bool`, optional
        Generate intermediate and additional files. Default is False.
    png : obj:'bool', optional
        Generate simple plots and figures. Default is false.
    png_cmap : obj:'str', optional
            Name of a matplotlib colormap to be used when generating figures.
            --png must still be used to request figures. Default is 'coolwarm'
    out_dir : :obj:`str`, optional
        Output directory.

    Other Parameters
    ----------------
    fixed_seed : :obj:`int`, optional
        Value passed to ``mdp.numx_rand.seed()``.
        Set to a positive integer value for reproducible ICA results;
        otherwise, set to -1 for varying results across calls.
    maxit : :obj:`int`, optional
        Maximum number of iterations for ICA. Default is 500.
    maxrestart : :obj:`int`, optional
        Maximum number of attempts for ICA. If ICA fails to converge, the
        fixed seed will be updated and ICA will be run again. If convergence
        is achieved before maxrestart attempts, ICA will finish early.
        Default is 10.
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
        # ensure old logs aren't over-written
        basename = 'tedana_run'
        extension = 'txt'
        logname = op.join(out_dir, (basename + '.' + extension))
        logex = op.join(out_dir, (basename + '*'))
        previouslogs = glob.glob(logex)
        previouslogs.sort(reverse=True)
        for f in previouslogs:
            previousparts = op.splitext(f)
            newname = previousparts[0] + '_old' + previousparts[1]
            os.rename(f, newname)

        # set logging format
        formatter = logging.Formatter(
                    '%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S')

        # set up logging file and open it for writing
        fh = logging.FileHandler(logname)
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

    # check if TR is 0
    tr = ref_img.header.get_zooms()[-1]
    if tr == 0 and user_tr == 0:
        raise IOError(' Dataset has a TR of 0. This indicates incorrect'
                      ' header information. Please override the TR value'
                      ' with the --TR flag (see tedana -h for more help)'
                      ' or fix your file header.')
    elif user_tr != 0:
        # Coerce TR to be user-supplied value
        zooms = ref_img.header.get_zooms()
        new_zooms = (zooms[0], zooms[1], zooms[2], user_tr)
        ref_img.header.set_zooms(new_zooms)

        if tr != user_tr:
            LGR.warning('Mismatch in header TR and user-supplied TR,'
                        ' please verify. Proceeding anyway.')

    if mixm is not None and op.isfile(mixm):
        mixm = op.abspath(mixm)
        # Allow users to re-run on same folder
        if mixm != op.join(out_dir, 'meica_mix.1D'):
            shutil.copyfile(mixm, op.join(out_dir, 'meica_mix.1D'))
            shutil.copyfile(mixm, op.join(out_dir, op.basename(mixm)))
    elif mixm is not None:
        raise IOError('Argument "mixm" must be an existing file.')

    if ctab is not None and op.isfile(ctab):
        ctab = op.abspath(ctab)
        # Allow users to re-run on same folder
        if ctab != op.join(out_dir, 'comp_table_ica.txt'):
            shutil.copyfile(ctab, op.join(out_dir, 'comp_table_ica.txt'))
            shutil.copyfile(ctab, op.join(out_dir, op.basename(ctab)))
    elif ctab is not None:
        raise IOError('Argument "ctab" must be an existing file.')

    if isinstance(manacc, str):
        manacc = [int(comp) for comp in manacc.split(',')]

    if ctab and not mixm:
        LGR.warning('Argument "ctab" requires argument "mixm".')
        ctab = None
    elif ctab and (manacc is None):
        LGR.warning('Argument "ctab" requires argument "manacc".')
        ctab = None
    elif manacc is not None and not mixm:
        LGR.warning('Argument "manacc" requires argument "mixm".')
        manacc = None

    if mask is None:
        LGR.info('Computing EPI mask from first echo')
        first_echo_img = io.new_nii_like(ref_img, catd[:, 0, :])
        mask = compute_epi_mask(first_echo_img)
    else:
        # TODO: add affine check
        LGR.info('Using user-defined mask')

    mask, masksum = utils.make_adaptive_mask(catd, mask=mask, getsum=True)
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
        catd, data_oc = gsc.gscontrol_raw(catd, data_oc, n_echos, ref_img)

    if mixm is None:
        # Identify and remove thermal noise from data
        dd, n_components = decomposition.tedpca(catd, data_oc, combmode, mask,
                                                t2s, t2sG, ref_img,
                                                tes=tes, algorithm=tedpca,
                                                source_tes=source_tes,
                                                kdaw=10., rdaw=1.,
                                                out_dir=out_dir, verbose=verbose)
        mmix_orig = decomposition.tedica(dd, n_components, fixed_seed,
                                         maxit, maxrestart)

        if verbose:
            np.savetxt(op.join(out_dir, '__meica_mix.1D'), mmix_orig)
            if source_tes == -1:
                io.filewrite(utils.unmask(dd, mask),
                             op.join(out_dir, 'ts_OC_whitened.nii'), ref_img)

        LGR.info('Making second component selection guess from ICA results')
        # Estimate betas and compute selection metrics for mixing matrix
        # generated from dimensionally reduced data using full data (i.e., data
        # with thermal noise)
        comptable, metric_maps, betas, mmix = metrics.dependence_metrics(
                    catd, data_oc, mmix_orig, mask, t2s, tes,
                    ref_img, reindex=True, label='meica_', out_dir=out_dir,
                    algorithm='kundu_v2', verbose=verbose)
        np.savetxt(op.join(out_dir, 'meica_mix.1D'), mmix)

        comptable = metrics.kundu_metrics(comptable, metric_maps)
        comptable = selection.kundu_selection_v2(comptable, n_echos, n_vols)
    else:
        LGR.info('Using supplied mixing matrix from ICA')
        mmix_orig = np.loadtxt(op.join(out_dir, 'meica_mix.1D'))
        comptable, metric_maps, betas, mmix = metrics.dependence_metrics(
                    catd, data_oc, mmix_orig, mask, t2s, tes,
                    ref_img, label='meica_', out_dir=out_dir,
                    algorithm='kundu_v2', verbose=verbose)
        if ctab is None:
            comptable = metrics.kundu_metrics(comptable, metric_maps)
            comptable = selection.kundu_selection_v2(comptable, n_echos, n_vols)
        else:
            comptable = pd.read_csv(ctab, sep='\t', index_col='component')
            comptable = selection.manual_selection(comptable, acc=manacc)

    comptable.to_csv(op.join(out_dir, 'comp_table_ica.txt'), sep='\t',
                     index=True, index_label='component', float_format='%.6f')

    if comptable[comptable.classification == 'accepted'].shape[0] == 0:
        LGR.warning('No BOLD components detected! Please check data and '
                    'results!')

    mmix_orig = mmix.copy()
    if tedort:
        acc_idx = comptable.loc[
            ~comptable.classification.str.contains('rejected')].index.values
        rej_idx = comptable.loc[
            comptable.classification.str.contains('rejected')].index.values
        acc_ts = mmix[:, acc_idx]
        rej_ts = mmix[:, rej_idx]
        betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
        pred_rej_ts = np.dot(acc_ts, betas)
        resid = rej_ts - pred_rej_ts
        mmix[:, rej_idx] = resid
        np.savetxt(op.join(out_dir, 'meica_mix_orth.1D'), mmix)

    io.writeresults(data_oc, mask=mask, comptable=comptable, mmix=mmix,
                    n_vols=n_vols, ref_img=ref_img)

    if 't1c' in gscontrol:
        LGR.info('Performing T1c global signal regression to remove spatially '
                 'diffuse noise')
        gsc.gscontrol_mmix(data_oc, mmix, mask, comptable, ref_img)

    if verbose:
        io.writeresults_echoes(catd, mmix, mask, comptable, ref_img)

    if png:
        LGR.info('Making figures folder with static component maps and '
                 'timecourse plots.')
        # make figure folder first
        if not op.isdir(op.join(out_dir, 'figures')):
            os.mkdir(op.join(out_dir, 'figures'))

        viz.write_comp_figs(data_oc, mask=mask, comptable=comptable,
                            mmix=mmix_orig, ref_img=ref_img,
                            out_dir=op.join(out_dir, 'figures'),
                            png_cmap=png_cmap)

        LGR.info('Making Kappa vs Rho scatter plot')
        viz.write_kappa_scatter(comptable=comptable,
                                out_dir=op.join(out_dir, 'figures'))

        LGR.info('Making overall summary figure')
        viz.write_summary_fig(comptable=comptable,
                              out_dir=op.join(out_dir, 'figures'))

    LGR.info('Workflow completed')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def _main(argv=None):
    """Tedana entry point"""
    options = _get_parser().parse_args(argv)
    tedana_workflow(**vars(options))


if __name__ == '__main__':
    _main()
