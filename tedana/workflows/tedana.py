"""
Run the "canonical" TE-Dependent ANAlysis workflow.
"""
import os
import sys
import os.path as op
import shutil
import logging
import datetime
from glob import glob

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
from nilearn.masking import compute_epi_mask

from tedana import (decay, combine, decomposition, io, metrics,
                    reporting, selection, utils)
import tedana.gscontrol as gsc
from tedana.stats import computefeats2
from tedana.workflows.parser_utils import is_valid_file, check_tedpca_value, ContextFilter

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
    from ..info import __version__
    verstr = 'tedana v{}'.format(__version__)
    parser = argparse.ArgumentParser()
    # Argument parser follow templtate provided by RalphyZ
    # https://stackoverflow.com/a/43456577
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('Required Arguments')
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
                          help=("Binary mask of voxels to include in TE "
                                "Dependent ANAlysis. Must be in the same "
                                "space as `data`. If an explicit mask is not "
                                "provided, then Nilearn's compute_epi_mask "
                                "function will be used to derive a mask "
                                "from the first echo's data."),
                          default=None)
    optional.add_argument('--fittype',
                          dest='fittype',
                          action='store',
                          choices=['loglin', 'curvefit'],
                          help=('Desired T2*/S0 fitting method. '
                                '"loglin" means that a linear model is fit '
                                'to the log of the data. '
                                '"curvefit" means that a more computationally '
                                'demanding monoexponential model is fit '
                                'to the raw data. '
                                'Default is "loglin".'),
                          default='loglin')
    optional.add_argument('--combmode',
                          dest='combmode',
                          action='store',
                          choices=['t2s'],
                          help=('Combination scheme for TEs: '
                                't2s (Posse 1999, default)'),
                          default='t2s')
    optional.add_argument('--tedpca',
                          dest='tedpca',
                          type=check_tedpca_value,
                          help=('Method with which to select components in TEDPCA. '
                                'PCA decomposition with the mdl, kic and aic options '
                                'is based on a Moving Average (stationary Gaussian) '
                                'process and are ordered from most to least aggressive. '
                                "Users may also provide a float from 0 to 1, "
                                "in which case components will be selected based on the "
                                "cumulative variance explained. "
                                "Default='mdl'."),
                          default='mdl')
    optional.add_argument('--seed',
                          dest='fixed_seed',
                          metavar='INT',
                          type=int,
                          help=('Value used for random initialization of ICA '
                                'algorithm. Set to an integer value for '
                                'reproducible ICA results. Set to -1 for '
                                'varying results across ICA calls. '
                                'Default=42.'),
                          default=42)
    optional.add_argument('--maxit',
                          dest='maxit',
                          metavar='INT',
                          type=int,
                          help=('Maximum number of iterations for ICA.'),
                          default=500)
    optional.add_argument('--maxrestart',
                          dest='maxrestart',
                          metavar='INT',
                          type=int,
                          help=('Maximum number of attempts for ICA. If ICA '
                                'fails to converge, the fixed seed will be '
                                'updated and ICA will be run again. If '
                                'convergence is achieved before maxrestart '
                                'attempts, ICA will finish early.'),
                          default=10)
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
                          choices=['mir', 'gsr'],
                          default=None)
    optional.add_argument('--no-reports',
                          dest='no_reports',
                          action='store_true',
                          help=('Creates a figures folder with static component '
                                'maps, timecourse plots and other diagnostic '
                                'images and displays these in an interactive '
                                'reporting framework'),
                          default=False)
    optional.add_argument('--png-cmap',
                          dest='png_cmap',
                          type=str,
                          help='Colormap for figures',
                          default='coolwarm')
    optional.add_argument('--verbose',
                          dest='verbose',
                          action='store_true',
                          help='Generate intermediate and additional files.',
                          default=False)
    optional.add_argument('--lowmem',
                          dest='low_mem',
                          action='store_true',
                          help=('Enables low-memory processing, including the '
                                'use of IncrementalPCA. May increase workflow '
                                'duration.'),
                          default=False)
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
                          action='store_true',
                          help=('Logs in the terminal will have increased '
                                'verbosity, and will also be written into '
                                'a .tsv file in the output directory.'),
                          default=False)
    optional.add_argument('--quiet',
                          dest='quiet',
                          help=argparse.SUPPRESS,
                          action='store_true',
                          default=False)
    optional.add_argument('-v', '--version', action='version', version=verstr)
    parser._action_groups.append(optional)

    rerungrp = parser.add_argument_group('Arguments for Rerunning the Workflow')
    rerungrp.add_argument('--t2smap',
                          dest='t2smap',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('Precalculated T2* map in the same space as '
                                'the input data.'),
                          default=None)
    rerungrp.add_argument('--mix',
                          dest='mixm',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('File containing mixing matrix. If not '
                                'provided, ME-PCA & ME-ICA is done.'),
                          default=None)
    rerungrp.add_argument('--ctab',
                          dest='ctab',
                          metavar='FILE',
                          type=lambda x: is_valid_file(parser, x),
                          help=('File containing a component table from which '
                                'to extract pre-computed classifications.'),
                          default=None)
    rerungrp.add_argument('--manacc',
                          dest='manacc',
                          metavar='INT',
                          type=int,
                          nargs='+',
                          help='List of manually accepted components.',
                          default=None)

    return parser


def tedana_workflow(data, tes, out_dir='.', mask=None,
                    fittype='loglin', combmode='t2s', tedpca='mdl',
                    fixed_seed=42, maxit=500, maxrestart=10,
                    tedort=False, gscontrol=None,
                    no_reports=False, png_cmap='coolwarm',
                    verbose=False, low_mem=False, debug=False, quiet=False,
                    t2smap=None, mixm=None, ctab=None, manacc=None):
    """
    Run the "canonical" TE-Dependent ANAlysis workflow.

    Parameters
    ----------
    data : :obj:`str` or :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list or str) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.
    out_dir : :obj:`str`, optional
        Output directory.
    mask : :obj:`str` or None, optional
        Binary mask of voxels to include in TE Dependent ANAlysis. Must be
        spatially aligned with `data`. If an explicit mask is not provided,
        then Nilearn's compute_epi_mask function will be used to derive a mask
        from the first echo's data.
    fittype : {'loglin', 'curvefit'}, optional
        Monoexponential fitting method. 'loglin' uses the the default linear
        fit to the log of the data. 'curvefit' uses a monoexponential fit to
        the raw data, which is slightly slower but may be more accurate.
        Default is 'loglin'.
    combmode : {'t2s'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default).
    tedpca : {'mdl', 'aic', 'kic', 'kundu', 'kundu-stabilize', float}, optional
        Method with which to select components in TEDPCA.
        If a float is provided, then it is assumed to represent percentage of variance
        explained (0-1) to retain from PCA.
        Default is 'mdl'.
    tedort : :obj:`bool`, optional
        Orthogonalize rejected components w.r.t. accepted ones prior to
        denoising. Default is False.
    gscontrol : {None, 'mir', 'gsr'} or :obj:`list`, optional
        Perform additional denoising to remove spatially diffuse noise. Default
        is None.
    verbose : :obj:`bool`, optional
        Generate intermediate and additional files. Default is False.
    no_reports : obj:'bool', optional
        Do not generate .html reports and .png plots. Default is false such
        that reports are generated.
    png_cmap : obj:'str', optional
        Name of a matplotlib colormap to be used when generating figures.
        Cannot be used with --no-png. Default is 'coolwarm'.
    t2smap : :obj:`str`, optional
        Precalculated T2* map in the same space as the input data. Values in
        the map must be in seconds.
    mixm : :obj:`str` or None, optional
        File containing mixing matrix, to be used when re-running the workflow.
        If not provided, ME-PCA and ME-ICA are done. Default is None.
    ctab : :obj:`str` or None, optional
        File containing component table from which to extract pre-computed
        classifications, to be used with 'mixm' when re-running the workflow.
        Default is None.
    manacc : :obj:`list` of :obj:`int` or None, optional
        List of manually accepted components. Can be a list of the components
        numbers or None.
        Default is None.

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
    low_mem : :obj:`bool`, optional
        Enables low-memory processing, including the use of IncrementalPCA.
        May increase workflow duration. Default is False.
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

    # boilerplate
    basename = 'report'
    extension = 'txt'
    repname = op.join(out_dir, (basename + '.' + extension))
    repex = op.join(out_dir, (basename + '*'))
    previousreps = glob(repex)
    previousreps.sort(reverse=True)
    for f in previousreps:
        previousparts = op.splitext(f)
        newname = previousparts[0] + '_old' + previousparts[1]
        os.rename(f, newname)
    refname = op.join(out_dir, '_references.txt')

    # create logfile name
    basename = 'tedana_'
    extension = 'tsv'
    start_time = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
    logname = op.join(out_dir, (basename + start_time + '.' + extension))

    # set logging format
    log_formatter = logging.Formatter(
        '%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')
    text_formatter = logging.Formatter('%(message)s')

    # set up logging file and open it for writing
    log_handler = logging.FileHandler(logname)
    log_handler.setFormatter(log_formatter)
    # Removing handlers after basicConfig doesn't work, so we use filters
    # for the relevant handlers themselves.
    log_handler.addFilter(ContextFilter())
    logging.root.addHandler(log_handler)
    sh = logging.StreamHandler()
    sh.addFilter(ContextFilter())
    logging.root.addHandler(sh)

    if quiet:
        logging.root.setLevel(logging.WARNING)
    elif debug:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    # Loggers for report and references
    rep_handler = logging.FileHandler(repname)
    rep_handler.setFormatter(text_formatter)
    ref_handler = logging.FileHandler(refname)
    ref_handler.setFormatter(text_formatter)
    RepLGR.setLevel(logging.INFO)
    RepLGR.addHandler(rep_handler)
    RefLGR.setLevel(logging.INFO)
    RefLGR.addHandler(ref_handler)

    LGR.info('Using output directory: {}'.format(out_dir))

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # Coerce gscontrol to list
    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]

    # Check value of tedpca *if* it is a float
    tedpca = check_tedpca_value(tedpca, is_parser=False)

    LGR.info('Loading input data: {}'.format([f for f in data]))
    catd, ref_img = io.load_data(data, n_echos=n_echos)
    n_samp, n_echos, n_vols = catd.shape
    LGR.debug('Resulting data shape: {}'.format(catd.shape))

    # check if TR is 0
    img_t_r = ref_img.header.get_zooms()[-1]
    if img_t_r == 0:
        raise IOError('Dataset has a TR of 0. This indicates incorrect'
                      ' header information. To correct this, we recommend'
                      ' using this snippet:'
                      '\n'
                      'https://gist.github.com/jbteves/032c87aeb080dd8de8861cb151bff5d6'
                      '\n'
                      'to correct your TR to the value it should be.')

    if mixm is not None and op.isfile(mixm):
        mixm = op.abspath(mixm)
        # Allow users to re-run on same folder
        if mixm != op.join(out_dir, 'ica_mixing.tsv'):
            shutil.copyfile(mixm, op.join(out_dir, 'ica_mixing.tsv'))
            shutil.copyfile(mixm, op.join(out_dir, op.basename(mixm)))
    elif mixm is not None:
        raise IOError('Argument "mixm" must be an existing file.')

    if ctab is not None and op.isfile(ctab):
        ctab = op.abspath(ctab)
        # Allow users to re-run on same folder
        if ctab != op.join(out_dir, 'ica_decomposition.json'):
            shutil.copyfile(ctab, op.join(out_dir, 'ica_decomposition.json'))
            shutil.copyfile(ctab, op.join(out_dir, op.basename(ctab)))
    elif ctab is not None:
        raise IOError('Argument "ctab" must be an existing file.')

    if ctab and not mixm:
        LGR.warning('Argument "ctab" requires argument "mixm".')
        ctab = None
    elif manacc is not None and not mixm:
        LGR.warning('Argument "manacc" requires argument "mixm".')
        manacc = None
    elif manacc is not None:
        # coerce to list of integers
        manacc = [int(m) for m in manacc]

    if t2smap is not None and op.isfile(t2smap):
        t2smap = op.abspath(t2smap)
        # Allow users to re-run on same folder
        if t2smap != op.join(out_dir, 't2sv.nii.gz'):
            shutil.copyfile(t2smap, op.join(out_dir, 't2sv.nii.gz'))
            shutil.copyfile(t2smap, op.join(out_dir, op.basename(t2smap)))
    elif t2smap is not None:
        raise IOError('Argument "t2smap" must be an existing file.')

    RepLGR.info("TE-dependence analysis was performed on input data.")
    if mask and not t2smap:
        # TODO: add affine check
        LGR.info('Using user-defined mask')
        RepLGR.info("A user-defined mask was applied to the data.")
    elif t2smap and not mask:
        LGR.info('Using user-defined T2* map to generate mask')
        t2s_limited_sec = utils.load_image(t2smap)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        t2s_full = t2s_limited.copy()
        mask = (t2s_limited != 0).astype(int)
    elif t2smap and mask:
        LGR.info('Combining user-defined mask and T2* map to generate mask')
        t2s_limited_sec = utils.load_image(t2smap)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        t2s_full = t2s_limited.copy()
        mask = utils.load_image(mask)
        mask[t2s_limited == 0] = 0  # reduce mask based on T2* map
    else:
        LGR.info('Computing EPI mask from first echo')
        first_echo_img = io.new_nii_like(ref_img, catd[:, 0, :])
        mask = compute_epi_mask(first_echo_img)
        RepLGR.info("An initial mask was generated from the first echo using "
                    "nilearn's compute_epi_mask function.")

    # Create an adaptive mask with at least 3 good echoes.
    mask, masksum = utils.make_adaptive_mask(catd, mask=mask, getsum=True, threshold=3)
    LGR.debug('Retaining {}/{} samples'.format(mask.sum(), n_samp))
    io.filewrite(masksum, op.join(out_dir, 'adaptive_mask.nii'), ref_img)

    if t2smap is None:
        LGR.info('Computing T2* map')
        t2s_limited, s0_limited, t2s_full, s0_full = decay.fit_decay(
            catd, tes, mask, masksum, fittype)

        # set a hard cap for the T2* map
        # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
        cap_t2s = stats.scoreatpercentile(t2s_limited.flatten(), 99.5,
                                          interpolation_method='lower')
        LGR.debug('Setting cap on T2* map at {:.5f}s'.format(
            utils.millisec2sec(cap_t2s)))
        t2s_limited[t2s_limited > cap_t2s * 10] = cap_t2s
        io.filewrite(utils.millisec2sec(t2s_limited), op.join(out_dir, 't2sv.nii'), ref_img)
        io.filewrite(s0_limited, op.join(out_dir, 's0v.nii'), ref_img)

        if verbose:
            io.filewrite(utils.millisec2sec(t2s_full), op.join(out_dir, 't2svG.nii'), ref_img)
            io.filewrite(s0_full, op.join(out_dir, 's0vG.nii'), ref_img)

    # optimally combine data
    data_oc = combine.make_optcom(catd, tes, masksum, t2s=t2s_full, combmode=combmode)

    # regress out global signal unless explicitly not desired
    if 'gsr' in gscontrol:
        catd, data_oc = gsc.gscontrol_raw(catd, data_oc, n_echos, ref_img,
                                          out_dir=out_dir)

    if mixm is None:
        # Identify and remove thermal noise from data
        dd, n_components = decomposition.tedpca(catd, data_oc, combmode, mask,
                                                masksum, t2s_full, ref_img,
                                                tes=tes, algorithm=tedpca,
                                                kdaw=10., rdaw=1.,
                                                out_dir=out_dir,
                                                verbose=verbose,
                                                low_mem=low_mem)
        if verbose:
            io.filewrite(utils.unmask(dd, mask),
                         op.join(out_dir, 'ts_OC_whitened.nii.gz'), ref_img)

        # Perform ICA, calculate metrics, and apply decision tree
        # Restart when ICA fails to converge or too few BOLD components found
        keep_restarting = True
        n_restarts = 0
        seed = fixed_seed
        while keep_restarting:
            mmix_orig, seed = decomposition.tedica(
                dd, n_components, seed,
                maxit, maxrestart=(maxrestart - n_restarts)
            )
            seed += 1
            n_restarts = seed - fixed_seed

            # Estimate betas and compute selection metrics for mixing matrix
            # generated from dimensionally reduced data using full data (i.e., data
            # with thermal noise)
            LGR.info('Making second component selection guess from ICA results')
            comptable, metric_maps, betas, mmix = metrics.dependence_metrics(
                catd, data_oc, mmix_orig, masksum, tes,
                ref_img, reindex=True, label='meica_', out_dir=out_dir,
                algorithm='kundu_v2', verbose=verbose
            )
            comptable = metrics.kundu_metrics(comptable, metric_maps)
            comptable = selection.kundu_selection_v2(comptable, n_echos, n_vols)

            n_bold_comps = comptable[comptable.classification == 'accepted'].shape[0]
            if (n_restarts < maxrestart) and (n_bold_comps == 0):
                LGR.warning("No BOLD components found. Re-attempting ICA.")
            elif (n_bold_comps == 0):
                LGR.warning("No BOLD components found, but maximum number of restarts reached.")
                keep_restarting = False
            else:
                keep_restarting = False

        # Write out ICA files.
        comp_names = [io.add_decomp_prefix(comp, prefix='ica', max_value=comptable.index.max())
                      for comp in comptable.index.values]
        mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
        mixing_df.to_csv(op.join(out_dir, 'ica_mixing.tsv'), sep='\t', index=False)
        betas_oc = utils.unmask(computefeats2(data_oc, mmix, mask), mask)
        io.filewrite(betas_oc,
                     op.join(out_dir, 'ica_components.nii.gz'),
                     ref_img)
    else:
        LGR.info('Using supplied mixing matrix from ICA')
        mmix_orig = pd.read_table(op.join(out_dir, 'ica_mixing.tsv')).values

        if ctab is None:
            comptable, metric_maps, betas, mmix = metrics.dependence_metrics(
                        catd, data_oc, mmix_orig, masksum, tes,
                        ref_img, label='meica_', out_dir=out_dir,
                        algorithm='kundu_v2', verbose=verbose)
            comptable = metrics.kundu_metrics(comptable, metric_maps)
            comptable = selection.kundu_selection_v2(comptable, n_echos, n_vols)
        else:
            mmix = mmix_orig.copy()
            comptable = io.load_comptable(ctab)
            if manacc is not None:
                comptable = selection.manual_selection(comptable, acc=manacc)
        betas_oc = utils.unmask(computefeats2(data_oc, mmix, mask), mask)
        io.filewrite(betas_oc,
                     op.join(out_dir, 'ica_components.nii.gz'),
                     ref_img)

    # Save component table
    comptable['Description'] = 'ICA fit to dimensionally-reduced optimally combined data.'
    mmix_dict = {}
    mmix_dict['Method'] = ('Independent components analysis with FastICA '
                           'algorithm implemented by sklearn. Components '
                           'are sorted by Kappa in descending order. '
                           'Component signs are flipped to best match the '
                           'data.')
    io.save_comptable(comptable, op.join(out_dir, 'ica_decomposition.json'),
                      label='ica', metadata=mmix_dict)

    if comptable[comptable.classification == 'accepted'].shape[0] == 0:
        LGR.warning('No BOLD components detected! Please check data and '
                    'results!')

    mmix_orig = mmix.copy()
    if tedort:
        acc_idx = comptable.loc[~comptable.classification.str.
                                contains('rejected')].index.values
        rej_idx = comptable.loc[comptable.classification.str.contains(
            'rejected')].index.values
        acc_ts = mmix[:, acc_idx]
        rej_ts = mmix[:, rej_idx]
        betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
        pred_rej_ts = np.dot(acc_ts, betas)
        resid = rej_ts - pred_rej_ts
        mmix[:, rej_idx] = resid
        comp_names = [io.add_decomp_prefix(comp, prefix='ica', max_value=comptable.index.max())
                      for comp in comptable.index.values]
        mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
        mixing_df.to_csv(op.join(out_dir, 'ica_orth_mixing.tsv'), sep='\t', index=False)
        RepLGR.info("Rejected components' time series were then "
                    "orthogonalized with respect to accepted components' time "
                    "series.")

    io.writeresults(data_oc,
                    mask=mask,
                    comptable=comptable,
                    mmix=mmix,
                    n_vols=n_vols,
                    ref_img=ref_img,
                    out_dir=out_dir)

    if 'mir' in gscontrol:
        gsc.minimum_image_regression(data_oc, mmix, mask, comptable, ref_img, out_dir=out_dir)

    if verbose:
        io.writeresults_echoes(catd, mmix, mask, comptable, ref_img, out_dir=out_dir)

    if not no_reports:
        LGR.info('Making figures folder with static component maps and '
                 'timecourse plots.')
        # make figure folder first
        if not op.isdir(op.join(out_dir, 'figures')):
            os.mkdir(op.join(out_dir, 'figures'))

        reporting.static_figures.comp_figures(data_oc, mask=mask,
                                              comptable=comptable,
                                              mmix=mmix_orig,
                                              ref_img=ref_img,
                                              out_dir=op.join(out_dir,
                                                              'figures'),
                                              png_cmap=png_cmap)

        if sys.version_info.major == 3 and sys.version_info.minor < 6:
            warn_msg = ("Reports requested but Python version is less than "
                        "3.6.0. Dynamic reports will not be generated.")
            LGR.warn(warn_msg)
        else:
            LGR.info('Generating dynamic report')
            reporting.generate_report(out_dir=out_dir, tr=img_t_r)

    LGR.info('Workflow completed')

    RepLGR.info("This workflow used numpy (Van Der Walt, Colbert, & "
                "Varoquaux, 2011), scipy (Jones et al., 2001), pandas "
                "(McKinney, 2010), scikit-learn (Pedregosa et al., 2011), "
                "nilearn, and nibabel (Brett et al., 2019).")
    RefLGR.info("Van Der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). The "
                "NumPy array: a structure for efficient numerical computation. "
                "Computing in Science & Engineering, 13(2), 22.")
    RefLGR.info("Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source "
                "Scientific Tools for Python, 2001-, http://www.scipy.org/")
    RefLGR.info("McKinney, W. (2010, June). Data structures for statistical "
                "computing in python. In Proceedings of the 9th Python in "
                "Science Conference (Vol. 445, pp. 51-56).")
    RefLGR.info("Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., "
                "Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). "
                "Scikit-learn: Machine learning in Python. Journal of machine "
                "learning research, 12(Oct), 2825-2830.")
    RefLGR.info("Brett, M., Markiewicz, C. J., Hanke, M., Côté, M.-A., "
                "Cipollini, B., McCarthy, P., … freec84. (2019, May 28). "
                "nipy/nibabel. Zenodo. http://doi.org/10.5281/zenodo.3233118")

    RepLGR.info("This workflow also used the Dice similarity index "
                "(Dice, 1945; Sørensen, 1948).")
    RefLGR.info("Dice, L. R. (1945). Measures of the amount of ecologic "
                "association between species. Ecology, 26(3), 297-302.")
    RefLGR.info("Sørensen, T. J. (1948). A method of establishing groups of "
                "equal amplitude in plant sociology based on similarity of "
                "species content and its application to analyses of the "
                "vegetation on Danish commons. I kommission hos E. Munksgaard.")

    with open(repname, 'r') as fo:
        report = [line.rstrip() for line in fo.readlines()]
        report = ' '.join(report)
    with open(refname, 'r') as fo:
        reference_list = sorted(list(set(fo.readlines())))
        references = '\n'.join(reference_list)
    report += '\n\nReferences:\n\n' + references
    with open(repname, 'w') as fo:
        fo.write(report)

    log_handler.close()
    logging.root.removeHandler(log_handler)
    sh.close()
    logging.root.removeHandler(sh)
    for local_logger in (RefLGR, RepLGR):
        for handler in local_logger.handlers[:]:
            handler.close()
            local_logger.removeHandler(handler)
    os.remove(refname)


def _main(argv=None):
    """Tedana entry point"""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    n_threads = kwargs.pop('n_threads')
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        tedana_workflow(**kwargs)


if __name__ == '__main__':
    _main()
