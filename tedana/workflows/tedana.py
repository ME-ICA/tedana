"""
Run the "canonical" TE-Dependent ANAlysis workflow.
"""
import argparse
import datetime
import json
import logging
import os
import os.path as op
import shutil
import sys
from glob import glob

import numpy as np
import pandas as pd
from nilearn.masking import compute_epi_mask
from scipy import stats
from threadpoolctl import threadpool_limits

import tedana.gscontrol as gsc
from tedana import (
    __version__,
    combine,
    decay,
    decomposition,
    io,
    metrics,
    reporting,
    selection,
    utils,
)
from tedana.stats import computefeats2
from tedana.workflows.parser_utils import check_tedpca_value, is_valid_file

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")
RefLGR = logging.getLogger("REFERENCES")


def _get_parser():
    """
    Parses command line inputs for tedana

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    from tedana import __version__

    verstr = "tedana v{}".format(__version__)
    parser = argparse.ArgumentParser()
    # Argument parser follow templtate provided by RalphyZ
    # https://stackoverflow.com/a/43456577
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-d",
        dest="data",
        nargs="+",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Multi-echo dataset for analysis. May be a "
            "single file with spatially concatenated data "
            "or a set of echo-specific files, in the same "
            "order as the TEs are listed in the -e "
            "argument."
        ),
        required=True,
    )
    required.add_argument(
        "-e",
        dest="tes",
        nargs="+",
        metavar="TE",
        type=float,
        help="Echo times (in ms). E.g., 15.0 39.0 63.0",
        required=True,
    )
    optional.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default=".",
    )
    optional.add_argument(
        "--mask",
        dest="mask",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Binary mask of voxels to include in TE "
            "Dependent ANAlysis. Must be in the same "
            "space as `data`. If an explicit mask is not "
            "provided, then Nilearn's compute_epi_mask "
            "function will be used to derive a mask "
            "from the first echo's data."
        ),
        default=None,
    )
    optional.add_argument(
        "--prefix", dest="prefix", type=str, help="Prefix for filenames generated.", default=""
    )
    optional.add_argument(
        "--convention",
        dest="convention",
        action="store",
        choices=["orig", "bids"],
        help=("Filenaming convention. bids will use the latest BIDS derivatives version."),
        default="bids",
    )
    optional.add_argument(
        "--fittype",
        dest="fittype",
        action="store",
        choices=["loglin", "curvefit"],
        help=(
            "Desired T2*/S0 fitting method. "
            '"loglin" means that a linear model is fit '
            "to the log of the data. "
            '"curvefit" means that a more computationally '
            "demanding monoexponential model is fit "
            "to the raw data. "
            'Default is "loglin".'
        ),
        default="loglin",
    )
    optional.add_argument(
        "--combmode",
        dest="combmode",
        action="store",
        choices=["t2s"],
        help=("Combination scheme for TEs: t2s (Posse 1999, default)"),
        default="t2s",
    )
    optional.add_argument(
        "--tedpca",
        dest="tedpca",
        type=check_tedpca_value,
        help=(
            "Method with which to select components in TEDPCA. "
            "PCA decomposition with the mdl, kic and aic options "
            "is based on a Moving Average (stationary Gaussian) "
            "process and are ordered from most to least aggressive. "
            "Users may also provide a float from 0 to 1, "
            "in which case components will be selected based on the "
            "cumulative variance explained or an integer greater than 1"
            "in which case the specificed number of components will be"
            "selected."
            "Default='aic'."
        ),
        default="aic",
    )
    optional.add_argument(
        "--seed",
        dest="fixed_seed",
        metavar="INT",
        type=int,
        help=(
            "Value used for random initialization of ICA "
            "algorithm. Set to an integer value for "
            "reproducible ICA results. Set to -1 for "
            "varying results across ICA calls. "
            "Default=42."
        ),
        default=42,
    )
    optional.add_argument(
        "--maxit",
        dest="maxit",
        metavar="INT",
        type=int,
        help=("Maximum number of iterations for ICA."),
        default=500,
    )
    optional.add_argument(
        "--maxrestart",
        dest="maxrestart",
        metavar="INT",
        type=int,
        help=(
            "Maximum number of attempts for ICA. If ICA "
            "fails to converge, the fixed seed will be "
            "updated and ICA will be run again. If "
            "convergence is achieved before maxrestart "
            "attempts, ICA will finish early."
        ),
        default=10,
    )
    optional.add_argument(
        "--tedort",
        dest="tedort",
        action="store_true",
        help=("Orthogonalize rejected components w.r.t. accepted components prior to denoising."),
        default=False,
    )
    optional.add_argument(
        "--gscontrol",
        dest="gscontrol",
        required=False,
        action="store",
        nargs="+",
        help=(
            "Perform additional denoising to remove "
            "spatially diffuse noise. Default is None. "
            "This argument can be single value or a space "
            "delimited list"
        ),
        choices=["mir", "gsr"],
        default=None,
    )
    optional.add_argument(
        "--no-reports",
        dest="no_reports",
        action="store_true",
        help=(
            "Creates a figures folder with static component "
            "maps, timecourse plots and other diagnostic "
            "images and displays these in an interactive "
            "reporting framework"
        ),
        default=False,
    )
    optional.add_argument(
        "--png-cmap", dest="png_cmap", type=str, help="Colormap for figures", default="coolwarm"
    )
    optional.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Generate intermediate and additional files.",
        default=False,
    )
    optional.add_argument(
        "--lowmem",
        dest="low_mem",
        action="store_true",
        help=(
            "Enables low-memory processing, including the "
            "use of IncrementalPCA. May increase workflow "
            "duration."
        ),
        default=False,
    )
    optional.add_argument(
        "--n-threads",
        dest="n_threads",
        type=int,
        action="store",
        help=(
            "Number of threads to use. Used by "
            "threadpoolctl to set the parameter outside "
            "of the workflow function. Higher numbers of "
            "threads tend to slow down performance on "
            "typical datasets. Default is 1."
        ),
        default=1,
    )
    optional.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=(
            "Logs in the terminal will have increased "
            "verbosity, and will also be written into "
            "a .tsv file in the output directory."
        ),
        default=False,
    )
    optional.add_argument(
        "--quiet", dest="quiet", help=argparse.SUPPRESS, action="store_true", default=False
    )
    optional.add_argument("-v", "--version", action="version", version=verstr)
    parser._action_groups.append(optional)

    rerungrp = parser.add_argument_group("Arguments for Rerunning the Workflow")
    rerungrp.add_argument(
        "--t2smap",
        dest="t2smap",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=("Precalculated T2* map in the same space as the input data."),
        default=None,
    )
    rerungrp.add_argument(
        "--mix",
        dest="mixm",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=("File containing mixing matrix. If not provided, ME-PCA & ME-ICA is done."),
        default=None,
    )
    rerungrp.add_argument(
        "--ctab",
        dest="ctab",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "File containing a component table from which "
            "to extract pre-computed classifications. "
            "Requires --mix."
        ),
        default=None,
    )
    rerungrp.add_argument(
        "--manacc",
        dest="manacc",
        metavar="INT",
        type=int,
        nargs="+",
        help=("List of manually accepted components. Requires --ctab and --mix."),
        default=None,
    )

    return parser


def tedana_workflow(
    data,
    tes,
    out_dir=".",
    mask=None,
    convention="bids",
    prefix="",
    fittype="loglin",
    combmode="t2s",
    tedpca="aic",
    fixed_seed=42,
    maxit=500,
    maxrestart=10,
    tedort=False,
    gscontrol=None,
    no_reports=False,
    png_cmap="coolwarm",
    verbose=False,
    low_mem=False,
    debug=False,
    quiet=False,
    t2smap=None,
    mixm=None,
    ctab=None,
    manacc=None,
):
    """
    Run the "canonical" TE-Dependent ANAlysis workflow.

    Please remember to cite [1]_.

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
        Default is 'aic'.
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
        If provided, this parameter requires ``mixm`` and ``ctab`` to be provided as well.
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

    References
    ----------
    .. [1] DuPre, E. M., Salo, T., Ahmed, Z., Bandettini, P. A., Bottenhorn, K. L.,
           Caballero-Gaudes, C., Dowdle, L. T., Gonzalez-Castillo, J., Heunis, S.,
           Kundu, P., Laird, A. R., Markello, R., Markiewicz, C. J., Moia, S.,
           Staden, I., Teves, J. B., Uruñuela, E., Vaziri-Pashkam, M.,
           Whitaker, K., & Handwerker, D. A. (2021).
           TE-dependent analysis of multi-echo fMRI with tedana.
           Journal of Open Source Software, 6(66), 3669. doi:10.21105/joss.03669.
    """
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # boilerplate
    basename = "report"
    extension = "txt"
    repname = op.join(out_dir, (basename + "." + extension))
    repex = op.join(out_dir, (basename + "*"))
    previousreps = glob(repex)
    previousreps.sort(reverse=True)
    for f in previousreps:
        previousparts = op.splitext(f)
        newname = previousparts[0] + "_old" + previousparts[1]
        os.rename(f, newname)
    refname = op.join(out_dir, "_references.txt")

    # create logfile name
    basename = "tedana_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, (basename + start_time + "." + extension))
    utils.setup_loggers(logname, repname, refname, quiet=quiet, debug=debug)

    LGR.info("Using output directory: {}".format(out_dir))

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # Coerce gscontrol to list
    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]

    # Check value of tedpca *if* it is a predefined string,
    # a float on [0, 1] or an int >= 1
    tedpca = check_tedpca_value(tedpca, is_parser=False)

    LGR.info("Loading input data: {}".format([f for f in data]))
    catd, ref_img = io.load_data(data, n_echos=n_echos)
    io_generator = io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        verbose=verbose,
    )

    n_samp, n_echos, n_vols = catd.shape
    LGR.debug("Resulting data shape: {}".format(catd.shape))

    # check if TR is 0
    img_t_r = io_generator.reference_img.header.get_zooms()[-1]
    if img_t_r == 0:
        raise IOError(
            "Dataset has a TR of 0. This indicates incorrect"
            " header information. To correct this, we recommend"
            " using this snippet:"
            "\n"
            "https://gist.github.com/jbteves/032c87aeb080dd8de8861cb151bff5d6"
            "\n"
            "to correct your TR to the value it should be."
        )

    if mixm is not None and op.isfile(mixm):
        mixm = op.abspath(mixm)
        # Allow users to re-run on same folder
        mixing_name = io_generator.get_name("ICA mixing tsv")
        if mixm != mixing_name:
            shutil.copyfile(mixm, mixing_name)
            shutil.copyfile(mixm, op.join(io_generator.out_dir, op.basename(mixm)))
    elif mixm is not None:
        raise IOError("Argument 'mixm' must be an existing file.")

    if ctab is not None and op.isfile(ctab):
        ctab = op.abspath(ctab)
        # Allow users to re-run on same folder
        metrics_name = io_generator.get_name("ICA metrics tsv")
        if ctab != metrics_name:
            shutil.copyfile(ctab, metrics_name)
            shutil.copyfile(ctab, op.join(io_generator.out_dir, op.basename(ctab)))
    elif ctab is not None:
        raise IOError("Argument 'ctab' must be an existing file.")

    if ctab and not mixm:
        LGR.warning("Argument 'ctab' requires argument 'mixm'.")
        ctab = None
    elif manacc is not None and (not mixm or not ctab):
        LGR.warning("Argument 'manacc' requires arguments 'mixm' and 'ctab'.")
        manacc = None
    elif manacc is not None:
        # coerce to list of integers
        manacc = [int(m) for m in manacc]

    if t2smap is not None and op.isfile(t2smap):
        t2smap_file = io_generator.get_name("t2star img")
        t2smap = op.abspath(t2smap)
        # Allow users to re-run on same folder
        if t2smap != t2smap_file:
            shutil.copyfile(t2smap, t2smap_file)
    elif t2smap is not None:
        raise IOError("Argument 't2smap' must be an existing file.")

    RepLGR.info(
        "TE-dependence analysis was performed on input data using the tedana workflow "
        "(DuPre, Salo et al., 2021)."
    )
    RefLGR.info(
        "DuPre, E. M., Salo, T., Ahmed, Z., Bandettini, P. A., Bottenhorn, K. L., "
        "Caballero-Gaudes, C., Dowdle, L. T., Gonzalez-Castillo, J., Heunis, S., "
        "Kundu, P., Laird, A. R., Markello, R., Markiewicz, C. J., Moia, S., "
        "Staden, I., Teves, J. B., Uruñuela, E., Vaziri-Pashkam, M., "
        "Whitaker, K., & Handwerker, D. A. (2021). "
        "TE-dependent analysis of multi-echo fMRI with tedana. "
        "Journal of Open Source Software, 6(66), 3669. doi:10.21105/joss.03669."
    )

    if mask and not t2smap:
        # TODO: add affine check
        LGR.info("Using user-defined mask")
        RepLGR.info("A user-defined mask was applied to the data.")
    elif t2smap and not mask:
        LGR.info("Using user-defined T2* map to generate mask")
        t2s_limited_sec = utils.reshape_niimg(t2smap)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        t2s_full = t2s_limited.copy()
        mask = (t2s_limited != 0).astype(int)
    elif t2smap and mask:
        LGR.info("Combining user-defined mask and T2* map to generate mask")
        t2s_limited_sec = utils.reshape_niimg(t2smap)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        t2s_full = t2s_limited.copy()
        mask = utils.reshape_niimg(mask)
        mask[t2s_limited == 0] = 0  # reduce mask based on T2* map
    else:
        LGR.info("Computing EPI mask from first echo")
        first_echo_img = io.new_nii_like(io_generator.reference_img, catd[:, 0, :])
        mask = compute_epi_mask(first_echo_img)
        RepLGR.info(
            "An initial mask was generated from the first echo using "
            "nilearn's compute_epi_mask function."
        )

    # Create an adaptive mask with at least 1 good echo, for denoising
    mask_denoise, masksum_denoise = utils.make_adaptive_mask(
        catd,
        mask=mask,
        getsum=True,
        threshold=1,
    )
    LGR.debug("Retaining {}/{} samples for denoising".format(mask_denoise.sum(), n_samp))
    io_generator.save_file(masksum_denoise, "adaptive mask img")

    # Create an adaptive mask with at least 3 good echoes, for classification
    masksum_clf = masksum_denoise.copy()
    masksum_clf[masksum_clf < 3] = 0
    mask_clf = masksum_clf.astype(bool)
    RepLGR.info(
        "A two-stage masking procedure was applied, in which a liberal mask "
        "(including voxels with good data in at least the first echo) was used for "
        "optimal combination, T2*/S0 estimation, and denoising, while a more conservative mask "
        "(restricted to voxels with good data in at least the first three echoes) was used for "
        "the component classification procedure."
    )
    LGR.debug("Retaining {}/{} samples for classification".format(mask_clf.sum(), n_samp))

    if t2smap is None:
        LGR.info("Computing T2* map")
        t2s_limited, s0_limited, t2s_full, s0_full = decay.fit_decay(
            catd, tes, mask_denoise, masksum_denoise, fittype
        )

        # set a hard cap for the T2* map
        # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
        cap_t2s = stats.scoreatpercentile(t2s_full.flatten(), 99.5, interpolation_method="lower")
        LGR.debug("Setting cap on T2* map at {:.5f}s".format(utils.millisec2sec(cap_t2s)))
        t2s_full[t2s_full > cap_t2s * 10] = cap_t2s
        io_generator.save_file(utils.millisec2sec(t2s_full), "t2star img")
        io_generator.save_file(s0_full, "s0 img")

        if verbose:
            io_generator.save_file(utils.millisec2sec(t2s_limited), "limited t2star img")
            io_generator.save_file(s0_limited, "limited s0 img")

    # optimally combine data
    data_oc = combine.make_optcom(catd, tes, masksum_denoise, t2s=t2s_full, combmode=combmode)

    # regress out global signal unless explicitly not desired
    if "gsr" in gscontrol:
        catd, data_oc = gsc.gscontrol_raw(catd, data_oc, n_echos, io_generator)

    fout = io_generator.save_file(data_oc, "combined img")
    LGR.info("Writing optimally combined data set: {}".format(fout))

    if mixm is None:
        # Identify and remove thermal noise from data
        dd, n_components = decomposition.tedpca(
            catd,
            data_oc,
            combmode,
            mask_clf,
            masksum_clf,
            t2s_full,
            io_generator,
            tes=tes,
            algorithm=tedpca,
            kdaw=10.0,
            rdaw=1.0,
            verbose=verbose,
            low_mem=low_mem,
        )
        if verbose:
            io_generator.save_file(utils.unmask(dd, mask_clf), "whitened img")

        # Perform ICA, calculate metrics, and apply decision tree
        # Restart when ICA fails to converge or too few BOLD components found
        keep_restarting = True
        n_restarts = 0
        seed = fixed_seed
        while keep_restarting:
            mmix, seed = decomposition.tedica(
                dd, n_components, seed, maxit, maxrestart=(maxrestart - n_restarts)
            )
            seed += 1
            n_restarts = seed - fixed_seed

            # Estimate betas and compute selection metrics for mixing matrix
            # generated from dimensionally reduced data using full data (i.e., data
            # with thermal noise)
            LGR.info("Making second component selection guess from ICA results")
            required_metrics = [
                "kappa",
                "rho",
                "countnoise",
                "countsigFT2",
                "countsigFS0",
                "dice_FT2",
                "dice_FS0",
                "signal-noise_t",
                "variance explained",
                "normalized variance explained",
                "d_table_score",
            ]
            comptable = metrics.collect.generate_metrics(
                catd,
                data_oc,
                mmix,
                masksum_clf,
                tes,
                io_generator,
                "ICA",
                metrics=required_metrics,
            )
            comptable, metric_metadata = selection.kundu_selection_v2(comptable, n_echos, n_vols)

            n_bold_comps = comptable[comptable.classification == "accepted"].shape[0]
            if (n_restarts < maxrestart) and (n_bold_comps == 0):
                LGR.warning("No BOLD components found. Re-attempting ICA.")
            elif n_bold_comps == 0:
                LGR.warning("No BOLD components found, but maximum number of restarts reached.")
                keep_restarting = False
            else:
                keep_restarting = False

            RepLGR.disabled = True  # Disable the report to avoid duplicate text
        RepLGR.disabled = False  # Re-enable the report after the while loop is escaped
    else:
        LGR.info("Using supplied mixing matrix from ICA")
        mixing_file = io_generator.get_name("ICA mixing tsv")
        mmix = pd.read_table(mixing_file).values

        if ctab is None:
            required_metrics = [
                "kappa",
                "rho",
                "countnoise",
                "countsigFT2",
                "countsigFS0",
                "dice_FT2",
                "dice_FS0",
                "signal-noise_t",
                "variance explained",
                "normalized variance explained",
                "d_table_score",
            ]
            comptable = metrics.collect.generate_metrics(
                catd,
                data_oc,
                mmix,
                masksum_clf,
                tes,
                io_generator,
                "ICA",
                metrics=required_metrics,
            )
            comptable, metric_metadata = selection.kundu_selection_v2(comptable, n_echos, n_vols)
        else:
            LGR.info("Using supplied component table for classification")
            comptable = pd.read_table(ctab)
            # Change rationale value of rows with NaN to empty strings
            comptable.loc[comptable.rationale.isna(), "rationale"] = ""

            if manacc is not None:
                comptable, metric_metadata = selection.manual_selection(comptable, acc=manacc)

    # Write out ICA files.
    comp_names = comptable["Component"].values
    mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
    io_generator.save_file(mixing_df, "ICA mixing tsv")
    betas_oc = utils.unmask(computefeats2(data_oc, mmix, mask_denoise), mask_denoise)
    io_generator.save_file(betas_oc, "z-scored ICA components img")

    # Save component table and associated json
    io_generator.save_file(comptable, "ICA metrics tsv")
    metric_metadata = metrics.collect.get_metadata(comptable)
    io_generator.save_file(metric_metadata, "ICA metrics json")

    decomp_metadata = {
        "Method": (
            "Independent components analysis with FastICA algorithm implemented by sklearn. "
        ),
    }
    for comp_name in comp_names:
        decomp_metadata[comp_name] = {
            "Description": "ICA fit to dimensionally-reduced optimally combined data.",
            "Method": "tedana",
        }
    with open(io_generator.get_name("ICA decomposition json"), "w") as fo:
        json.dump(decomp_metadata, fo, sort_keys=True, indent=4)

    if comptable[comptable.classification == "accepted"].shape[0] == 0:
        LGR.warning("No BOLD components detected! Please check data and results!")

    mmix_orig = mmix.copy()
    if tedort:
        acc_idx = comptable.loc[~comptable.classification.str.contains("rejected")].index.values
        rej_idx = comptable.loc[comptable.classification.str.contains("rejected")].index.values
        acc_ts = mmix[:, acc_idx]
        rej_ts = mmix[:, rej_idx]
        betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
        pred_rej_ts = np.dot(acc_ts, betas)
        resid = rej_ts - pred_rej_ts
        mmix[:, rej_idx] = resid
        comp_names = [
            io.add_decomp_prefix(comp, prefix="ica", max_value=comptable.index.max())
            for comp in comptable.index.values
        ]
        mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
        io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")
        RepLGR.info(
            "Rejected components' time series were then "
            "orthogonalized with respect to accepted components' time "
            "series."
        )

    io.writeresults(
        data_oc,
        mask=mask_denoise,
        comptable=comptable,
        mmix=mmix,
        n_vols=n_vols,
        io_generator=io_generator,
    )

    if "mir" in gscontrol:
        gsc.minimum_image_regression(data_oc, mmix, mask_denoise, comptable, io_generator)

    if verbose:
        io.writeresults_echoes(catd, mmix, mask_denoise, comptable, io_generator)

    # Write out BIDS-compatible description file
    derivative_metadata = {
        "Name": "tedana Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "tedana",
                "Version": __version__,
                "Description": (
                    "A denoising pipeline for the identification and removal "
                    "of non-BOLD noise from multi-echo fMRI data."
                ),
                "CodeURL": "https://github.com/ME-ICA/tedana",
            }
        ],
    }
    with open(io_generator.get_name("data description json"), "w") as fo:
        json.dump(derivative_metadata, fo, sort_keys=True, indent=4)

    RepLGR.info(
        "This workflow used numpy (Van Der Walt, Colbert, & "
        "Varoquaux, 2011), scipy (Jones et al., 2001), pandas "
        "(McKinney, 2010), scikit-learn (Pedregosa et al., 2011), "
        "nilearn, and nibabel (Brett et al., 2019)."
    )
    RefLGR.info(
        "Van Der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). The "
        "NumPy array: a structure for efficient numerical computation. "
        "Computing in Science & Engineering, 13(2), 22."
    )
    RefLGR.info(
        "Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source "
        "Scientific Tools for Python, 2001-, http://www.scipy.org/"
    )
    RefLGR.info(
        "McKinney, W. (2010, June). Data structures for statistical "
        "computing in python. In Proceedings of the 9th Python in "
        "Science Conference (Vol. 445, pp. 51-56)."
    )
    RefLGR.info(
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., "
        "Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). "
        "Scikit-learn: Machine learning in Python. Journal of machine "
        "learning research, 12(Oct), 2825-2830."
    )
    RefLGR.info(
        "Brett, M., Markiewicz, C. J., Hanke, M., Côté, M.-A., "
        "Cipollini, B., McCarthy, P., … freec84. (2019, May 28). "
        "nipy/nibabel. Zenodo. http://doi.org/10.5281/zenodo.3233118"
    )

    RepLGR.info(
        "This workflow also used the Dice similarity index " "(Dice, 1945; Sørensen, 1948)."
    )
    RefLGR.info(
        "Dice, L. R. (1945). Measures of the amount of ecologic "
        "association between species. Ecology, 26(3), 297-302."
    )
    RefLGR.info(
        "Sørensen, T. J. (1948). A method of establishing groups of "
        "equal amplitude in plant sociology based on similarity of "
        "species content and its application to analyses of the "
        "vegetation on Danish commons. I kommission hos E. Munksgaard."
    )

    with open(repname, "r") as fo:
        report = [line.rstrip() for line in fo.readlines()]
        report = " ".join(report)
    with open(refname, "r") as fo:
        reference_list = sorted(list(set(fo.readlines())))
        references = "\n".join(reference_list)
    report += "\n\nReferences:\n\n" + references
    with open(repname, "w") as fo:
        fo.write(report)

    if not no_reports:
        LGR.info("Making figures folder with static component maps and timecourse plots.")

        dn_ts, hikts, lowkts = io.denoise_ts(data_oc, mmix, mask_denoise, comptable)

        reporting.static_figures.carpet_plot(
            optcom_ts=data_oc,
            denoised_ts=dn_ts,
            hikts=hikts,
            lowkts=lowkts,
            mask=mask_denoise,
            io_generator=io_generator,
            gscontrol=gscontrol,
        )
        reporting.static_figures.comp_figures(
            data_oc,
            mask=mask_denoise,
            comptable=comptable,
            mmix=mmix_orig,
            io_generator=io_generator,
            png_cmap=png_cmap,
        )

        if sys.version_info.major == 3 and sys.version_info.minor < 6:
            warn_msg = (
                "Reports requested but Python version is less than "
                "3.6.0. Dynamic reports will not be generated."
            )
            LGR.warn(warn_msg)
        else:
            LGR.info("Generating dynamic report")
            reporting.generate_report(io_generator, tr=img_t_r)

    LGR.info("Workflow completed")
    utils.teardown_loggers()
    os.remove(refname)


def _main(argv=None):
    """Tedana entry point"""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    n_threads = kwargs.pop("n_threads")
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        tedana_workflow(**kwargs)


if __name__ == "__main__":
    _main()
