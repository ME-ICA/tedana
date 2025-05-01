"""Run the "canonical" TE-Dependent ANAlysis workflow."""

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
from tedana.bibtex import get_description_references
from tedana.config import (
    DEFAULT_ICA_METHOD,
    DEFAULT_N_MAX_ITER,
    DEFAULT_N_MAX_RESTART,
    DEFAULT_N_ROBUST_RUNS,
    DEFAULT_SEED,
)
from tedana.selection.component_selector import ComponentSelector
from tedana.stats import computefeats2
from tedana.workflows.parser_utils import (
    check_n_robust_runs_value,
    check_tedpca_value,
    is_valid_file,
)

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def _get_parser():
    """Parse command line inputs for tedana.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    from tedana import __version__

    verstr = f"tedana v{__version__}"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Argument parser follow template provided by RalphyZ
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
            "from the first echo's data. "
            "Providing a mask is recommended."
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
        "--masktype",
        dest="masktype",
        required=False,
        action="store",
        nargs="+",
        help="Method(s) by which to define the adaptive mask.",
        choices=["dropout", "decay", "none"],
        default=["dropout"],
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
        ),
        default="loglin",
    )
    optional.add_argument(
        "--combmode",
        dest="combmode",
        action="store",
        choices=["t2s"],
        help=("Combination scheme for TEs: t2s (Posse 1999)"),
        default="t2s",
    )
    optional.add_argument(
        "--tedpca",
        dest="tedpca",
        type=check_tedpca_value,
        help=(
            "Method by which to select number of components in TEDPCA. "
            "This can be one of the following: "
            "String ('mdl', 'kic', 'aic', 'kundu', or 'kundu-stabilize'); "
            "floating-point value in the range (0.0, 1.0); "
            "positive integer value. "
            "PCA decomposition with the mdl, kic and aic options "
            "are based on a Moving Average (stationary Gaussian) process, "
            "and are ordered from most to least aggressive. "
            "'kundu' or 'kundu-stabilize' are legacy selection methods "
            "that were distributed with MEICA. "
            "Floating-point inputs select components based on the "
            "cumulative variance explained. "
            "Integer inputs select the specificed number of components. "
            "Default: 'aic'."
        ),
        default="aic",
    )
    optional.add_argument(
        "--tree",
        dest="tree",
        help=(
            "Decision tree to use. You may use a "
            "packaged tree (tedana_orig, meica, minimal) or supply a JSON "
            "file which matches the decision tree file "
            "specification. Minimal still being tested with more "
            "details in docs"
        ),
        default="tedana_orig",
    )
    optional.add_argument(
        "--external",
        dest="external_regressors",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "File containing external regressors to compare to ICA component be used in the "
            "decision tree. For example, to identify components fit head motion time series. "
            "The file must be a TSV file with the same number of rows as the number of volumes in "
            "the input data. Column labels and statistical tests are defined with external_labels."
        ),
        default=None,
    )
    optional.add_argument(
        "--ica-method",
        "--ica_method",
        dest="ica_method",
        help=(
            "The applied ICA method. "
            "fastica runs FastICA from sklearn once with the seed value. "
            "robustica will run FastICA n_robust_runs times and uses "
            "clustering methods to overcome the randomness of the FastICA algorithm. "
            "robustica will be slower."
        ),
        choices=["robustica", "fastica"],
        type=str.lower,
        default=DEFAULT_ICA_METHOD,
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
            "varying results across ICA calls. This "
            "applies to both fastica and robustica methods."
        ),
        default=DEFAULT_SEED,
    )
    optional.add_argument(
        "--n-robust-runs",
        "--n_robust_runs",
        dest="n_robust_runs",
        metavar="[5-500]",
        type=check_n_robust_runs_value,
        help=(
            "The number of times robustica will run. "
            "This is only effective when ica_method is "
            "set to robustica."
        ),
        default=DEFAULT_N_ROBUST_RUNS,
    )
    optional.add_argument(
        "--maxit",
        dest="maxit",
        metavar="INT",
        type=int,
        help=("Maximum number of iterations for ICA."),
        default=DEFAULT_N_MAX_ITER,
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
        default=DEFAULT_N_MAX_RESTART,
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
            "spatially diffuse noise. "
            "This argument can be single value or a space "
            "delimited list."
        ),
        choices=["mir", "gsr"],
        default="",
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
            "typical datasets."
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
        "--t2smap",
        dest="t2smap",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=("Precalculated T2* map in the same space as the input data."),
        default=None,
    )
    optional.add_argument(
        "--mix",
        dest="mixing_file",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=("File containing mixing matrix. If not provided, ME-PCA & ME-ICA is done."),
        default=None,
    )

    optional.add_argument(
        "--quiet", dest="quiet", help=argparse.SUPPRESS, action="store_true", default=False
    )
    optional.add_argument(
        "--overwrite",
        "-f",
        dest="overwrite",
        action="store_true",
        help="Force overwriting of files.",
        default=False,
    )

    optional.add_argument(
        "--n-independent-echos",
        dest="n_independent_echos",
        metavar="INT",
        type=int,
        help=(
            "Number of independent echoes to use in goodness of fit metrics (fstat)."
            "Primarily used for EPTI acquisitions."
            "If not provided, number of echoes will be used."
        ),
        default=None,
    )
    optional.add_argument("-v", "--version", action="version", version=verstr)
    parser._action_groups.append(optional)

    return parser


def tedana_workflow(
    data,
    tes,
    out_dir=".",
    mask=None,
    convention="bids",
    prefix="",
    masktype=["dropout"],
    fittype="loglin",
    combmode="t2s",
    n_independent_echos=None,
    tree="tedana_orig",
    external_regressors=None,
    ica_method=DEFAULT_ICA_METHOD,
    n_robust_runs=DEFAULT_N_ROBUST_RUNS,
    tedpca="aic",
    fixed_seed=DEFAULT_SEED,
    maxit=DEFAULT_N_MAX_ITER,
    maxrestart=DEFAULT_N_MAX_RESTART,
    tedort=False,
    gscontrol=None,
    no_reports=False,
    png_cmap="coolwarm",
    verbose=False,
    low_mem=False,
    debug=False,
    quiet=False,
    overwrite=False,
    t2smap=None,
    mixing_file=None,
    tedana_command=None,
):
    """Run the "canonical" TE-Dependent ANAlysis workflow.

    Please remember to cite :footcite:t:`dupre2021te`.

    Parameters
    ----------
    data : :obj:`str` or :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list or str) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.

    Other Parameters
    ----------------
    out_dir : :obj:`str`, optional
        Output directory.
    mask : :obj:`str` or None, optional
        Binary mask of voxels to include in TE Dependent ANAlysis. Must be
        spatially aligned with `data`. If an explicit mask is not provided,
        then Nilearn's compute_epi_mask function will be used to derive a mask
        from the first echo's data.
        Since most pipelines use better masking tools,
        providing a mask, rather than using compute_epi_mask, is recommended.
    convention : {'bids', 'orig'}, optional
        Filenaming convention. bids uses the latest BIDS derivatives version (1.5.0).
        Default is 'bids'.
    prefix : :obj:`str` or None, optional
        Prefix for filenames generated.
        Default is ""
    masktype : :obj:`list` with 'dropout' and/or 'decay' or None, optional
        Method(s) by which to define the adaptive mask. Default is ["dropout"].
    fittype : {'loglin', 'curvefit'}, optional
        Monoexponential fitting method. 'loglin' uses the the default linear
        fit to the log of the data. 'curvefit' uses a monoexponential fit to
        the raw data, which is slightly slower but may be more accurate.
        Default is 'loglin'.
    combmode : {'t2s'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default).
    n_independent_echos : :obj:`int`, optional
        Number of independent echoes to use in goodness of fit metrics (fstat).
        Primarily used for EPTI acquisitions.
        If None, number of echoes will be used. Default is None.
    tree : {'tedana_orig', 'meica', 'minimal', 'json file'}, optional
        Decision tree to use for component selection. Can be a
        packaged tree (tedana_orig, meica, minimal) or a user-supplied JSON file that
        matches the decision tree file specification. tedana_orig is the tree that has
        been distributed with tedana from the beginning and was designed to match the
        process in MEICA. A difference between that tree and the older MEICA was
        identified so the original meica tree is also included. meica will always
        accept the same or more components, but those accepted components are sometimes
        high variance so the differences can be non-trivial. Minimal is intended
        to be a simpler process, but it accepts and rejects some distinct components
        compared to the others. Testing to better understand the effects of the
        differences is ongoing. Default is 'tedana_orig'.
    external_regressors : :obj:`str` or None, optional
        File containing external regressors to be used in the decision tree.
        The file must be a TSV file with the same number of rows as the number of volumes in
        the input data. Each column in the file will be treated as a separate regressor.
        Default is None.
    ica_method : {'fastica', 'robustica'}, optional
        The applied ICA method. fastica runs FastICA from sklearn
        once with the seed value. 'robustica' will run
        'FastICA' n_robust_runs times and uses clustering methods to overcome
        the randomness of the FastICA algorithm.
        robustica will be slower.
        Default is 'fastica'
    n_robust_runs : :obj:`int`, optional
        The number of times robustica will run. This is only effective when 'ica_method' is
        set to 'robustica'.
    tedpca : {'mdl', 'aic', 'kic', 'kundu', 'kundu-stabilize', float, int}, optional
        Method with which to select components in TEDPCA.
        If a float is provided, then it is assumed to represent percentage of variance
        explained (0.0-1.0) to retain from PCA. If an int is provided, it will output
        a fixed number of components defined by the integer; must be between 2 and the
        number of time points. If 1 is provided as an integer, it will considered as 100%
        of the variance explained.
        Default is 'aic'.
    fixed_seed : :obj:`int`, optional
        Value passed to ``mdp.numx_rand.seed()``.
        Set to a positive integer value for reproducible ICA results (fastica/robustica);
        otherwise, set to -1 for varying results across ICA (fastica/robustica) calls.
    maxit : :obj:`int`, optional
        Maximum number of iterations for ICA. Default is 500.
    maxrestart : :obj:`int`, optional
        Maximum number of attempts for ICA. If ICA fails to converge, the
        fixed seed will be updated and ICA will be run again. If convergence
        is achieved before maxrestart attempts, ICA will finish early.
        Default is 10.
    tedort : :obj:`bool`, optional
        Orthogonalize rejected components w.r.t. accepted ones prior to
        denoising. Default is False.
    gscontrol : {None, 'mir', 'gsr'} or :obj:`list`, optional
        Perform additional denoising to remove spatially diffuse noise. Default
        is None.
    no_reports : obj:'bool', optional
        Do not generate .html reports and .png plots. Default is false such
        that reports are generated.
    png_cmap : obj:'str', optional
        Name of a matplotlib colormap to be used when generating figures.
        Cannot be used with --no-png. Default is 'coolwarm'.
    verbose : :obj:`bool`, optional
        Generate intermediate and additional files. Default is False.
    low_mem : :obj:`bool`, optional
        Enables low-memory processing, including the use of IncrementalPCA.
        May increase workflow duration. Default is False.
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    t2smap : :obj:`str`, optional
        Precalculated T2* map in the same space as the input data. Values in
        the map must be in seconds.
    mixing_file : :obj:`str` or None, optional
        File containing mixing matrix, to be used when re-running the workflow.
        If not provided, ME-PCA and ME-ICA are done. Default is None.
    quiet : :obj:`bool`, optional
        If True, suppresses logging/printing of messages. Default is False.
    overwrite : :obj:`bool`, optional
        If True, force overwriting of files. Default is False.
    tedana_command : :obj:`str`, optional
        If the command-line interface was used, this is the command that was
        run. Default is None.

    Notes
    -----
    This workflow writes out several files. For a complete list of the files
    generated by this workflow, please visit
    https://tedana.readthedocs.io/en/latest/outputs.html

    References
    ----------
    .. footbibliography::
    """
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # boilerplate
    prefix = io._infer_prefix(prefix)
    basename = f"{prefix}report"
    extension = "txt"
    repname = op.join(out_dir, (basename + "." + extension))
    bibtex_file = op.join(out_dir, f"{prefix}references.bib")
    repex = op.join(out_dir, (basename + "*"))
    previousreps = glob(repex)
    previousreps.sort(reverse=True)
    for f in previousreps:
        previousparts = op.splitext(f)
        newname = previousparts[0] + "_old" + previousparts[1]
        os.rename(f, newname)

    # create logfile name
    basename = "tedana_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, (basename + start_time + "." + extension))
    utils.setup_loggers(logname, repname, quiet=quiet, debug=debug)

    # Save command into sh file, if the command-line interface was used
    # TODO: use io_generator to save command
    if tedana_command is not None:
        command_file = open(os.path.join(out_dir, "tedana_call.sh"), "w")
        command_file.write(tedana_command)
        command_file.close()
    else:
        # Get variables passed to function if the tedana command is None
        variables = ", ".join(f"{name}={value}" for name, value in locals().items())
        # From variables, remove everything after ", tedana_command"
        variables = variables.split(", tedana_command")[0]
        tedana_command = f"tedana_workflow({variables})"

    LGR.info(f"Using output directory: {out_dir}")

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    tes = utils.check_te_values(tes)
    n_echos = len(tes)

    # Coerce gscontrol to list
    if not isinstance(gscontrol, list):
        gscontrol = [gscontrol]

    # Check value of tedpca *if* it is a predefined string,
    # a float in (0.0, 1.0) or an int >= 1
    tedpca = check_tedpca_value(tedpca, is_parser=False)

    # For z-catted files, make sure it's a list of size 1
    if isinstance(data, str):
        data = [data]

    LGR.info("Initializing and validating component selection tree")
    selector = ComponentSelector(tree, out_dir)

    LGR.info(f"Loading input data: {[f for f in data]}")
    data_cat, ref_img = io.load_data(data, n_echos=n_echos)

    # Load external regressors if provided
    # Decided to do the validation here so that, if there are issues, an error
    #  will be raised before PCA/ICA
    if (
        "external_regressor_config" in set(selector.tree.keys())
        and selector.tree["external_regressor_config"] is not None
    ):
        external_regressors, selector.tree["external_regressor_config"] = (
            metrics.external.load_validate_external_regressors(
                external_regressors, selector.tree["external_regressor_config"], data_cat.shape[2]
            )
        )

    io_generator = io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        overwrite=overwrite,
        verbose=verbose,
    )

    # Record inputs to OutputGenerator
    # TODO: turn this into an IOManager since this isn't really output
    io_generator.register_input(data)

    # Save system info to json
    info_dict = utils.get_system_version_info()
    info_dict["Command"] = tedana_command

    n_samp, n_echos, n_vols = data_cat.shape
    LGR.debug(f"Resulting data shape: {data_cat.shape}")

    # check if TR is 0
    img_t_r = io_generator.reference_img.header.get_zooms()[-1]
    if img_t_r == 0:
        raise OSError(
            "Dataset has a TR of 0. This indicates incorrect"
            " header information. To correct this, we recommend"
            " using this snippet:"
            "\n"
            "https://gist.github.com/jbteves/032c87aeb080dd8de8861cb151bff5d6"
            "\n"
            "to correct your TR to the value it should be."
        )

    if mixing_file is not None and op.isfile(mixing_file):
        mixing_file = op.abspath(mixing_file)
        # Allow users to re-run on same folder
        mixing_name_output = io_generator.get_name("ICA mixing tsv")
        mixing_file_new_path = op.join(io_generator.out_dir, op.basename(mixing_file))
        if op.basename(mixing_file) != op.basename(mixing_name_output) and not op.isfile(
            mixing_file_new_path
        ):
            shutil.copyfile(mixing_file, mixing_file_new_path)
        else:
            # Add "user_provided" to the mixing file's name if it's identical to the new file name
            # or if there's already a file in the output directory with the same name
            shutil.copyfile(
                mixing_file,
                op.join(io_generator.out_dir, f"user_provided_{op.basename(mixing_file)}"),
            )
    elif mixing_file is not None:
        raise OSError("Argument '--mix' must be an existing file.")

    if t2smap is not None and op.isfile(t2smap):
        t2smap_file = io_generator.get_name("t2star img")
        t2smap = op.abspath(t2smap)
        # Allow users to re-run on same folder
        if t2smap != t2smap_file:
            shutil.copyfile(t2smap, t2smap_file)
    elif t2smap is not None:
        raise OSError("Argument 't2smap' must be an existing file.")

    RepLGR.info(
        "TE-dependence analysis was performed on input data using the tedana workflow "
        "\\citep{dupre2021te}."
    )

    if mask and not t2smap:
        # TODO: add affine check
        LGR.info("Using user-defined mask")
        RepLGR.info("A user-defined mask was applied to the data.")
        mask = utils.reshape_niimg(mask).astype(int)
    elif t2smap and not mask:
        LGR.info("Assuming user=defined T2* map is masked and using it to generate mask")
        t2s_limited_sec = utils.reshape_niimg(t2smap)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        t2s_full = t2s_limited.copy()
        mask = (t2s_limited != 0).astype(int)
    elif t2smap and mask:
        LGR.info("Combining user-defined mask and T2* map to generate mask")
        t2s_limited_sec = utils.reshape_niimg(t2smap)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        t2s_full = t2s_limited.copy()
        mask = utils.reshape_niimg(mask).astype(int)
        mask[t2s_limited == 0] = 0  # reduce mask based on T2* map
    else:
        LGR.warning(
            "Computing EPI mask from first echo using nilearn's compute_epi_mask function. "
            "Most external pipelines include more reliable masking functions. "
            "It is strongly recommended to provide an external mask, "
            "and to visually confirm that mask accurately conforms to data boundaries."
        )
        first_echo_img = io.new_nii_like(io_generator.reference_img, data_cat[:, 0, :])
        mask = compute_epi_mask(first_echo_img).get_fdata()
        mask = utils.reshape_niimg(mask).astype(int)
        RepLGR.info(
            "An initial mask was generated from the first echo using "
            "nilearn's compute_epi_mask function."
        )

    # Create an adaptive mask with at least 1 good echo, for denoising
    mask_denoise, masksum_denoise = utils.make_adaptive_mask(
        data_cat,
        mask=mask,
        n_independent_echos=n_independent_echos,
        threshold=1,
        methods=masktype,
    )
    LGR.debug(f"Retaining {mask_denoise.sum()}/{n_samp} samples for denoising")
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
    LGR.debug(f"Retaining {mask_clf.sum()}/{n_samp} samples for classification")

    if t2smap is None:
        LGR.info("Computing T2* map")
        t2s_limited, s0_limited, t2s_full, s0_full = decay.fit_decay(
            data_cat, tes, mask_denoise, masksum_denoise, fittype
        )

        # set a hard cap for the T2* map
        # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
        cap_t2s = stats.scoreatpercentile(t2s_full.flatten(), 99.5, interpolation_method="lower")
        LGR.debug(f"Setting cap on T2* map at {utils.millisec2sec(cap_t2s):.5f}s")
        t2s_full[t2s_full > cap_t2s * 10] = cap_t2s
        io_generator.save_file(utils.millisec2sec(t2s_full), "t2star img")
        io_generator.save_file(s0_full, "s0 img")

        if verbose:
            io_generator.save_file(utils.millisec2sec(t2s_limited), "limited t2star img")
            io_generator.save_file(s0_limited, "limited s0 img")

        # Calculate RMSE if S0 and T2* are fit
        rmse_map, rmse_df = decay.rmse_of_fit_decay_ts(
            data=data_cat,
            tes=tes,
            adaptive_mask=masksum_denoise,
            t2s=t2s_limited,
            s0=s0_limited,
            fitmode="all",
        )
        io_generator.save_file(rmse_map, "rmse img")
        io_generator.add_df_to_file(rmse_df, "confounds tsv")

    # optimally combine data
    data_optcom = combine.make_optcom(
        data_cat,
        tes,
        masksum_denoise,
        t2s=t2s_full,
        combmode=combmode,
    )

    if "gsr" in gscontrol:
        # regress out global signal
        data_cat, data_optcom = gsc.gscontrol_raw(
            data_cat=data_cat,
            data_optcom=data_optcom,
            n_echos=n_echos,
            io_generator=io_generator,
        )

    fout = io_generator.save_file(data_optcom, "combined img")
    LGR.info(f"Writing optimally combined data set: {fout}")

    # Default r_ica results to None as they are expected for the reports
    cluster_labels = None
    similarity_t_sne = None
    fastica_convergence_warning_count = None

    if mixing_file is None:
        # Identify and remove thermal noise from data
        data_reduced, n_components = decomposition.tedpca(
            data_cat,
            data_optcom,
            mask_clf,
            masksum_clf,
            io_generator,
            tes=tes,
            n_independent_echos=n_independent_echos,
            algorithm=tedpca,
            kdaw=10.0,
            rdaw=1.0,
            low_mem=low_mem,
        )
        if verbose:
            io_generator.save_file(utils.unmask(data_reduced, mask_clf), "whitened img")

        # Perform ICA, calculate metrics, and apply decision tree
        # Restart when ICA fails to converge or too few BOLD components found
        keep_restarting = True
        n_restarts = 0
        seed = fixed_seed

        while keep_restarting:
            (
                mixing,
                seed,
                cluster_labels,
                similarity_t_sne,
                fastica_convergence_warning_count,
                index_quality,
            ) = decomposition.tedica(
                data_reduced,
                n_components,
                seed,
                ica_method,
                n_robust_runs,
                maxit,
                maxrestart=(maxrestart - n_restarts),
            )
            seed += 1
            n_restarts = seed - fixed_seed

            # Estimate betas and compute selection metrics for mixing matrix
            # generated from dimensionally reduced data using full data (i.e., data
            # with thermal noise)
            necessary_metrics = selector.necessary_metrics
            # The figures require some metrics that might not be used by the decision tree.
            extra_metrics = ["variance explained", "normalized variance explained", "kappa", "rho"]
            necessary_metrics = sorted(list(set(necessary_metrics + extra_metrics)))

            component_table, mixing = metrics.collect.generate_metrics(
                data_cat=data_cat,
                data_optcom=data_optcom,
                mixing=mixing,
                adaptive_mask=masksum_clf,
                tes=tes,
                n_independent_echos=n_independent_echos,
                io_generator=io_generator,
                label="ICA",
                metrics=necessary_metrics,
                external_regressors=external_regressors,
                external_regressor_config=selector.tree["external_regressor_config"],
            )
            LGR.info("Selecting components from ICA results")
            selector = selection.automatic_selection(
                component_table,
                selector,
                n_echos=n_echos,
                n_vols=n_vols,
                n_independent_echos=n_independent_echos,
            )
            n_likely_bold_comps = selector.n_likely_bold_comps_

            if n_likely_bold_comps == 0:
                if ica_method.lower() == "robustica":
                    LGR.warning("No BOLD components found with robustICA mixing matrix.")
                    keep_restarting = False
                elif n_restarts >= maxrestart:
                    LGR.warning(
                        "No BOLD components found, but maximum number of restarts reached."
                    )
                    keep_restarting = False
                else:
                    LGR.warning("No BOLD components found. Re-attempting ICA.")
                    # If we're going to restart, temporarily allow force overwrite
                    io_generator.overwrite = True
                    # Create a re-initialized selector object if rerunning
                    # Since external_regressor_config might have been expanded to remove
                    # regular expressions immediately after initialization,
                    # store and copy this key
                    tmp_external_regressor_config = selector.tree["external_regressor_config"]
                    selector = ComponentSelector(tree)
                    selector.tree["external_regressor_config"] = tmp_external_regressor_config
                    RepLGR.disabled = True  # Disable the report to avoid duplicate text
            else:
                keep_restarting = False

        RepLGR.disabled = False  # Re-enable the report after the while loop is escaped
        io_generator.overwrite = overwrite  # Re-enable original overwrite behavior
    else:
        LGR.info("Using supplied mixing matrix from ICA")
        mixing = pd.read_table(mixing_file).values

        # selector = ComponentSelector(tree)
        necessary_metrics = selector.necessary_metrics
        # The figures require some metrics that might not be used by the decision tree.
        extra_metrics = ["variance explained", "normalized variance explained", "kappa", "rho"]
        necessary_metrics = sorted(list(set(necessary_metrics + extra_metrics)))

        component_table, mixing = metrics.collect.generate_metrics(
            data_cat=data_cat,
            data_optcom=data_optcom,
            mixing=mixing,
            adaptive_mask=masksum_clf,
            tes=tes,
            n_independent_echos=n_independent_echos,
            io_generator=io_generator,
            label="ICA",
            metrics=necessary_metrics,
            external_regressors=external_regressors,
            external_regressor_config=selector.tree["external_regressor_config"],
        )
        selector = selection.automatic_selection(
            component_table,
            selector,
            n_echos=n_echos,
            n_vols=n_vols,
            n_independent_echos=n_independent_echos,
        )

        if selector.n_likely_bold_comps_ == 0:
            LGR.warning("No BOLD components found with user-provided ICA mixing matrix.")

    if ica_method.lower() == "robustica":
        # If robustica was used, store number of iterations where ICA failed
        selector.cross_component_metrics_["fastica_convergence_warning_count"] = (
            fastica_convergence_warning_count
        )
        selector.cross_component_metrics_["robustica_mean_index_quality"] = index_quality

    # TODO The ICA mixing matrix should be written out after it is created
    #     It is currently being written after component selection is done
    #     and rewritten if an existing mixing matrix is given as an input
    comp_names = component_table["Component"].values
    mixing_df = pd.DataFrame(data=mixing, columns=comp_names)
    io_generator.save_file(mixing_df, "ICA mixing tsv")

    betas_oc = utils.unmask(computefeats2(data_optcom, mixing, mask_denoise), mask_denoise)
    io_generator.save_file(betas_oc, "z-scored ICA components img")

    # calculate the fit of rejected to accepted components to use as a quality measure
    # Note: This adds a column to component_table & needs to run before the table is saved
    reporting.quality_metrics.calculate_rejected_components_impact(selector, mixing)

    # Save component selector and tree
    selector.to_files(io_generator)
    # Save metrics and metadata
    metric_metadata = metrics.collect.get_metadata(component_table)
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
    io_generator.save_file(decomp_metadata, "ICA decomposition json")

    if selector.n_likely_bold_comps_ == 0:
        LGR.warning("No BOLD components detected! Please check data and results!")

    # TODO: un-hack separate component_table
    component_table = selector.component_table_

    mixing_orig = mixing.copy()
    if tedort:
        comps_accepted = selector.accepted_comps_
        comps_rejected = selector.rejected_comps_
        acc_ts = mixing[:, comps_accepted]
        rej_ts = mixing[:, comps_rejected]
        betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
        pred_rej_ts = np.dot(acc_ts, betas)
        resid = rej_ts - pred_rej_ts
        mixing[:, comps_rejected] = resid
        comp_names = [
            io.add_decomp_prefix(comp, prefix="ICA", max_value=component_table.index.max())
            for comp in range(selector.n_comps_)
        ]

        mixing_df = pd.DataFrame(data=mixing, columns=comp_names)
        io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")
        RepLGR.info(
            "Rejected components' time series were then "
            "orthogonalized with respect to accepted components' time "
            "series."
        )

    io.writeresults(
        data_optcom,
        mask=mask_denoise,
        component_table=component_table,
        mixing=mixing,
        io_generator=io_generator,
    )

    if "mir" in gscontrol:
        gsc.minimum_image_regression(
            data_optcom=data_optcom,
            mixing=mixing,
            mask=mask_denoise,
            component_table=component_table,
            classification_tags=selector.classification_tags,
            io_generator=io_generator,
        )

    if verbose:
        io.writeresults_echoes(data_cat, mixing, mask_denoise, component_table, io_generator)

    # Write out registry of outputs
    io_generator.save_self()

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
                "Node": {
                    "Name": info_dict["Node"],
                    "System": info_dict["System"],
                    "Machine": info_dict["Machine"],
                    "Processor": info_dict["Processor"],
                    "Release": info_dict["Release"],
                    "Version": info_dict["Version"],
                },
                "Python": info_dict["Python"],
                "Python_Libraries": info_dict["Python_Libraries"],
                "Command": info_dict["Command"],
            }
        ],
    }
    with open(io_generator.get_name("data description json"), "w") as fo:
        json.dump(derivative_metadata, fo, sort_keys=True, indent=4)

    RepLGR.info(
        "\n\nThis workflow used numpy \\citep{van2011numpy}, scipy \\citep{virtanen2020scipy}, "
        "pandas \\citep{mckinney2010data,reback2020pandas}, "
        "scikit-learn \\citep{pedregosa2011scikit}, "
        "nilearn, bokeh \\citep{bokehmanual}, matplotlib \\citep{Hunter2007}, "
        "and nibabel \\citep{brett_matthew_2019_3233118}."
    )

    RepLGR.info(
        "This workflow also used the Dice similarity index "
        "\\citep{dice1945measures,sorensen1948method}."
    )

    with open(repname) as fo:
        report = [line.rstrip() for line in fo.readlines()]
        report = " ".join(report)
        # Double-spaces reflect new paragraphs
        report = report.replace("  ", "\n\n")

    with open(repname, "w") as fo:
        fo.write(report)

    # Collect BibTeX entries for cited papers
    references = get_description_references(report)

    with open(bibtex_file, "w") as fo:
        fo.write(references)

    if not no_reports:
        LGR.info("Making figures folder with static component maps and timecourse plots.")

        data_denoised, data_accepted, data_rejected = io.denoise_ts(
            data_optcom,
            mixing,
            mask_denoise,
            component_table,
        )

        reporting.static_figures.plot_adaptive_mask(
            optcom=data_optcom,
            base_mask=mask,
            io_generator=io_generator,
        )
        reporting.static_figures.carpet_plot(
            optcom_ts=data_optcom,
            denoised_ts=data_denoised,
            hikts=data_accepted,
            lowkts=data_rejected,
            mask=mask_denoise,
            io_generator=io_generator,
            gscontrol=gscontrol,
        )
        reporting.static_figures.comp_figures(
            data_optcom,
            mask=mask_denoise,
            component_table=component_table,
            mixing=mixing_orig,
            io_generator=io_generator,
            png_cmap=png_cmap,
        )
        reporting.static_figures.plot_t2star_and_s0(io_generator=io_generator, mask=mask_denoise)
        if t2smap is None:
            reporting.static_figures.plot_rmse(
                io_generator=io_generator,
                adaptive_mask=masksum_denoise,
            )

        LGR.info("Generating dynamic report")
        reporting.generate_report(io_generator, cluster_labels, similarity_t_sne)

    LGR.info("Workflow completed")

    # Add newsletter info to the log
    utils.log_newsletter_info()

    utils.teardown_loggers()


def _main(argv=None):
    """Run the tedana workflow."""
    if argv:
        # relevant for tests when CLI called with tedana_cli._main(args)
        tedana_command = "tedana " + " ".join(argv)
    else:
        tedana_command = "tedana " + " ".join(sys.argv[1:])
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    n_threads = kwargs.pop("n_threads")
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        tedana_workflow(**kwargs, tedana_command=tedana_command)


if __name__ == "__main__":
    _main()
