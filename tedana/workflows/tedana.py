"""Run the "canonical" TE-Dependent ANAlysis workflow."""

import argparse
import logging
import os
import os.path as op
import shutil
import sys
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from threadpoolctl import threadpool_limits

from tedana import decomposition, io, metrics, reporting, selection, utils
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
from tedana.workflows.shared import (
    apply_mir,
    apply_tedort,
    compute_optimal_combination,
    create_adaptive_masks,
    finalize_report_text,
    fit_decay_model,
    generate_dynamic_report,
    generate_static_figures,
    load_multiecho_data,
    rename_previous_reports,
    save_derivative_metadata,
    save_workflow_command,
    setup_logging,
    teardown_workflow,
    validate_tr,
    write_denoised_results,
    write_echo_results,
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
        "--dummy-scans",
        dest="dummy_scans",
        type=int,
        help="Number of dummy scans to remove from the beginning of the data.",
        default=0,
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
    data: Union[str, List[str]],
    tes: List[float],
    out_dir: str = ".",
    mask: Optional[str] = None,
    convention: str = "bids",
    prefix: str = "",
    dummy_scans: int = 0,
    masktype: Optional[List[str]] = None,
    fittype: str = "loglin",
    combmode: str = "t2s",
    n_independent_echos: Optional[int] = None,
    tree: str = "tedana_orig",
    external_regressors: Optional[str] = None,
    ica_method: str = DEFAULT_ICA_METHOD,
    n_robust_runs: int = DEFAULT_N_ROBUST_RUNS,
    tedpca: Any = "aic",
    fixed_seed: int = DEFAULT_SEED,
    maxit: int = DEFAULT_N_MAX_ITER,
    maxrestart: int = DEFAULT_N_MAX_RESTART,
    tedort: bool = False,
    gscontrol: Optional[Union[str, List[str]]] = None,
    no_reports: bool = False,
    png_cmap: str = "coolwarm",
    verbose: bool = False,
    low_mem: bool = False,
    debug: bool = False,
    quiet: bool = False,
    overwrite: bool = False,
    t2smap: Optional[str] = None,
    mixing_file: Optional[str] = None,
    n_threads: int = 1,
    tedana_command: Optional[str] = None,
) -> Dict[str, Any]:
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
    dummy_scans : :obj:`int`, optional
        Number of dummy scans to remove from the beginning of the data
        (both in the BOLD data and in any confounds).
        Default is 0.
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
    n_threads : :obj:`int`, optional
        Number of threads to use. Used by threadpoolctl to set the parameter
        outside of the workflow function, as well as the number of threads to use
        for the decay model fitting. Default is 1.
    tedana_command : :obj:`str`, optional
        If the command-line interface was used, this is the command that was
        run. Default is None.

    Returns
    -------
    results : dict
        Dictionary containing workflow results including:
        - 'component_table': Component metrics table
        - 'selector': Component selector with classification results
        - 'mixing': Mixing matrix
        - 'n_accepted': Number of accepted components

    Notes
    -----
    This workflow writes out several files. For a complete list of the files
    generated by this workflow, please visit
    https://tedana.readthedocs.io/en/latest/outputs.html

    References
    ----------
    .. footbibliography::


    Examples
    --------
    Basic usage with three echoes:

    >>> results = tedana_workflow(
    ...     data=['echo1.nii.gz', 'echo2.nii.gz', 'echo3.nii.gz'],
    ...     tes=[14.5, 29.0, 43.5],
    ...     out_dir='tedana_output'
    ... )

    Access results from the returned dictionary:

    >>> component_table = results['component_table']
    >>> mixing_matrix = results['mixing']
    >>> n_accepted = results['n_accepted']
    """
    # ===========================================================================
    # Stage 1: Setup and Initialization
    # ===========================================================================
    # Set defaults
    if masktype is None:
        masktype = ["dropout"]

    if gscontrol is None:
        gscontrol = []
    elif not isinstance(gscontrol, list):
        gscontrol = [gscontrol]

    # Ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    tes = utils.check_te_values(tes)
    n_echos = len(tes)

    # Check tedpca value
    tedpca = check_tedpca_value(tedpca, is_parser=False)

    # For z-catted files, make sure data is a list
    if isinstance(data, str):
        data = [data]

    # Setup output directory
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # Setup boilerplate
    prefix = io._infer_prefix(prefix)
    repname, bibtex_file = rename_previous_reports(out_dir, prefix)
    setup_logging(out_dir, repname, quiet, debug)

    # Save command
    if tedana_command is not None:
        save_workflow_command(out_dir, tedana_command, "tedana_call.sh")
    else:
        # Generate command from arguments
        variables = ", ".join(f"{name}={value}" for name, value in locals().items())
        variables = variables.split(", tedana_command")[0]
        tedana_command = f"tedana_workflow({variables}, ...)"

    # Save system info
    info_dict = utils.get_system_version_info()
    info_dict["Command"] = tedana_command

    LGR.info(f"Using output directory: {out_dir}")

    # Initialize component selector
    LGR.info("Initializing and validating component selection tree")
    selector = ComponentSelector(tree, out_dir)

    # ===========================================================================
    # Stage 2: Data Loading
    # ===========================================================================
    LGR.info(f"Loading input data: {[f for f in data]}")
    me_data = load_multiecho_data(data, tes, dummy_scans)
    n_vols = me_data.n_vols

    # Setup IO generator
    io_generator = io.OutputGenerator(
        me_data.ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        overwrite=overwrite,
        verbose=verbose,
    )
    io_generator.register_input(data)

    # Validate TR
    validate_tr(me_data.ref_img)

    # Load external regressors if provided
    external_regressors_data = None
    if (
        "external_regressor_config" in set(selector.tree.keys())
        and selector.tree["external_regressor_config"] is not None
    ):
        external_regressors_data, selector.tree["external_regressor_config"] = (
            metrics.external.load_validate_external_regressors(
                external_regressors=external_regressors,
                external_regressor_config=selector.tree["external_regressor_config"],
                n_vols=me_data.data_cat.shape[2],
                dummy_scans=dummy_scans,
            )
        )

    # Handle pre-computed mixing file
    if mixing_file is not None and op.isfile(mixing_file):
        mixing_file = op.abspath(mixing_file)
        mixing_name_output = io_generator.get_name("ICA mixing tsv")
        mixing_file_new_path = op.join(io_generator.out_dir, op.basename(mixing_file))
        if op.basename(mixing_file) != op.basename(mixing_name_output) and not op.isfile(
            mixing_file_new_path
        ):
            shutil.copyfile(mixing_file, mixing_file_new_path)
        else:
            shutil.copyfile(
                mixing_file,
                op.join(io_generator.out_dir, f"user_provided_{op.basename(mixing_file)}"),
            )
    elif mixing_file is not None:
        raise OSError("Argument '--mix' must be an existing file.")

    # Handle pre-computed T2* map
    t2smap_provided = t2smap is not None
    if t2smap is not None and op.isfile(t2smap):
        t2smap_output = io_generator.get_name("t2star img")
        t2smap = op.abspath(t2smap)
        if t2smap != t2smap_output:
            shutil.copyfile(t2smap, t2smap_output)
    elif t2smap is not None:
        raise OSError("Argument 't2smap' must be an existing file.")

    RepLGR.info(
        "TE-dependence analysis was performed on input data using the tedana workflow "
        "\\citep{dupre2021te}."
    )

    # ===========================================================================
    # Stage 3: Masking
    # ===========================================================================
    masks = create_adaptive_masks(
        me_data,
        mask_file=mask,
        masktype=masktype,
        io_generator=io_generator,
        t2smap_file=t2smap,
        n_independent_echos=n_independent_echos,
    )

    # ===========================================================================
    # Stage 4: T2*/S0 Estimation
    # ===========================================================================
    if not t2smap_provided:
        decay_maps = fit_decay_model(
            me_data,
            masks,
            fittype,
            io_generator,
            verbose=verbose,
            n_threads=n_threads,
        )
        t2s_full = decay_maps.t2s_full
    else:
        # T2* map was provided, load it
        t2s_limited_sec = utils.reshape_niimg(t2smap)
        t2s_full = utils.sec2millisec(t2s_limited_sec)

    # ===========================================================================
    # Stage 5: Optimal Combination
    # ===========================================================================
    optcom = compute_optimal_combination(
        me_data,
        masks,
        type("DecayMaps", (), {"t2s_full": t2s_full})(),  # Simple object with t2s_full
        combmode,
        io_generator,
        gscontrol,
    )
    data_optcom = optcom.data_optcom

    # ===========================================================================
    # Stage 6: Decomposition (PCA + ICA)
    # ===========================================================================
    cluster_labels = None
    similarity_t_sne = None
    fastica_convergence_warning_count = None

    if mixing_file is None:
        # PCA decomposition
        data_reduced, n_components = decomposition.tedpca(
            me_data.data_cat,
            data_optcom,
            masks.mask_clf,
            masks.masksum_clf,
            io_generator,
            tes=tes,
            n_independent_echos=n_independent_echos,
            algorithm=tedpca,
            kdaw=10.0,
            rdaw=1.0,
            low_mem=low_mem,
        )

        if verbose:
            io_generator.save_file(utils.unmask(data_reduced, masks.mask_clf), "whitened img")

        # ICA with restart logic
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
                n_threads=n_threads,
            )
            seed += 1
            n_restarts = seed - fixed_seed

            # Compute component metrics
            necessary_metrics = selector.necessary_metrics
            extra_metrics = ["variance explained", "normalized variance explained", "kappa", "rho"]
            necessary_metrics = sorted(list(set(necessary_metrics + extra_metrics)))

            component_table, mixing = metrics.collect.generate_metrics(
                data_cat=me_data.data_cat,
                data_optcom=data_optcom,
                mixing=mixing,
                adaptive_mask=masks.masksum_clf,
                tes=tes,
                n_independent_echos=n_independent_echos,
                io_generator=io_generator,
                label="ICA",
                metrics=necessary_metrics,
                external_regressors=external_regressors_data,
                external_regressor_config=selector.tree["external_regressor_config"],
            )

            # Perform component selection
            LGR.info("Selecting components from ICA results")
            selector = selection.automatic_selection(
                component_table,
                selector,
                n_echos=n_echos,
                n_vols=n_vols,
                n_independent_echos=n_independent_echos,
            )
            n_likely_bold = selector.n_likely_bold_comps_

            if n_likely_bold == 0:
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
                    io_generator.overwrite = True
                    # Re-initialize selector
                    tmp_ext_config = selector.tree["external_regressor_config"]
                    selector = ComponentSelector(tree)
                    selector.tree["external_regressor_config"] = tmp_ext_config
                    RepLGR.disabled = True
            else:
                keep_restarting = False

        RepLGR.disabled = False
        io_generator.overwrite = overwrite

        # Store robustica metrics
        if ica_method.lower() == "robustica":
            if selector.cross_component_metrics_ is None:
                selector.cross_component_metrics_ = {}
            selector.cross_component_metrics_["fastica_convergence_warning_count"] = (
                fastica_convergence_warning_count
            )
            selector.cross_component_metrics_["robustica_mean_index_quality"] = index_quality

    else:
        # Use supplied mixing matrix
        LGR.info("Using supplied mixing matrix from ICA")
        mixing = pd.read_table(mixing_file).values

        # Compute metrics
        necessary_metrics = selector.necessary_metrics
        extra_metrics = ["variance explained", "normalized variance explained", "kappa", "rho"]
        necessary_metrics = sorted(list(set(necessary_metrics + extra_metrics)))

        component_table, mixing = metrics.collect.generate_metrics(
            data_cat=me_data.data_cat,
            data_optcom=data_optcom,
            mixing=mixing,
            adaptive_mask=masks.masksum_clf,
            tes=tes,
            n_independent_echos=n_independent_echos,
            io_generator=io_generator,
            label="ICA",
            metrics=necessary_metrics,
            external_regressors=external_regressors_data,
            external_regressor_config=selector.tree["external_regressor_config"],
        )

        # Perform component selection
        LGR.info("Selecting components from ICA results")
        selector = selection.automatic_selection(
            component_table,
            selector,
            n_echos=n_echos,
            n_vols=n_vols,
            n_independent_echos=n_independent_echos,
        )

        if selector.n_likely_bold_comps_ == 0:
            LGR.warning("No BOLD components found with user-provided ICA mixing matrix.")

    # ===========================================================================
    # Stage 7: Output Generation
    # ===========================================================================
    component_table = selector.component_table_

    # Save mixing matrix
    comp_names = component_table["Component"].values
    mixing_df = pd.DataFrame(data=mixing, columns=comp_names)
    io_generator.save_file(mixing_df, "ICA mixing tsv")

    # Save z-scored component maps
    betas_oc = utils.unmask(
        computefeats2(data_optcom, mixing, masks.mask_denoise), masks.mask_denoise
    )
    io_generator.save_file(betas_oc, "z-scored ICA components img")

    # Calculate rejected component impact
    reporting.quality_metrics.calculate_rejected_components_impact(selector, mixing)

    # Save selector and metrics
    selector.to_files(io_generator)
    metric_metadata = metrics.collect.get_metadata(component_table)
    io_generator.save_file(metric_metadata, "ICA metrics json")

    # Save decomposition metadata
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

    # Apply tedort if requested
    mixing_orig = mixing.copy()
    if tedort:
        mixing = apply_tedort(mixing, selector.accepted_comps_, selector.rejected_comps_)
        comp_names = [
            io.add_decomp_prefix(comp, prefix="ICA", max_value=component_table.index.max())
            for comp in range(selector.n_comps_)
        ]
        mixing_df = pd.DataFrame(data=mixing, columns=comp_names)
        io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")

    # Write denoised results
    write_denoised_results(
        data_optcom,
        mask=masks.mask_denoise,
        component_table=component_table,
        mixing=mixing,
        io_generator=io_generator,
    )

    # Apply MIR if requested
    if "mir" in gscontrol:
        apply_mir(
            data_optcom=data_optcom,
            mixing=mixing,
            mask=masks.mask_denoise,
            component_table=component_table,
            classification_tags=selector.classification_tags,
            io_generator=io_generator,
        )

    # Write per-echo results if verbose
    if verbose:
        write_echo_results(
            me_data.data_cat, mixing, masks.mask_denoise, component_table, io_generator
        )

    # Save registry and metadata
    io_generator.save_self()
    save_derivative_metadata(
        io_generator,
        info_dict,
        workflow_name="tedana",
        workflow_description=(
            "A denoising pipeline for the identification and removal "
            "of non-BOLD noise from multi-echo fMRI data."
        ),
    )

    # Finalize report text
    finalize_report_text(repname, bibtex_file)

    # ===========================================================================
    # Stage 8: Report Generation
    # ===========================================================================
    if not no_reports:
        generate_static_figures(
            data_optcom=data_optcom,
            mask_denoise=masks.mask_denoise,
            base_mask=masks.base_mask,
            component_table=component_table,
            mixing=mixing_orig,
            io_generator=io_generator,
            png_cmap=png_cmap,
            gscontrol=gscontrol,
            masksum_denoise=masks.masksum_denoise,
            external_regressors=external_regressors_data,
            t2smap_provided=t2smap_provided,
        )
        generate_dynamic_report(io_generator, cluster_labels, similarity_t_sne)

    # ===========================================================================
    # Stage 9: Cleanup
    # ===========================================================================
    LGR.info("Workflow completed")
    teardown_workflow()

    return {
        "component_table": component_table,
        "selector": selector,
        "mixing": mixing,
        "n_accepted": selector.n_accepted_comps_,
    }


def _main(argv=None):
    """Run the tedana workflow."""
    if argv:
        # relevant for tests when CLI called with tedana_cli._main(args)
        tedana_command = "tedana " + " ".join(argv)
    else:
        tedana_command = "tedana " + " ".join(sys.argv[1:])
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    n_threads = kwargs.get("n_threads", 1)
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        tedana_workflow(**kwargs, tedana_command=tedana_command)


if __name__ == "__main__":
    _main()
