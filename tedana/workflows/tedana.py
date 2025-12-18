"""Run the "canonical" TE-Dependent ANAlysis workflow."""

import argparse
import logging
import sys
from typing import Any, List, Optional, Union

from threadpoolctl import threadpool_limits

from tedana.config import (
    DEFAULT_ICA_METHOD,
    DEFAULT_N_MAX_ITER,
    DEFAULT_N_MAX_RESTART,
    DEFAULT_N_ROBUST_RUNS,
    DEFAULT_SEED,
)
from tedana.workflows.parser_utils import (
    check_n_robust_runs_value,
    check_tedpca_value,
    is_valid_file,
)
from tedana.workflows.pipeline_context import PipelineContext, create_context_from_args
from tedana.workflows.pipeline_stages import run_tedana_pipeline

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
    tedana_command: Optional[str] = None,
) -> PipelineContext:
    """Run the "canonical" TE-Dependent ANAlysis workflow.

    Please remember to cite :footcite:t:`dupre2021te`.

    This function provides a high-level interface to the tedana pipeline.
    The workflow is organized into modular stages for better maintainability
    and memory management.

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
    tedana_command : :obj:`str`, optional
        If the command-line interface was used, this is the command that was
        run. Default is None.

    Returns
    -------
    ctx : PipelineContext
        The pipeline context containing all workflow state and results.
        This provides access to the component table, mixing matrix,
        selector, and other outputs.

    Notes
    -----
    This workflow writes out several files. For a complete list of the files
    generated by this workflow, please visit
    https://tedana.readthedocs.io/en/latest/outputs.html

    The workflow is organized into the following stages:

    1. **Setup**: Initialize output directory, logging, and validate inputs
    2. **Data Loading**: Load multi-echo data and external regressors
    3. **Masking**: Create adaptive masks for denoising and classification
    4. **T2*/S0 Estimation**: Fit decay model to estimate T2* and S0 maps
    5. **Optimal Combination**: Combine echoes using T2*-weighted averaging
    6. **Decomposition**: PCA for dimensionality reduction, ICA for source separation
    7. **Component Selection**: Classify components using decision tree
    8. **Output Generation**: Save denoised data, component maps, and metrics
    9. **Reporting**: Generate HTML report and static figures

    References
    ----------
    .. footbibliography::

    Examples
    --------
    Basic usage with three echoes:

    >>> ctx = tedana_workflow(
    ...     data=['echo1.nii.gz', 'echo2.nii.gz', 'echo3.nii.gz'],
    ...     tes=[14.5, 29.0, 43.5],
    ...     out_dir='tedana_output'
    ... )

    Access results from the returned context:

    >>> component_table = ctx.component_table
    >>> mixing_matrix = ctx.mixing
    >>> n_accepted = ctx.selector.n_accepted_comps_
    """
    # Set default masktype
    if masktype is None:
        masktype = ["dropout"]

    # Create pipeline context from arguments
    ctx = create_context_from_args(
        data=data,
        tes=tes,
        out_dir=out_dir,
        mask=mask,
        convention=convention,
        prefix=prefix,
        dummy_scans=dummy_scans,
        masktype=masktype,
        fittype=fittype,
        combmode=combmode,
        n_independent_echos=n_independent_echos,
        tree=tree,
        external_regressors=external_regressors,
        ica_method=ica_method,
        n_robust_runs=n_robust_runs,
        tedpca=tedpca,
        fixed_seed=fixed_seed,
        maxit=maxit,
        maxrestart=maxrestart,
        tedort=tedort,
        gscontrol=gscontrol,
        no_reports=no_reports,
        png_cmap=png_cmap,
        verbose=verbose,
        low_mem=low_mem,
        debug=debug,
        quiet=quiet,
        overwrite=overwrite,
        t2smap=t2smap,
        mixing_file=mixing_file,
        tedana_command=tedana_command,
    )

    # Run the complete pipeline
    run_tedana_pipeline(ctx)

    return ctx


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
