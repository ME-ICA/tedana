"""Pipeline stages for the tedana workflow.

This module provides modular functions for each stage of the tedana workflow.
Each stage function takes a PipelineContext and modifies it in place,
enabling cleaner code organization and explicit memory management.
"""

import datetime
import json
import logging
import os
import os.path as op
import shutil
from glob import glob

import numpy as np
import pandas as pd
from nilearn.masking import compute_epi_mask
from scipy import stats

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
from tedana.selection.component_selector import ComponentSelector
from tedana.stats import computefeats2
from tedana.workflows.parser_utils import check_tedpca_value
from tedana.workflows.pipeline_context import PipelineContext

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


# =============================================================================
# Stage 1: Setup and Initialization
# =============================================================================


def setup_output_directory(ctx: PipelineContext) -> None:
    """Set up the output directory and initialize logging.

    Creates the output directory if it doesn't exist, sets up log files,
    and renames any previous report files.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    ctx.out_dir = op.abspath(ctx.out_dir)
    if not op.isdir(ctx.out_dir):
        os.mkdir(ctx.out_dir)

    # Set up boilerplate file naming
    prefix = io._infer_prefix(ctx.prefix)
    ctx.prefix = prefix
    basename = f"{prefix}report"
    extension = "txt"
    ctx.repname = op.join(ctx.out_dir, f"{basename}.{extension}")
    ctx.bibtex_file = op.join(ctx.out_dir, f"{prefix}references.bib")

    # Rename any previous report files
    repex = op.join(ctx.out_dir, f"{basename}*")
    previousreps = glob(repex)
    previousreps.sort(reverse=True)
    for f in previousreps:
        previousparts = op.splitext(f)
        newname = previousparts[0] + "_old" + previousparts[1]
        os.rename(f, newname)

    # Create logfile
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(ctx.out_dir, f"tedana_{start_time}.tsv")
    utils.setup_loggers(logname, ctx.repname, quiet=ctx.quiet, debug=ctx.debug)

    # Save command into sh file
    if ctx.tedana_command is not None:
        command_file = open(os.path.join(ctx.out_dir, "tedana_call.sh"), "w")
        command_file.write(ctx.tedana_command)
        command_file.close()
    else:
        # Generate command from context attributes
        variables = ", ".join(
            f"{name}={getattr(ctx, name)}"
            for name in [
                "data",
                "tes",
                "out_dir",
                "mask_file",
                "convention",
                "prefix",
                "dummy_scans",
                "masktype",
                "fittype",
                "combmode",
                "tree",
                "tedpca",
                "fixed_seed",
            ]
        )
        ctx.tedana_command = f"tedana_workflow({variables}, ...)"

    LGR.info(f"Using output directory: {ctx.out_dir}")


def validate_inputs(ctx: PipelineContext) -> None:
    """Validate and normalize input parameters.

    Checks echo times, validates tedpca value, and ensures data is in list format.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to validate and modify.
    """
    # Ensure tes are in appropriate format
    ctx.tes = [float(te) for te in ctx.tes]
    ctx.tes = utils.check_te_values(ctx.tes)
    ctx.n_echos = len(ctx.tes)

    # Check tedpca value
    ctx.tedpca = check_tedpca_value(ctx.tedpca, is_parser=False)

    # For z-catted files, make sure data is a list
    if isinstance(ctx.data, str):
        ctx.data = [ctx.data]


def initialize_component_selector(ctx: PipelineContext) -> None:
    """Initialize and validate the component selection tree.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    LGR.info("Initializing and validating component selection tree")
    ctx.selector = ComponentSelector(ctx.tree, ctx.out_dir)


def setup_io_generator(ctx: PipelineContext) -> None:
    """Set up the output generator and system info.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    ctx.io_generator = io.OutputGenerator(
        ctx.ref_img,
        convention=ctx.convention,
        out_dir=ctx.out_dir,
        prefix=ctx.prefix,
        config="auto",
        overwrite=ctx.overwrite,
        verbose=ctx.verbose,
    )

    # Register inputs
    ctx.io_generator.register_input(ctx.data)

    # Save system info
    ctx.info_dict = utils.get_system_version_info()
    ctx.info_dict["Command"] = ctx.tedana_command


# =============================================================================
# Stage 2: Data Loading
# =============================================================================


def load_data(ctx: PipelineContext) -> None:
    """Load multi-echo data from input files.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    LGR.info(f"Loading input data: {[f for f in ctx.data]}")
    ctx.data_cat, ctx.ref_img = io.load_data(
        ctx.data, n_echos=ctx.n_echos, dummy_scans=ctx.dummy_scans
    )

    ctx.n_samp, ctx.n_echos, ctx.n_vols = ctx.data_cat.shape
    LGR.debug(f"Resulting data shape: {ctx.data_cat.shape}")


def validate_tr(ctx: PipelineContext) -> None:
    """Validate that TR is non-zero.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to validate.

    Raises
    ------
    OSError
        If TR is 0.
    """
    ctx.img_t_r = ctx.io_generator.reference_img.header.get_zooms()[-1]
    if ctx.img_t_r == 0:
        raise OSError(
            "Dataset has a TR of 0. This indicates incorrect"
            " header information. To correct this, we recommend"
            " using this snippet:"
            "\n"
            "https://gist.github.com/jbteves/032c87aeb080dd8de8861cb151bff5d6"
            "\n"
            "to correct your TR to the value it should be."
        )


def load_external_regressors(ctx: PipelineContext) -> None:
    """Load and validate external regressors if provided.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    if (
        "external_regressor_config" in set(ctx.selector.tree.keys())
        and ctx.selector.tree["external_regressor_config"] is not None
    ):
        ctx.external_regressors, ctx.selector.tree["external_regressor_config"] = (
            metrics.external.load_validate_external_regressors(
                external_regressors=ctx.external_regressors_file,
                external_regressor_config=ctx.selector.tree["external_regressor_config"],
                n_vols=ctx.data_cat.shape[2],
                dummy_scans=ctx.dummy_scans,
            )
        )


def handle_precomputed_files(ctx: PipelineContext) -> None:
    """Handle pre-computed mixing matrix and T2* map files.

    Copies the files to the output directory if provided.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.

    Raises
    ------
    OSError
        If provided files do not exist.
    """
    # Handle mixing file
    if ctx.mixing_file is not None and op.isfile(ctx.mixing_file):
        ctx.mixing_file = op.abspath(ctx.mixing_file)
        mixing_name_output = ctx.io_generator.get_name("ICA mixing tsv")
        mixing_file_new_path = op.join(ctx.io_generator.out_dir, op.basename(ctx.mixing_file))
        if op.basename(ctx.mixing_file) != op.basename(mixing_name_output) and not op.isfile(
            mixing_file_new_path
        ):
            shutil.copyfile(ctx.mixing_file, mixing_file_new_path)
        else:
            shutil.copyfile(
                ctx.mixing_file,
                op.join(ctx.io_generator.out_dir, f"user_provided_{op.basename(ctx.mixing_file)}"),
            )
    elif ctx.mixing_file is not None:
        raise OSError("Argument '--mix' must be an existing file.")

    # Handle T2* map file
    if ctx.t2smap_file is not None and op.isfile(ctx.t2smap_file):
        t2smap_output = ctx.io_generator.get_name("t2star img")
        ctx.t2smap_file = op.abspath(ctx.t2smap_file)
        if ctx.t2smap_file != t2smap_output:
            shutil.copyfile(ctx.t2smap_file, t2smap_output)
    elif ctx.t2smap_file is not None:
        raise OSError("Argument 't2smap' must be an existing file.")

    RepLGR.info(
        "TE-dependence analysis was performed on input data using the tedana workflow "
        "\\citep{dupre2021te}."
    )


# =============================================================================
# Stage 3: Masking
# =============================================================================


def create_masks(ctx: PipelineContext) -> None:
    """Create masks for denoising and classification.

    Creates both a liberal mask (for denoising) and a conservative mask
    (for classification), along with adaptive masks that account for
    signal dropout across echoes.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    if ctx.mask_file and not ctx.t2smap_file:
        LGR.info("Using user-defined mask")
        RepLGR.info("A user-defined mask was applied to the data.")
        ctx.mask = utils.reshape_niimg(ctx.mask_file).astype(int)
    elif ctx.t2smap_file and not ctx.mask_file:
        LGR.info("Assuming user-defined T2* map is masked and using it to generate mask")
        t2s_limited_sec = utils.reshape_niimg(ctx.t2smap_file)
        ctx.t2s_limited = utils.sec2millisec(t2s_limited_sec)
        ctx.t2s_full = ctx.t2s_limited.copy()
        ctx.mask = (ctx.t2s_limited != 0).astype(int)
    elif ctx.t2smap_file and ctx.mask_file:
        LGR.info("Combining user-defined mask and T2* map to generate mask")
        t2s_limited_sec = utils.reshape_niimg(ctx.t2smap_file)
        ctx.t2s_limited = utils.sec2millisec(t2s_limited_sec)
        ctx.t2s_full = ctx.t2s_limited.copy()
        ctx.mask = utils.reshape_niimg(ctx.mask_file).astype(int)
        ctx.mask[ctx.t2s_limited == 0] = 0
    else:
        LGR.warning(
            "Computing EPI mask from first echo using nilearn's compute_epi_mask function. "
            "Most external pipelines include more reliable masking functions. "
            "It is strongly recommended to provide an external mask, "
            "and to visually confirm that mask accurately conforms to data boundaries."
        )
        first_echo_img = io.new_nii_like(ctx.io_generator.reference_img, ctx.data_cat[:, 0, :])
        ctx.mask = compute_epi_mask(first_echo_img).get_fdata()
        ctx.mask = utils.reshape_niimg(ctx.mask).astype(int)
        RepLGR.info(
            "An initial mask was generated from the first echo using "
            "nilearn's compute_epi_mask function."
        )

    # Create adaptive mask with at least 1 good echo, for denoising
    ctx.mask_denoise, ctx.masksum_denoise = utils.make_adaptive_mask(
        ctx.data_cat,
        mask=ctx.mask,
        n_independent_echos=ctx.n_independent_echos,
        threshold=1,
        methods=ctx.masktype,
    )
    LGR.debug(f"Retaining {ctx.mask_denoise.sum()}/{ctx.n_samp} samples for denoising")
    ctx.io_generator.save_file(ctx.masksum_denoise, "adaptive mask img")

    # Create adaptive mask with at least 3 good echoes, for classification
    ctx.masksum_clf = ctx.masksum_denoise.copy()
    ctx.masksum_clf[ctx.masksum_clf < 3] = 0
    ctx.mask_clf = ctx.masksum_clf.astype(bool)
    RepLGR.info(
        "A two-stage masking procedure was applied, in which a liberal mask "
        "(including voxels with good data in at least the first echo) was used for "
        "optimal combination, T2*/S0 estimation, and denoising, while a more conservative mask "
        "(restricted to voxels with good data in at least the first three echoes) was used for "
        "the component classification procedure."
    )
    LGR.debug(f"Retaining {ctx.mask_clf.sum()}/{ctx.n_samp} samples for classification")


# =============================================================================
# Stage 4: T2*/S0 Estimation
# =============================================================================


def fit_decay_model(ctx: PipelineContext) -> None:
    """Fit T2*/S0 decay model if not pre-computed.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    if ctx.t2smap_file is None:
        LGR.info("Computing T2* map")
        ctx.t2s_limited, ctx.s0_limited, ctx.t2s_full, ctx.s0_full = decay.fit_decay(
            ctx.data_cat, ctx.tes, ctx.mask_denoise, ctx.masksum_denoise, ctx.fittype
        )

        # Set a hard cap for the T2* map
        cap_t2s = stats.scoreatpercentile(
            ctx.t2s_full.flatten(), 99.5, interpolation_method="lower"
        )
        LGR.debug(f"Setting cap on T2* map at {utils.millisec2sec(cap_t2s):.5f}s")
        ctx.t2s_full[ctx.t2s_full > cap_t2s * 10] = cap_t2s
        ctx.io_generator.save_file(utils.millisec2sec(ctx.t2s_full), "t2star img")
        ctx.io_generator.save_file(ctx.s0_full, "s0 img")

        if ctx.verbose:
            ctx.io_generator.save_file(utils.millisec2sec(ctx.t2s_limited), "limited t2star img")
            ctx.io_generator.save_file(ctx.s0_limited, "limited s0 img")

        # Calculate RMSE
        rmse_map, rmse_df = decay.rmse_of_fit_decay_ts(
            data=ctx.data_cat,
            tes=ctx.tes,
            adaptive_mask=ctx.masksum_denoise,
            t2s=ctx.t2s_limited,
            s0=ctx.s0_limited,
            fitmode="all",
        )
        ctx.io_generator.save_file(rmse_map, "rmse img")
        ctx.io_generator.add_df_to_file(rmse_df, "confounds tsv")


# =============================================================================
# Stage 5: Optimal Combination
# =============================================================================


def compute_optimal_combination(ctx: PipelineContext) -> None:
    """Compute optimally combined data.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    ctx.data_optcom = combine.make_optcom(
        ctx.data_cat,
        ctx.tes,
        ctx.masksum_denoise,
        t2s=ctx.t2s_full,
        combmode=ctx.combmode,
    )

    # Apply global signal regression if requested
    if "gsr" in ctx.gscontrol:
        ctx.data_cat, ctx.data_optcom = gsc.gscontrol_raw(
            data_cat=ctx.data_cat,
            data_optcom=ctx.data_optcom,
            n_echos=ctx.n_echos,
            io_generator=ctx.io_generator,
        )

    fout = ctx.io_generator.save_file(ctx.data_optcom, "combined img")
    LGR.info(f"Writing optimally combined data set: {fout}")


# =============================================================================
# Stage 6: Decomposition (PCA + ICA)
# =============================================================================


def perform_pca_decomposition(ctx: PipelineContext) -> None:
    """Perform PCA decomposition for dimensionality reduction.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    ctx.data_reduced, ctx.n_components = decomposition.tedpca(
        ctx.data_cat,
        ctx.data_optcom,
        ctx.mask_clf,
        ctx.masksum_clf,
        ctx.io_generator,
        tes=ctx.tes,
        n_independent_echos=ctx.n_independent_echos,
        algorithm=ctx.tedpca,
        kdaw=10.0,
        rdaw=1.0,
        low_mem=ctx.low_mem,
    )

    if ctx.verbose:
        ctx.io_generator.save_file(utils.unmask(ctx.data_reduced, ctx.mask_clf), "whitened img")


def perform_ica_decomposition(ctx: PipelineContext) -> int:
    """Perform a single ICA decomposition attempt.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.

    Returns
    -------
    int
        Updated seed value.
    """
    seed = ctx.fixed_seed

    (
        ctx.mixing,
        seed,
        ctx.cluster_labels,
        ctx.similarity_t_sne,
        ctx.fastica_convergence_warning_count,
        index_quality,
    ) = decomposition.tedica(
        ctx.data_reduced,
        ctx.n_components,
        seed,
        ctx.ica_method,
        ctx.n_robust_runs,
        ctx.maxit,
        ctx.maxrestart,
    )

    # Store robustica quality metrics
    if ctx.ica_method.lower() == "robustica":
        if ctx.selector.cross_component_metrics_ is None:
            ctx.selector.cross_component_metrics_ = {}
        ctx.selector.cross_component_metrics_["robustica_mean_index_quality"] = index_quality

    return seed


def compute_component_metrics(ctx: PipelineContext) -> None:
    """Compute component metrics from ICA results.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    necessary_metrics = ctx.selector.necessary_metrics
    # Figures require some extra metrics
    extra_metrics = ["variance explained", "normalized variance explained", "kappa", "rho"]
    necessary_metrics = sorted(list(set(necessary_metrics + extra_metrics)))

    ctx.component_table, ctx.mixing = metrics.collect.generate_metrics(
        data_cat=ctx.data_cat,
        data_optcom=ctx.data_optcom,
        mixing=ctx.mixing,
        adaptive_mask=ctx.masksum_clf,
        tes=ctx.tes,
        n_independent_echos=ctx.n_independent_echos,
        io_generator=ctx.io_generator,
        label="ICA",
        metrics=necessary_metrics,
        external_regressors=ctx.external_regressors,
        external_regressor_config=ctx.selector.tree["external_regressor_config"],
    )


def perform_component_selection(ctx: PipelineContext) -> int:
    """Perform automatic component selection.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.

    Returns
    -------
    int
        Number of likely BOLD components found.
    """
    LGR.info("Selecting components from ICA results")
    ctx.selector = selection.automatic_selection(
        ctx.component_table,
        ctx.selector,
        n_echos=ctx.n_echos,
        n_vols=ctx.n_vols,
        n_independent_echos=ctx.n_independent_echos,
    )
    return ctx.selector.n_likely_bold_comps_


def run_decomposition_with_restarts(ctx: PipelineContext) -> None:
    """Run PCA and ICA decomposition with restart logic.

    This handles the case where ICA fails to find BOLD components
    and needs to be restarted with different seeds.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    # Initialize defaults
    ctx.cluster_labels = None
    ctx.similarity_t_sne = None
    ctx.fastica_convergence_warning_count = None

    if ctx.mixing_file is None:
        # Perform PCA
        perform_pca_decomposition(ctx)

        # ICA with restart logic
        keep_restarting = True
        n_restarts = 0
        seed = ctx.fixed_seed

        while keep_restarting:
            (
                ctx.mixing,
                seed,
                ctx.cluster_labels,
                ctx.similarity_t_sne,
                ctx.fastica_convergence_warning_count,
                index_quality,
            ) = decomposition.tedica(
                ctx.data_reduced,
                ctx.n_components,
                seed,
                ctx.ica_method,
                ctx.n_robust_runs,
                ctx.maxit,
                maxrestart=(ctx.maxrestart - n_restarts),
            )
            seed += 1
            n_restarts = seed - ctx.fixed_seed

            # Compute metrics and perform selection
            compute_component_metrics(ctx)
            n_likely_bold = perform_component_selection(ctx)

            if n_likely_bold == 0:
                if ctx.ica_method.lower() == "robustica":
                    LGR.warning("No BOLD components found with robustICA mixing matrix.")
                    keep_restarting = False
                elif n_restarts >= ctx.maxrestart:
                    LGR.warning(
                        "No BOLD components found, but maximum number of restarts reached."
                    )
                    keep_restarting = False
                else:
                    LGR.warning("No BOLD components found. Re-attempting ICA.")
                    ctx.io_generator.overwrite = True
                    # Re-initialize selector
                    tmp_ext_config = ctx.selector.tree["external_regressor_config"]
                    ctx.selector = ComponentSelector(ctx.tree)
                    ctx.selector.tree["external_regressor_config"] = tmp_ext_config
                    RepLGR.disabled = True
            else:
                keep_restarting = False

        RepLGR.disabled = False
        ctx.io_generator.overwrite = ctx.overwrite

        # Store robustica metrics
        if ctx.ica_method.lower() == "robustica":
            ctx.selector.cross_component_metrics_["fastica_convergence_warning_count"] = (
                ctx.fastica_convergence_warning_count
            )
            ctx.selector.cross_component_metrics_["robustica_mean_index_quality"] = index_quality

    else:
        # Use supplied mixing matrix
        LGR.info("Using supplied mixing matrix from ICA")
        ctx.mixing = pd.read_table(ctx.mixing_file).values
        compute_component_metrics(ctx)
        n_likely_bold = perform_component_selection(ctx)

        if n_likely_bold == 0:
            LGR.warning("No BOLD components found with user-provided ICA mixing matrix.")

    # Clear intermediate decomposition data to free memory
    ctx.clear_intermediate_data("decomposition")


# =============================================================================
# Stage 7: Output Generation
# =============================================================================


def save_component_outputs(ctx: PipelineContext) -> None:
    """Save component-related outputs (mixing matrix, betas, etc.).

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to use.
    """
    # Update component table from selector
    ctx.component_table = ctx.selector.component_table_

    # Save mixing matrix
    comp_names = ctx.component_table["Component"].values
    mixing_df = pd.DataFrame(data=ctx.mixing, columns=comp_names)
    ctx.io_generator.save_file(mixing_df, "ICA mixing tsv")

    # Save z-scored component maps
    betas_oc = utils.unmask(
        computefeats2(ctx.data_optcom, ctx.mixing, ctx.mask_denoise), ctx.mask_denoise
    )
    ctx.io_generator.save_file(betas_oc, "z-scored ICA components img")

    # Calculate rejected component impact
    reporting.quality_metrics.calculate_rejected_components_impact(ctx.selector, ctx.mixing)

    # Save selector and metrics
    ctx.selector.to_files(ctx.io_generator)
    metric_metadata = metrics.collect.get_metadata(ctx.component_table)
    ctx.io_generator.save_file(metric_metadata, "ICA metrics json")

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
    ctx.io_generator.save_file(decomp_metadata, "ICA decomposition json")

    if ctx.selector.n_likely_bold_comps_ == 0:
        LGR.warning("No BOLD components detected! Please check data and results!")


def apply_tedort(ctx: PipelineContext) -> None:
    """Apply tedort orthogonalization if requested.

    Orthogonalizes rejected components with respect to accepted components.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to modify.
    """
    ctx.mixing_orig = ctx.mixing.copy()

    if ctx.tedort:
        comps_accepted = ctx.selector.accepted_comps_
        comps_rejected = ctx.selector.rejected_comps_
        acc_ts = ctx.mixing[:, comps_accepted]
        rej_ts = ctx.mixing[:, comps_rejected]
        betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
        pred_rej_ts = np.dot(acc_ts, betas)
        resid = rej_ts - pred_rej_ts
        ctx.mixing[:, comps_rejected] = resid

        comp_names = [
            io.add_decomp_prefix(comp, prefix="ICA", max_value=ctx.component_table.index.max())
            for comp in range(ctx.selector.n_comps_)
        ]

        mixing_df = pd.DataFrame(data=ctx.mixing, columns=comp_names)
        ctx.io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")
        RepLGR.info(
            "Rejected components' time series were then "
            "orthogonalized with respect to accepted components' time "
            "series."
        )


def write_denoised_data(ctx: PipelineContext) -> None:
    """Write denoised data and component results.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to use.
    """
    io.writeresults(
        ctx.data_optcom,
        mask=ctx.mask_denoise,
        component_table=ctx.component_table,
        mixing=ctx.mixing,
        io_generator=ctx.io_generator,
    )

    if "mir" in ctx.gscontrol:
        gsc.minimum_image_regression(
            data_optcom=ctx.data_optcom,
            mixing=ctx.mixing,
            mask=ctx.mask_denoise,
            component_table=ctx.component_table,
            classification_tags=ctx.selector.classification_tags,
            io_generator=ctx.io_generator,
        )

    if ctx.verbose:
        io.writeresults_echoes(
            ctx.data_cat, ctx.mixing, ctx.mask_denoise, ctx.component_table, ctx.io_generator
        )


def save_registry_and_metadata(ctx: PipelineContext) -> None:
    """Save output registry and BIDS metadata.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to use.
    """
    # Save registry
    ctx.io_generator.save_self()

    # Save BIDS derivative metadata
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
                    "Name": ctx.info_dict["Node"],
                    "System": ctx.info_dict["System"],
                    "Machine": ctx.info_dict["Machine"],
                    "Processor": ctx.info_dict["Processor"],
                    "Release": ctx.info_dict["Release"],
                    "Version": ctx.info_dict["Version"],
                },
                "Python": ctx.info_dict["Python"],
                "Python_Libraries": ctx.info_dict["Python_Libraries"],
                "Command": ctx.info_dict["Command"],
            }
        ],
    }
    with open(ctx.io_generator.get_name("data description json"), "w") as fo:
        json.dump(derivative_metadata, fo, sort_keys=True, indent=4)


def finalize_report_text(ctx: PipelineContext) -> None:
    """Finalize report text and BibTeX references.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to use.
    """
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

    with open(ctx.repname) as fo:
        report = [line.rstrip() for line in fo.readlines()]
        report = " ".join(report)
        report = report.replace("  ", "\n\n")

    with open(ctx.repname, "w") as fo:
        fo.write(report)

    # Collect BibTeX entries
    references = get_description_references(report)

    with open(ctx.bibtex_file, "w") as fo:
        fo.write(references)


# =============================================================================
# Stage 8: Report Generation
# =============================================================================


def generate_reports(ctx: PipelineContext) -> None:
    """Generate HTML reports and figures.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to use.
    """
    if ctx.no_reports:
        return

    LGR.info("Making figures folder with static component maps and timecourse plots.")

    data_denoised, data_accepted, data_rejected = io.denoise_ts(
        ctx.data_optcom,
        ctx.mixing,
        ctx.mask_denoise,
        ctx.component_table,
    )

    reporting.static_figures.plot_adaptive_mask(
        optcom=ctx.data_optcom,
        base_mask=ctx.mask,
        io_generator=ctx.io_generator,
    )
    reporting.static_figures.carpet_plot(
        optcom_ts=ctx.data_optcom,
        denoised_ts=data_denoised,
        hikts=data_accepted,
        lowkts=data_rejected,
        mask=ctx.mask_denoise,
        io_generator=ctx.io_generator,
        gscontrol=ctx.gscontrol,
    )
    reporting.static_figures.comp_figures(
        ctx.data_optcom,
        mask=ctx.mask_denoise,
        component_table=ctx.component_table,
        mixing=ctx.mixing_orig,
        io_generator=ctx.io_generator,
        png_cmap=ctx.png_cmap,
    )
    reporting.static_figures.plot_t2star_and_s0(
        io_generator=ctx.io_generator, mask=ctx.mask_denoise
    )

    if ctx.t2smap_file is None:
        reporting.static_figures.plot_rmse(
            io_generator=ctx.io_generator,
            adaptive_mask=ctx.masksum_denoise,
        )

    if ctx.gscontrol:
        reporting.static_figures.plot_gscontrol(
            io_generator=ctx.io_generator,
            gscontrol=ctx.gscontrol,
        )

    if ctx.external_regressors is not None:
        comp_names = ctx.component_table["Component"].values
        mixing_df = pd.DataFrame(data=ctx.mixing, columns=comp_names)
        reporting.static_figures.plot_heatmap(
            mixing=mixing_df,
            external_regressors=ctx.external_regressors,
            component_table=ctx.component_table,
            out_file=os.path.join(
                ctx.io_generator.out_dir,
                "figures",
                f"{ctx.io_generator.prefix}confound_correlations.svg",
            ),
        )

    LGR.info("Generating dynamic report")
    reporting.generate_report(ctx.io_generator, ctx.cluster_labels, ctx.similarity_t_sne)


# =============================================================================
# Stage 9: Cleanup
# =============================================================================


def cleanup(ctx: PipelineContext) -> None:
    """Perform final cleanup and logging teardown.

    Parameters
    ----------
    ctx : PipelineContext
        Pipeline context to clean up.
    """
    LGR.info("Workflow completed")

    # Log newsletter info
    utils.log_newsletter_info()

    # Final memory cleanup
    ctx.clear_intermediate_data("all")

    # Teardown loggers
    utils.teardown_loggers()


# =============================================================================
# High-level Workflow Functions
# =============================================================================


def run_tedana_pipeline(ctx: PipelineContext) -> None:
    """Run the complete tedana pipeline.

    This is the main orchestration function that calls each stage
    in sequence.

    Parameters
    ----------
    ctx : PipelineContext
        Initialized pipeline context.
    """
    # Stage 1: Setup
    setup_output_directory(ctx)
    validate_inputs(ctx)
    initialize_component_selector(ctx)

    # Stage 2: Data Loading
    load_data(ctx)
    setup_io_generator(ctx)
    validate_tr(ctx)
    load_external_regressors(ctx)
    handle_precomputed_files(ctx)

    # Stage 3: Masking
    create_masks(ctx)

    # Stage 4: T2*/S0 Estimation
    fit_decay_model(ctx)

    # Stage 5: Optimal Combination
    compute_optimal_combination(ctx)

    # Stage 6: Decomposition
    run_decomposition_with_restarts(ctx)

    # Stage 7: Output Generation
    save_component_outputs(ctx)
    apply_tedort(ctx)
    write_denoised_data(ctx)
    save_registry_and_metadata(ctx)
    finalize_report_text(ctx)

    # Stage 8: Report Generation
    generate_reports(ctx)

    # Stage 9: Cleanup
    cleanup(ctx)
