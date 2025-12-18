"""Shared modules for tedana workflows.

This package provides reusable components that are shared across
the tedana, t2smap, and ica_reclassify workflows.
"""

from tedana.workflows.shared.combination import (
    compute_optimal_combination,
    compute_optimal_combination_simple,
)
from tedana.workflows.shared.containers import (
    DecayMaps,
    DecompositionResult,
    MaskData,
    MultiEchoData,
    OptcomData,
    WorkflowConfig,
)
from tedana.workflows.shared.data_loading import load_multiecho_data, validate_tr
from tedana.workflows.shared.fitting import fit_decay_model, fit_decay_model_simple
from tedana.workflows.shared.masking import (
    create_adaptive_masks,
    create_simple_adaptive_mask,
)
from tedana.workflows.shared.output import (
    apply_mir,
    apply_tedort,
    finalize_report_text,
    save_derivative_metadata,
    write_denoised_results,
    write_echo_results,
)
from tedana.workflows.shared.reporting import (
    generate_dynamic_report,
    generate_reclassify_figures,
    generate_static_figures,
)
from tedana.workflows.shared.setup import (
    rename_previous_reports,
    save_workflow_command,
    setup_logging,
    setup_output_directory,
    teardown_workflow,
)

__all__ = [
    # Containers
    "WorkflowConfig",
    "MultiEchoData",
    "MaskData",
    "DecayMaps",
    "OptcomData",
    "DecompositionResult",
    # Setup
    "setup_output_directory",
    "setup_logging",
    "save_workflow_command",
    "rename_previous_reports",
    "teardown_workflow",
    # Data loading
    "load_multiecho_data",
    "validate_tr",
    # Masking
    "create_adaptive_masks",
    "create_simple_adaptive_mask",
    # Fitting
    "fit_decay_model",
    "fit_decay_model_simple",
    # Combination
    "compute_optimal_combination",
    "compute_optimal_combination_simple",
    # Output
    "apply_tedort",
    "write_denoised_results",
    "apply_mir",
    "write_echo_results",
    "save_derivative_metadata",
    "finalize_report_text",
    # Reporting
    "generate_static_figures",
    "generate_dynamic_report",
    "generate_reclassify_figures",
]
