"""Reporting code for tedana."""

from tedana.reporting.html_report import generate_report
from tedana.reporting.quality_metrics import calculate_rejected_components_impact
from tedana.reporting.static_figures import comp_figures, pca_results

__all__ = [
    "calculate_rejected_components_impact",
    "comp_figures",
    "generate_report",
    "pca_results",
]
