"""Reporting code for tedana."""

from tedana.reporting.html_report import generate_report
from tedana.reporting.static_figures import comp_figures, pca_results

__all__ = ["generate_report", "comp_figures", "pca_results"]
