"""
Reporting code for tedana
"""

from .html_report import generate_report
from .static_figures import comp_figures

__all__ = ['generate_report', 'comp_figures']
