"""Colorblind-safe classification styling for tedana report figures.

Single source of truth for how each component classification is drawn across
the Bokeh (dynamic) and Matplotlib (static) report figures. Colors follow the
Okabe-Ito colorblind-safe palette; marker shape and line style provide
redundant, non-color encoding so classes remain distinguishable under
color-vision deficiency.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassStyle:
    """Visual style for one component classification."""

    color: str
    bokeh_marker: str
    mpl_linestyle: str


CLASSIFICATION_STYLES = {
    "accepted": ClassStyle("#009E73", "circle", "solid"),
    "rejected": ClassStyle("#D55E00", "square", "dashed"),
    "ignored": ClassStyle("#0072B2", "triangle", "dotted"),
    "other": ClassStyle("#999999", "diamond", "dashdot"),
}


def get_style(classification):
    """Return the ClassStyle for a classification, falling back to 'other'."""
    return CLASSIFICATION_STYLES.get(classification, CLASSIFICATION_STYLES["other"])


def color_for(classification):
    """Return the fill/line color for a classification."""
    return get_style(classification).color


def marker_for(classification):
    """Return the Bokeh marker name for a classification."""
    return get_style(classification).bokeh_marker


def linestyle_for(classification):
    """Return the Matplotlib line style for a classification."""
    return get_style(classification).mpl_linestyle
