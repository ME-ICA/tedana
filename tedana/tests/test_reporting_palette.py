"""Tests for tedana.reporting._palette."""

from tedana.reporting import _palette


def test_known_classes_have_distinct_color_and_marker():
    classes = ["accepted", "rejected", "ignored"]
    colors = [_palette.get_style(c).color for c in classes]
    markers = [_palette.get_style(c).bokeh_marker for c in classes]
    assert len(set(colors)) == len(classes)
    assert len(set(markers)) == len(classes)


def test_unknown_class_falls_back_to_other():
    style = _palette.get_style("brand-new-label")
    assert style is _palette.CLASSIFICATION_STYLES["other"]


def test_helpers_match_get_style():
    assert _palette.color_for("accepted") == _palette.get_style("accepted").color
    assert _palette.marker_for("rejected") == _palette.get_style("rejected").bokeh_marker
    assert _palette.linestyle_for("ignored") == _palette.get_style("ignored").mpl_linestyle


def test_okabe_ito_values():
    assert _palette.color_for("accepted") == "#009E73"
    assert _palette.color_for("rejected") == "#D55E00"
    assert _palette.color_for("ignored") == "#0072B2"
    assert _palette.color_for("other") == "#999999"
