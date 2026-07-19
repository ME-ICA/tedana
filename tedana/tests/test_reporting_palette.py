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


def test_create_data_struct_adds_color_and_marker(tmp_path):
    import pandas as pd

    from tedana.reporting import dynamic_figures as df_mod

    comptable = pd.DataFrame(
        {
            "kappa": [50.0, 10.0],
            "rho": [5.0, 40.0],
            "variance explained": [60.0, 40.0],
            "normalized variance explained": [0.6, 0.4],
            "Var Exp of rejected to accepted": [0.0, 0.0],
            "classification": ["accepted", "rejected"],
            "classification_tags": ["", ""],
        }
    )
    comptable_path = tmp_path / "comptable.tsv"
    comptable.to_csv(comptable_path, sep="\t", index=False)

    cds = df_mod._create_data_struct(str(comptable_path))

    assert "marker" in cds.data
    assert "color" in cds.data
    # Rows are re-sorted by classification then var_exp; check as a set-of-triples.
    triples = set(zip(cds.data["classif"], cds.data["color"], cds.data["marker"]))
    assert ("accepted", "#009E73", "circle") in triples
    assert ("rejected", "#D55E00", "square") in triples


def test_plot_component_accepts_linestyle():
    import inspect

    from tedana.reporting import static_figures

    sig = inspect.signature(static_figures.plot_component)
    assert "classification_linestyle" in sig.parameters
    assert sig.parameters["classification_linestyle"].default == "solid"
