"""Tests for tedana.reporting."""

import json
import re
import shutil
from os.path import dirname, join

import numpy as np
import pandas as pd
import pytest

from tedana import reporting
from tedana.reporting import html_report
from tedana.tests.test_external_metrics import sample_mixing_matrix
from tedana.tests.test_selection_utils import sample_selector


def test_smoke_trim_edge_zeros():
    """Ensures that trim_edge_zeros works with random inputs."""
    arr = np.random.random((100, 100))
    assert reporting.static_figures._trim_edge_zeros(arr) is not None


def test_calculate_rejected_components_impact():
    selector = sample_selector()
    mixing = sample_mixing_matrix()

    component_table = selector.component_table_

    reporting.calculate_rejected_components_impact(selector, mixing)

    # Total matches expecting value in testing data
    assert np.round(
        selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"], 4
    ) == np.round(4.0927834, 4)

    assert "Var Exp of rejected to accepted" in selector.component_table_.columns
    rej = component_table[component_table["classification"] == "rejected"]
    acc = component_table[component_table["classification"] == "accepted"]

    assert rej["Var Exp of rejected to accepted"].isna().all()
    assert not acc["Var Exp of rejected to accepted"].isna().all()


def test_calculate_rejected_components_impact_no_rej():
    selector = sample_selector()
    mixing = sample_mixing_matrix()

    component_table = selector.component_table_

    component_table.drop(
        component_table[component_table["classification"] == "rejected"].index, inplace=True
    )

    reporting.calculate_rejected_components_impact(selector, mixing)

    assert "Var Exp of rejected to accepted" in selector.component_table_.columns
    assert selector.component_table_["Var Exp of rejected to accepted"].isna().all()
    assert np.isnan(
        selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"]
    )


def test_calculate_rejected_components_impact_no_acc():
    selector = sample_selector()
    mixing = sample_mixing_matrix()

    component_table = selector.component_table_

    component_table.drop(
        component_table[component_table["classification"] == "accepted"].index, inplace=True
    )

    reporting.calculate_rejected_components_impact(selector, mixing)

    assert "Var Exp of rejected to accepted" in selector.component_table_.columns
    assert selector.component_table_["Var Exp of rejected to accepted"].isna().all()
    assert np.isnan(
        selector.cross_component_metrics_["total_var_exp_rejected_components_on_accepted"]
    )


def test_plot_heatmap_nonfinite_distances_warns_and_succeeds(tmp_path):
    """Ensure plot_heatmap does not crash when correlation-derived distances are non-finite.

    This can happen when an external regressor has zero variance (constant time series),
    which yields NaNs in the correlation computations used for hierarchical clustering.
    """

    # Construct a correlation matrix with one regressor that has a constant
    # correlation pattern across components, which yields zero variance and
    # can produce non-finite correlations when clustering regressors.
    correlation_df = pd.DataFrame(
        {
            "ICA_00": [0.5, 0.1],
            "ICA_01": [0.5, -0.2],
        },
        index=["constant", "varying"],
    )

    # component_table needs at least one model column that matches the regex
    # used in plot_heatmap: "(R2stat .* model)".
    component_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01"],
            "R2stat demo model": [0.1, 0.9],
        },
        index=[0, 1],
    )

    out_file = tmp_path / "heatmap.png"

    with pytest.warns(UserWarning, match=r"Non-finite correlations detected.*constant"):
        reporting.static_figures.plot_heatmap(
            correlation_df=correlation_df,
            component_table=component_table,
            out_file=str(out_file),
        )

    assert out_file.exists()


def test_plot_stat_mosaic_writes_output(tmp_path):
    import nibabel as nb
    import numpy as np

    from tedana.reporting import static_figures

    affine = np.eye(4)
    data = np.abs(np.random.RandomState(0).randn(12, 12, 12)).astype("float32")
    img = nb.Nifti1Image(data, affine)
    in_file = tmp_path / "map.nii.gz"
    img.to_filename(in_file)
    mask = nb.Nifti1Image(np.ones((12, 12, 12), dtype="int16"), affine)

    out_file = tmp_path / "map.svg"
    static_figures._plot_stat_mosaic(
        in_file=str(in_file), out_file=str(out_file), cmap="Reds", mask_img=mask
    )
    assert out_file.exists()


class _FailuresIO:
    """Minimal io_generator stub for plot_fit_failures."""

    def __init__(self, out_dir, failures_path):
        self.out_dir = str(out_dir)
        self.prefix = ""
        self._failures_path = str(failures_path)

    def get_name(self, description):
        assert description == "fit failures img"
        return self._failures_path


def _write_failures_nifti(path, values):
    import nibabel as nb
    import numpy as np

    arr = np.zeros((10, 10, 10), dtype="int16")
    for i, v in enumerate(values):
        arr[i, 0, 0] = v
    nb.Nifti1Image(arr, np.eye(4)).to_filename(str(path))


def test_plot_fit_failures_writes_when_failures_exist(tmp_path):
    from tedana.reporting import static_figures

    (tmp_path / "figures").mkdir()
    failures = tmp_path / "fit_failures.nii.gz"
    _write_failures_nifti(failures, [1, 2, 1])  # some failures

    static_figures.plot_fit_failures(io_generator=_FailuresIO(tmp_path, failures))
    assert (tmp_path / "figures" / "fit_failures.svg").exists()


def test_plot_fit_failures_skips_when_no_failures(tmp_path):
    from tedana.reporting import static_figures

    (tmp_path / "figures").mkdir()
    failures = tmp_path / "fit_failures.nii.gz"
    _write_failures_nifti(failures, [0, 0, 0])  # all clean

    static_figures.plot_fit_failures(io_generator=_FailuresIO(tmp_path, failures))
    assert not (tmp_path / "figures" / "fit_failures.svg").exists()


class _StubIOGenerator:
    """Return paths inside out_dir, like OutputGenerator does for the tree files."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.prefix = ""

    def get_name(self, description):
        names = {
            "ICA decision tree json": "desc-ICA_decision_tree.json",
            "ICA status table tsv": "desc-ICA_status_table.tsv",
        }
        return join(self.out_dir, names[description])


def _render_body(tmp_path, **kwargs):
    """Render the report body template with the minimum viable arguments."""
    # _update_template_bokeh derives the figures directory from the references path.
    references = str(tmp_path / "references.bib")
    shutil.copyfile(join(dirname(reporting.__file__), "../resources/references.bib"), references)
    (tmp_path / "figures").mkdir(exist_ok=True)

    render_kwargs = {
        "bokeh_id": "",
        "info_table": "",
        "about": "",
        "prefix": "",
        "references": references,
        "bokeh_js": "",
        "buttons": "",
        "tsne": "",
        "tree_table": None,
        "status_table": None,
    }
    render_kwargs.update(kwargs)
    return html_report._update_template_bokeh(**render_kwargs)


def test_update_template_bokeh_omits_empty_tabs(tmp_path):
    """Info/ICA/Carpet are always rendered, but empty Decay and Tree tabs are not."""
    body = _render_body(tmp_path)

    assert 'role="tablist"' in body
    for pane in ("pane-info", "pane-ica", "pane-carpet"):
        assert f'id="{pane}"' in body

    # No decay figures and no tree files, so neither tab should exist at all.
    assert 'id="pane-decay"' not in body
    assert 'id="pane-tree"' not in body
    assert body.count('aria-controls="pane-') == 3

    # ICA is the only tab open on load.
    assert body.count("tab-pane is-active") == 1
    assert '<div class="tab-pane is-active" id="pane-ica"' in body
    assert re.search(r'id="tab-ica"\s+aria-controls="pane-ica" aria-selected="true"', body)
    assert re.search(r'id="tab-info"\s+aria-controls="pane-info" aria-selected="false"', body)


def test_update_template_bokeh_decay_tab_needs_only_one_figure_set(tmp_path):
    """The Decay tab appears when any one of its figure sets exists, not only all of them."""
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ("rmse_brain.svg", "rmse_timeseries.svg"):
        (figures / name).touch()

    body = _render_body(tmp_path)

    assert 'id="pane-decay"' in body
    assert body.count('aria-controls="pane-') == 4


def test_update_template_bokeh_fit_failures(tmp_path):
    """The fit-failure map is rendered under Curve-fit quality when the SVG exists."""
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ("rmse_brain.svg", "rmse_timeseries.svg", "fit_failures.svg"):
        (figures / name).touch()

    body = _render_body(tmp_path)

    assert 'id="pane-decay"' in body
    assert 'id="fitFailuresPlot"' in body
    assert "Curve-fit quality" in body


def test_update_template_bokeh_no_fit_failures(tmp_path):
    """The fit-failure map is omitted when the SVG does not exist."""
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ("rmse_brain.svg", "rmse_timeseries.svg"):
        (figures / name).touch()

    body = _render_body(tmp_path)

    assert 'id="fitFailuresPlot"' not in body


def test_update_template_bokeh_tree_tab(tmp_path):
    """The Tree tab appears when the decision tree tables were generated."""
    body = _render_body(tmp_path, tree_table="<table></table>", status_table="<table></table>")

    assert 'id="pane-tree"' in body
    assert body.count('aria-controls="pane-') == 4


def test_update_template_bokeh_tab_order(tmp_path):
    """Tree sits next to ICA, and the panes follow the same order as the tabs."""
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in ("rmse_brain.svg", "rmse_timeseries.svg"):
        (figures / name).touch()

    body = _render_body(tmp_path, tree_table="<table></table>", status_table="<table></table>")

    expected = ["pane-info", "pane-ica", "pane-tree", "pane-carpet", "pane-decay"]
    assert re.findall(r'aria-controls="(pane-[a-z]+)"', body) == expected
    assert re.findall(r'<div class="tab-pane[^"]*" id="(pane-[a-z]+)"', body) == expected


def test_generate_tree_tables_missing_files(tmp_path):
    """Missing decision tree files should not raise, just yield no tables."""
    tree_table, status_table = html_report._generate_tree_tables(_StubIOGenerator(str(tmp_path)))

    assert tree_table is None
    assert status_table is None


def test_generate_tree_tables(tmp_path):
    """Nodes missing n_true/n_false or with a list node_label should still render."""
    tree = {
        "nodes": [
            {
                "functionname": "manual_classify",
                "outputs": {
                    "decision_node_idx": 0,
                    "node_label": "Set all to unclassified",
                    "n_true": 39,
                    "n_false": 0,
                    "used_metrics": [],
                },
            },
            {
                # This node reports no n_true/n_false and labels itself with a list,
                # both of which happen in real decision trees.
                "functionname": "calc_kappa_elbow",
                "outputs": {
                    "decision_node_idx": 1,
                    "node_label": ["countsigFS0>countsigFT2", "countsigFT2>0"],
                    "used_metrics": ["kappa", "rho"],
                },
            },
        ]
    }
    (tmp_path / "desc-ICA_decision_tree.json").write_text(json.dumps(tree))
    pd.DataFrame({"Component": ["ICA_00"], "Node 0": ["accepted"]}).to_csv(
        tmp_path / "desc-ICA_status_table.tsv", sep="\t", index=False
    )

    tree_table, status_table = html_report._generate_tree_tables(_StubIOGenerator(str(tmp_path)))

    assert "Set all to unclassified" in tree_table
    assert "countsigFS0&gt;countsigFT2, countsigFT2&gt;0" in tree_table
    assert "kappa, rho" in tree_table
    assert "pure-table" in tree_table
    assert "ICA_00" in status_table


def test_update_template_bokeh_omits_empty_curvefit_quality(tmp_path):
    """The Curve-fit quality heading is not shown when it would have no content."""
    figures = tmp_path / "figures"
    figures.mkdir()
    # T2* estimate present (so the Decay tab exists), but no RMSE/variance/failures.
    for name in ("t2star_brain.svg", "t2star_histogram.svg"):
        (figures / name).touch()

    body = _render_body(tmp_path)

    assert 'id="pane-decay"' in body
    assert "Parameter estimates" in body
    assert "Curve-fit quality" not in body
