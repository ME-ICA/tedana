# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: holoviz-tutorial
#     language: python
#     name: holoviz-tutorial
# ---

# %%
import holoviews as hv
import pandas as pd

from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div, Range1d
from bokeh.events import Tap
from bokeh.embed import components
from bokeh.layouts import row, column
from bokeh.plotting import figure

from tedana.reporting import generate_report

from sklearn.preprocessing import MinMaxScaler
import os.path as osp

hv.extension('bokeh')

# %%
OUTDIR = '/opt/tedana-hack/tedana/docs/DynReports/data/TED.five-echo/'
state2col = {'accepted': '#00ff00', 'rejected': '#ff0000', 'ignored': '#0000ff'}


# %%
def prepare_comp_table(comp_table_DIR):
    CompTable_Path = osp.join(comp_table_DIR, 'comp_table_ica.tsv')
    DF = pd.read_csv(CompTable_Path, sep='\t')
    Nc = DF.shape[0]
    # When needed, remove space from column names (bokeh is not happy about it)
    DF.rename(columns={'variance explained': 'var_exp'}, inplace=True)

    # For providing sizes based on Var Explained that are visible
    mm_scaler = MinMaxScaler(feature_range=(4, 20))
    DF['var_exp_size'] = mm_scaler.fit_transform(
        DF[['var_exp', 'normalized variance explained']])[:, 0]

    # Remove unsed columns to decrease size of final HTML
    DF.drop(['normalized variance explained', 'countsigFR2', 'countsigFS0', 'dice_FS0',
             'countnoise', 'dice_FR2', 'signal-noise_t', 'signal-noise_p',
             'd_table_score', 'kappa ratio', 'rationale', 'd_table_score_scrub'],
            axis=1, inplace=True)

    # Create additional Column with colors based on final classification
    DF['color'] = [state2col[i] for i in DF['classification']]

    CompTable_CDS = ColumnDataSource(data=dict(
        x=DF['kappa'],
        y=DF['rho'],
        varexp=DF['var_exp'],
        component=[str(i) for i in DF['component']],
        color=DF['color'],
        size=DF['var_exp_size'],
        classif=DF['classification']))

    return CompTable_CDS, Nc


# %%
def tap_callback(CSD, div):
    return CustomJS(args=dict(source=CSD, div=div_content, outdir=OUTDIR), code="""
    // the event that triggered the callback is cb_obj:
    // The event type determines the relevant attributes
    var data     = source.data;
    var selected = source.selected.indices
    var selected_padded = '' + selected;
    while (selected_padded.length < 3) {
        selected_padded = '0' + selected_padded;
    }
    console.log('selected = ' + selected_padded)
    div.text = ""
    var line = "<span><img src='" + outdir + "/figures/comp_"+selected_padded+".png'" +
        " alt='Component Map' height=1000 width=900><span>\\n";
    console.log('Linea: ' + line)
    var text = div.text.concat(line);
    var lines = text.split("\\n")
        if (lines.length > 35)
            lines.shift();
    div.text = lines.join("\\n");
    """)


# %%
def create_krPlot(CompTable_CDS, div):
    # Create Panel for the Kappa - Rho Scatter
    kr_hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@x'),
                                       ('Rho', '@y'), ('Var. Expl.', '@varexp')])
    krFig = figure(plot_width=400, plot_height=400,
                   tools=["tap,wheel_zoom,reset,pan,crosshair", kr_hovertool],
                   title="Kappa / Rho Plot")
    krFig.circle('x', 'y', size='size', color='color', alpha=0.5, source=CompTable_CDS,
                 legend_group='classif')
    krFig.xaxis.axis_label = 'Kappa'
    krFig.yaxis.axis_label = 'Rho'
    krFig.toolbar.logo = None
    krFig.legend.background_fill_alpha = 0.5
    krFig.legend.orientation = 'horizontal'
    krFig.legend.location = 'bottom_right'
    krFig.js_on_event(Tap, tap_callback(CompTable_CDS, div))
    return krFig


# %%
def create_ksortedPlot(CompTable_CDS, Nc, div):
    # Create Panel for the Ranked Kappa Plot
    ksorted_hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@x'),
                                            ('Rho', '@y'), ('Var. Expl.', '@varexp')])
    ksorted_Fig = figure(plot_width=400, plot_height=400,
                         tools=["tap,wheel_zoom,reset,pan,crosshair", ksorted_hovertool],
                         title="Ranked Kappa Plot")
    ksorted_Fig.circle('component', 'x', source=CompTable_CDS, size=3, color='color')
    ksorted_Fig.xaxis.axis_label = 'Kappa Rank'
    ksorted_Fig.yaxis.axis_label = 'Kappa'
    ksorted_Fig.x_range = Range1d(-1, Nc + 1)
    ksorted_Fig.toolbar.logo = None
    ksorted_Fig.js_on_event(Tap, tap_callback(CompTable_CDS, div))

    return ksorted_Fig


# %%
# Embed into Report Templates
# ---------------------------
[CompTable_CDS, Nc] = prepare_comp_table(OUTDIR)
div_content = Div(width=600, height=900, height_policy='fixed')
kappa_rho_plot = create_krPlot(CompTable_CDS, div_content)
kappa_sorted_plot = create_ksortedPlot(CompTable_CDS, Nc, div_content)
app = row(column(kappa_rho_plot, kappa_sorted_plot), div_content)
(kr_script, kr_div) = components(app)
generate_report(kr_div, kr_script,file_path='/opt/report.html')

# %%
