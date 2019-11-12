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

hv.extension('bokeh')

# %%
# For Translating Classification into Colors
# ------------------------------------------
state2col = {'accepted': '#00ff00', 'rejected': '#ff0000', 'ignored': '#0000ff'}
# Load Comp Table
# ---------------
CompTable_Path = './data/TED.five-echo/comp_table_ica.tsv'
CompTable_DF = pd.read_csv(CompTable_Path, sep='\t')
Nc = CompTable_DF.shape[0]  # Obtain number of components
# When needed, remove space from column names (bokeh is not happy about it)
CompTable_DF.rename(columns={'variance explained': 'var_exp'}, inplace=True)

# For providing sizes based on Var Explained that are visible
# -----------------------------------------------------------
mm_scaler = MinMaxScaler(feature_range=(4, 20))
CompTable_DF['var_exp_size'] = mm_scaler.fit_transform(
        CompTable_DF[['var_exp', 'normalized variance explained']])[:, 0]

# Remove unsed columns to decrease size of final HTML
# ---------------------------------------------------
CompTable_DF.drop(['normalized variance explained', 'countsigFR2', 'countsigFS0', 'dice_FS0',
                   'countnoise', 'dice_FR2', 'signal-noise_t', 'signal-noise_p', 'd_table_score',
                   'kappa ratio', 'rationale', 'd_table_score_scrub'], axis=1, inplace=True)

# Create additional Column with colors based on final classification
# ------------------------------------------------------------------
CompTable_DF['color'] = [state2col[i] for i in CompTable_DF['classification']]


# %%
def tap_callback(div):
    return CustomJS(args=dict(source=krPlot_source, div=div_content), code="""
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
    var line = "<span><img src='data/TED.five-echo/figures/comp_"+selected_padded+".png'" +
        " alt='Component Map' height=1000 width=900><span>\\n";
    console.log('Linea: ' + line)
    var text = div.text.concat(line);
    var lines = text.split("\\n")
        if (lines.length > 35)
            lines.shift();
    div.text = lines.join("\\n");
    """)


# Create Panel for the Kappa - Rho Scatter
# ----------------------------------------
kr_hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@x'),
                                   ('Rho', '@y'), ('Var. Expl.', '@varexp')])
krFig = figure(plot_width=400, plot_height=400,
               tools=["tap,wheel_zoom,reset,pan,crosshair",kr_hovertool],
               title="Kappa / Rho Plot")

krPlot_source = ColumnDataSource(data=dict(
        x=CompTable_DF['kappa'],
        y=CompTable_DF['rho'],
        varexp=CompTable_DF['var_exp'],
        component=[str(i) for i in CompTable_DF['component']],
        color=CompTable_DF['color'],
        size=CompTable_DF['var_exp_size'],
        classif=CompTable_DF['classification']))
krFig.circle('x', 'y', size='size', color='color', alpha=0.5, source=krPlot_source, 
             legend_group='classif')
krFig.xaxis.axis_label = 'Kappa'
krFig.yaxis.axis_label = 'Rho'
krFig.toolbar.logo = None
krFig.legend.background_fill_alpha = 0.5
krFig.legend.orientation = 'horizontal'
krFig.legend.location = 'bottom_right'
# Create Panel for the Ranked Kappa Plot
# --------------------------------------
ksorted_hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@x'),
                                        ('Rho', '@y'), ('Var. Expl.', '@varexp')])
ksorted_Fig = figure(plot_width=400, plot_height=400,
                     tools=["tap,wheel_zoom,reset,pan,crosshair",ksorted_hovertool],
                     title="Ranked Kappa Plot")
ksorted_Fig.circle('component', 'x', source=krPlot_source, size=3, color='color')
ksorted_Fig.xaxis.axis_label = 'Kappa Rank'
ksorted_Fig.yaxis.axis_label = 'Kappa'
ksorted_Fig.x_range = Range1d(-1, Nc + 1)
ksorted_Fig.toolbar.logo = None

# Create DIV where images will go
# -------------------------------
div_content = Div(width=600, height=krFig.plot_height, height_policy='fixed')

# Link responses to Plots
# -----------------------
krFig.js_on_event(Tap, tap_callback(div_content))
ksorted_Fig.js_on_event(Tap, tap_callback(div_content))

# Create Final Layout
# -------------------
app = row(column(krFig, ksorted_Fig), div_content)

# %%
# Embed into Report Templates
# ---------------------------
(kr_script, kr_div) = components(app)
generate_report(kr_div, kr_script)

# %% [markdown]
# ***

# %%
krFig.legend.orientation
