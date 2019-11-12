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
#     display_name: tedana-hack
#     language: python
#     name: tedana-hack
# ---

# %%
import holoviews as hv
import pandas as pd
import xarray as xr
import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div, Range1d
from bokeh.events import Tap
from bokeh.embed import components
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
import hvplot.xarray
from tedana.reporting import generate_report

from sklearn.preprocessing import MinMaxScaler
from tedana.utils import get_spectrum
import os.path as osp

hv.extension('bokeh')

# %%
OUTDIR = '/opt/tedana-hack/tedana/docs/DynReports/data/TED.five-echo/'
TR = 2
state2col = {'accepted': '#00ff00', 'rejected': '#ff0000', 'ignored': '#0000ff'}


# %%
def load_comp_ts(comp_ts_DIR):
    meica_mix_Path = osp.join(comp_ts_DIR,'meica_mix.1D')
    meica_mix = np.loadtxt(meica_mix_Path)
    [Nt,Nc]   = meica_mix.shape
    DF = pd.DataFrame(meica_mix)
    DF.columns = ['C'+str(c).zfill(3) for c in np.arange(Nc)]
    DF.reset_index(inplace=True)
    DF.rename(columns={'index':'Volume'}, inplace=True)
    CDS = ColumnDataSource(DF)
    return CDS, Nt, Nc


# %%
def prepare_comp_table(comp_table_DIR):
    """
    Create Bokeh ColumnDataSource with all info dynamic plots need

    Parameters
    ----------
    comp_table_DIR: str
        tedana output directory where to find comp_table
    
    Returns
    -------
    CompTable_CDS: bokeh.models.ColumnDataSource
        Data structure with all the fields to plot or hover over
    Nc: int
        Number of components
    """
    CompTable_Path = osp.join(comp_table_DIR, 'comp_table_ica.tsv')
    DF = pd.read_csv(CompTable_Path, sep='\t')
    Nc = DF.shape[0]
    # When needed, remove space from column names (bokeh is not happy about it)
    DF.rename(columns={'variance explained': 'var_exp'}, inplace=True)

    # For providing sizes based on Var Explained that are visible
    mm_scaler = MinMaxScaler(feature_range=(4, 20))
    DF['var_exp_size'] = mm_scaler.fit_transform(
        DF[['var_exp', 'normalized variance explained']])[:, 0]

    # Ranks
    DF['rho_rank'] = DF['rho'].rank(ascending=False).values
    DF['kappa_rank'] = DF['kappa'].rank(ascending=False).values
    DF['var_exp_rank'] = DF['var_exp'].rank(ascending=False).values
    # Remove unsed columns to decrease size of final HTML
    DF.drop(['normalized variance explained', 'countsigFR2', 'countsigFS0', 'dice_FS0',
             'countnoise', 'dice_FR2', 'signal-noise_t', 'signal-noise_p',
             'd_table_score', 'kappa ratio', 'rationale', 'd_table_score_scrub'],
            axis=1, inplace=True)

    # Create additional Column with colors based on final classification
    DF['color'] = [state2col[i] for i in DF['classification']]

    CompTable_CDS = ColumnDataSource(data=dict(
        kappa=DF['kappa'],
        rho=DF['rho'],
        varexp=DF['var_exp'],
        kappa_rank = DF['kappa_rank'],
        rho_rank = DF['rho_rank'],
        varexp_rank = DF['var_exp_rank'],
        component=[str(i) for i in DF['component']],
        color=DF['color'],
        size=DF['var_exp_size'],
        classif=DF['classification']))

    return CompTable_CDS, Nc


# %%
def tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, div):
    """
    Javacript function to animate tap events and show component info on the right

    Parameters
    ----------
    CDS: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    div: bokeh.models.Div
        Target Div element where component images will be loaded
    
    Returns
    -------
    CustomJS: bokeh.models.CustomJS
        Javascript function that adds the tapping functionality
    """
    return CustomJS(args=dict(source_comp_table=CDS_comp_table, source_meica_ts=CDS_meica_ts, 
                              source_tsplot=CDS_TSplot, 
                              source_meica_fft=CDS_meica_fft, 
                              source_fftplot=CDS_FFTplot,
                              div=div_content, outdir=OUTDIR), code="""
    // Accessing the selected component ID
    var data     = source_comp_table.data;
    var selected = source_comp_table.selected.indices;
    var selected_padded = '' + selected;
    while (selected_padded.length < 3) {
        selected_padded = '0' + selected_padded;
    }
    // Creating a new version 000 --> C000
    var selected_padded_C = 'C' + selected_padded
    
    // Updating TS Plot
    var Plot_TS = source_tsplot.data;
    var TS_x    = Plot_TS['x']
    var TS_y    = Plot_TS['y']
    var Comp_TS   = source_meica_ts.data;
    var Comp_TS_y = Comp_TS[selected_padded_C]
    
    for (var i = 0; i < TS_x.length; i ++) {
        TS_y[i] = Comp_TS_y[i]
    }
    source_tsplot.change.emit();
    
    // Updating FFT Plot
    var Plot_FFT = source_fftplot.data;
    var FFT_x = Plot_FFT['x']
    var FFT_y = Plot_FFT['y']
    var Comp_FFT = source_meica_fft.data;
    var Comp_FFT_y = Comp_FFT[selected_padded_C]
    for (var i = 0; i < FFT_x.length; i ++) {
        FFT_y[i] = Comp_FFT_y[i]
    }
    source_fftplot.change.emit();
    
    
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
def create_krPlot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, div):
    """
    Create Dymamic Kappa/Rho Scatter Plot

    Parameters
    ----------
    CDS_comp_table: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    div: bokeh.models.Div
        Target Div element where component images will be loaded
    
    Returns
    -------
    krFig: bokeh.plotting.figure.Figure
        Bokeh scatter plot of kappa vs. rho
    """
    # Create Panel for the Kappa - Rho Scatter
    kr_hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa'),
                                       ('Rho', '@rho'), ('Var. Expl.', '@varexp')])
    krFig = figure(plot_width=400, plot_height=400,
                   tools=["tap,wheel_zoom,reset,pan,crosshair", kr_hovertool],
                   title="Kappa / Rho Plot")
    krFig.circle('kappa', 'rho', size='size', color='color', alpha=0.5, source=CDS_comp_table,
                 legend_group='classif')
    krFig.xaxis.axis_label = 'Kappa'
    krFig.yaxis.axis_label = 'Rho'
    krFig.toolbar.logo = None
    krFig.legend.background_fill_alpha = 0.5
    krFig.legend.orientation = 'horizontal'
    krFig.legend.location = 'bottom_right'
    krFig.js_on_event(Tap, tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, div))
    return krFig


# %%
def create_ksortedPlot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div):
    """
    Create Dymamic Sorted Kappa Plot

    Parameters
    ----------
    CDS_comp_table: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    div: bokeh.models.Div
        Target Div element where component images will be loaded
    
    Returns
    -------
    ksorted_Fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by kappa
    """
    # Create Panel for the Ranked Kappa Plot
    ksorted_hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa'),
                                            ('Rho', '@rho'), ('Var. Expl.', '@varexp')])
    ksorted_Fig = figure(plot_width=400, plot_height=400,
                         tools=["tap,wheel_zoom,reset,pan,crosshair", ksorted_hovertool],
                         title="Components sorted by Kappa")
    ksorted_Fig.circle('kappa_rank', 'kappa', source=CDS_comp_table, size=3, color='color')
    ksorted_Fig.xaxis.axis_label = 'Kappa Rank'
    ksorted_Fig.yaxis.axis_label = 'Kappa'
    ksorted_Fig.x_range = Range1d(-1, Nc + 1)
    ksorted_Fig.toolbar.logo = None
    ksorted_Fig.js_on_event(Tap, tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, div))

    return ksorted_Fig


# %%
def create_rho_sortedPlot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div):
    """
    Create Dymamic Sorted Kappa Plot

    Parameters
    ----------
    CDS_comp_table: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    div: bokeh.models.Div
        Target Div element where component images will be loaded
    
    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by kappa
    """
    # Create Panel for the Ranked Kappa Plot
    hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa'),
                                            ('Rho', '@rho'), ('Var. Expl.', '@varexp')])
    fig = figure(plot_width=400, plot_height=400,
                         tools=["tap,wheel_zoom,reset,pan,crosshair", hovertool],
                         title="Components sorted by Rho")
    fig.circle('rho_rank', 'rho', source=CDS_comp_table, size=3, color='color')
    fig.xaxis.axis_label = 'Rho Rank'
    fig.yaxis.axis_label = 'Rho'
    fig.x_range = Range1d(-1, Nc + 1)
    fig.toolbar.logo = None
    fig.js_on_event(Tap, tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, div))

    return fig


# %%
def create_varexp_sortedPlot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div):
    """
    Create Dymamic Sorted Kappa Plot

    Parameters
    ----------
    CDS_comp_table: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    div: bokeh.models.Div
        Target Div element where component images will be loaded
    
    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by kappa
    """
    # Create Panel for the Ranked Kappa Plot
    hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa'),
                                            ('Rho', '@rho'), ('Var. Expl.', '@varexp')])
    fig = figure(plot_width=400, plot_height=400,
                         tools=["tap,wheel_zoom,reset,pan,crosshair", hovertool],
                         title="Components sorted by Variance Explained")
    fig.circle('varexp_rank', 'varexp', source=CDS_comp_table, size=3, color='color')
    fig.xaxis.axis_label = 'Variance Rank'
    fig.yaxis.axis_label = 'Variance'
    fig.x_range = Range1d(-1, Nc + 1)
    fig.toolbar.logo = None
    fig.js_on_event(Tap, tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, div))

    return fig


# %%
def generate_spectrum_CDS(CDS_meica_mix,TR,Nc):
    spectrum, freqs = get_spectrum(CDS_meica_mix.data['C000'], TR)
    Nf = spectrum.shape[0]
    DF = pd.DataFrame(columns=['C'+str(c).zfill(3) for c in np.arange(Nc)],index=np.arange(Nf))
    for c in np.arange(Nc):
        cid = 'C' + str(c).zfill(3)
        ts = CDS_meica_mix.data[cid]
        spectrum, freqs = get_spectrum(ts, 2)
        DF[cid] = spectrum
    DF['Freq'] = freqs
    CDS = ColumnDataSource(DF)
    return CDS,Nf


# %%
def create_ts_plot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nt, Nc):
    fig    = figure(plot_width=800, plot_height=200,
                         tools=["tap,wheel_zoom,reset,pan,crosshair", HoverTool(tooltips=[('x','@x'),('y','@y')])],
                         title="Component Time Series")
    fig.line('x','y',source=CDS_TSplot, line_color='black',line_width=3)
    fig.xaxis.axis_label='Time [Volume]'
    fig.yaxis.axis_label='Signal'
    fig.toolbar.logo = None
    fig.toolbar_location='above'
    fig.x_range = Range1d(0, Nt)
    return fig


# %%
def create_fft_plot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nt, Nc,Nf):
    fig    = figure(plot_width=800, plot_height=200,
                         tools=["tap,wheel_zoom,reset,pan,crosshair", HoverTool(tooltips=[('x','@x'),('y','@y')])],
                         title="Component Spectrum")
    fig.line('x','y',source=CDS_FFTplot, line_color='black',line_width=3)
    fig.xaxis.axis_label='Frequency [Hz]'
    fig.yaxis.axis_label='Power'
    fig.toolbar.logo = None
    fig.toolbar_location='above'
    fig.x_range = Range1d(0, Nf)
    return fig


# %%

# %%
# Page Genearation
# 1) Load the Comp_table file into a bokeh CDS
[CDS_CompTable, Nc] = prepare_comp_table(OUTDIR)
# 2) Load the Component Timeseries into a bokeh CDS
[CDS_meica_mix,Nt, Nc] = load_comp_ts(OUTDIR)
# 3) Generate the Component Spectrum and store it into a bokeh CDS
[CDS_meica_fft,Nf] = generate_spectrum_CDS(CDS_meica_mix,TR,Nc)
# 4) Generate a Flat Line into a bokeh CDS (this is what get plotted in the TS graph)
CDS_TSplot = ColumnDataSource(data=dict(x=np.arange(Nt),y=np.zeros(Nt,)))
# 5) Generate a Flat Line into a bokeh CDS (this is what get plotted in the FFT graph)
CDS_FFTplot = ColumnDataSource(data=dict(x=np.arange(Nt),y=np.zeros(Nt,)))
# 6) DIV
div_content = Div(width=600, height=900, height_policy='fixed')
# -------------------------------------------------------------------------------------
# 7) Create the Kappa/Rho Scatter Plot
kappa_rho_plot = create_krPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, div_content)
# 8) Create the Ranked Kappa Plot
kappa_sorted_plot = create_ksortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div_content)
# 9) Create the Ranked Rho Plot
rho_sorted_plot = create_rho_sortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div_content)
# 10) Create the Ranked Variance Explained Plot
varexp_sorted_plot = create_varexp_sortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div_content)
# 11) Create the Component Timeseries Plot
ts_plot = create_ts_plot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nt, Nc)
# 12) Create the Component FFT Plot
fft_plot = create_fft_plot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nt, Nc,Nf)
# 13) Create a layout
app = column(row(kappa_rho_plot, kappa_sorted_plot, rho_sorted_plot, varexp_sorted_plot), row(ts_plot,fft_plot), div_content)
# 14) Create Script and Div
(kr_script, kr_div) = components(app)
# 15) Embed into Report Template
generate_report(kr_div, kr_script, file_path='/opt/report_v2.html')


# %% [markdown]
# ***
