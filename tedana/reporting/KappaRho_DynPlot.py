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
import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div, Range1d, Line
from bokeh.events import Tap
from bokeh.embed import components
from bokeh.layouts import row, column
from bokeh.plotting import figure
from tedana.reporting import generate_report
from sklearn.preprocessing import MinMaxScaler
from tedana.utils import get_spectrum
import os.path as osp
from math import pi
from tedana.io import load_comptable
hv.extension('bokeh')

# %%
OUTDIR = '/opt/tedana-hack/tedana/docs/DynReports/data/TED.five-echo/'
TR = 2
state2col = {'accepted': '#00ff00', 'rejected': '#ff0000', 'ignored': '#0000ff'}


# %%
def load_comp_ts(out_dir):
    """
    Load Component Timeseries (meica_mix.1D) into a bokeh.ColumnDataSource

    Parameters
    ----------
    out_dir: str
        tedana output directory where to find comp_table

    Returns
    -------
    CDS: bokeh.models.ColumnDataSource
        Contains the component time series
    Nt: int
        Number of Time points
    Nc: int
        Number of components
    """
    file_path = osp.join(OUTDIR,'ica_mixing.tsv')
    DF = pd.read_csv(file_path, sep='\t')
    [Nt,Nc] = DF.shape
    DF['Volume'] = np.arange(Nt)
    CDS = ColumnDataSource(DF)
    return CDS, Nt, Nc


# %%
def load_comp_table(out_dir):
    """
    Create Bokeh ColumnDataSource with all info dynamic plots need

    Parameters
    ----------
    out_dir: str
        tedana output directory where to find comp_table

    Returns
    -------
    CDS: bokeh.models.ColumnDataSource
        Data structure with all the fields to plot or hover over
    Nc: int
        Number of components
    """
    comptable_path = osp.join(OUTDIR,'ica_decomposition.json')
    DF = pd.read_json(comptable_path)
    DF.drop('Description', axis=0, inplace=True)
    DF.drop('Method', axis=1, inplace=True)
    DF = DF.T

    Nc = DF.shape[0]
    # When needed, remove space from column names (bokeh is not happy about it)
    DF.rename(columns={'variance explained': 'var_exp'}, inplace=True)
    # For providing sizes based on Var Explained that are visible
    mm_scaler = MinMaxScaler(feature_range=(4, 20))
    DF['var_exp_size'] = mm_scaler.fit_transform(DF[['var_exp', 'normalized variance explained']])[:, 0]
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
    DF['component'] = np.arange(Nc)
    
    CDS = ColumnDataSource(data=dict(
        kappa=DF['kappa'],
        rho=DF['rho'],
        varexp=DF['var_exp'],
        kappa_rank=DF['kappa_rank'],
        rho_rank=DF['rho_rank'],
        varexp_rank=DF['var_exp_rank'],
        component=[str(i) for i in DF['component']],
        color=DF['color'],
        size=DF['var_exp_size'],
        classif=DF['classification']))
    return CDS, Nc


# %%
def generate_spectrum_CDS(CDS_meica_mix, TR, Nc):
    """
    Computes FFT for all time series and puts them on a bokeh.models.ColumnDataSource

    Parameters
    ----------
    CDS_meica_mix: bokeh.models.ColumnDataSource
        Contains the time series for all components
    TR: int
        Data Repetition Time in seconds
    Nc: int
        Number of components

    Returns
    -------
    CDS: bokeh.models.ColumnDataSource
        Contains the spectrum for all time series
    Nf: int
        Number of frequency points
    """
    spectrum, freqs = get_spectrum(CDS_meica_mix.data['ica_00'], TR)
    Nf = spectrum.shape[0]
    DF = pd.DataFrame(columns=['ica_' + str(c).zfill(2) for c in np.arange(Nc)], index=np.arange(Nf))
    for c in np.arange(Nc):
        cid = 'ica_' + str(c).zfill(2)
        ts = CDS_meica_mix.data[cid]
        spectrum, freqs = get_spectrum(ts, 2)
        DF[cid] = spectrum
    DF['Freq'] = freqs
    CDS = ColumnDataSource(DF)
    return CDS, Nf


# %%
tap_callback_jscode = """
    // Accessing the selected component ID
    var data     = source_comp_table.data;
    var selected = source_comp_table.selected.indices;
    var selected_padded = '' + selected;
    while (selected_padded.length < 2) {
        selected_padded = '0' + selected_padded;
    }
    // Creating a new version 00 --> ica_00
    var selected_padded_C = 'ica_' + selected_padded

    // Find color for selected component
    var colors = data['color']
    var this_component_color = colors[selected]
    // var ts_line_color = ts_line.line_color;

    // Update time series line color
    ts_line.line_color = this_component_color;

    // Update spectrum line color
    fft_line.line_color = this_component_color;

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
    """


def tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot,
                 CDS_FFTplot, ts_line_glyph, fft_line_glyph, div):
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
    return CustomJS(args=dict(source_comp_table=CDS_comp_table,
                              source_meica_ts=CDS_meica_ts,
                              source_tsplot=CDS_TSplot,
                              source_meica_fft=CDS_meica_fft,
                              source_fftplot=CDS_FFTplot,
                              div=div_content,
                              ts_line=ts_line_glyph,
                              fft_line=fft_line_glyph,
                              outdir=OUTDIR), code=tap_callback_jscode)


# %%
def create_krPlot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot,
                  CDS_FFTplot, ts_line_glyph, fft_line_glyph, div):
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
    fig: bokeh.plotting.figure.Figure
        Bokeh scatter plot of kappa vs. rho
    """
    # Create Panel for the Kappa - Rho Scatter
    kr_hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa'),
                                       ('Rho', '@rho'), ('Var. Expl.', '@varexp')])
    fig = figure(plot_width=400, plot_height=400,
                 tools=["tap,wheel_zoom,reset,pan,crosshair", kr_hovertool],
                 title="Kappa / Rho Plot")
    fig.circle('kappa', 'rho', size='size', color='color', alpha=0.5, source=CDS_comp_table,
               legend_group='classif')
    fig.xaxis.axis_label = 'Kappa'
    fig.yaxis.axis_label = 'Rho'
    fig.toolbar.logo = None
    fig.legend.background_fill_alpha = 0.5
    fig.legend.orientation = 'horizontal'
    fig.legend.location = 'bottom_right'
    fig.js_on_event(Tap, tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft,
                    CDS_TSplot, CDS_FFTplot, ts_line_glyph, fft_line_glyph, div))
    return fig


# %%
def create_ksortedPlot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot,
                       CDS_FFTplot, ts_line_glyph, fft_line_glyph, Nc, div):
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
    ksorted_hovertool = HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa'),
                                            ('Rho', '@rho'), ('Var. Expl.', '@varexp')])
    fig = figure(plot_width=400, plot_height=400,
                 tools=["tap,wheel_zoom,reset,pan,crosshair", ksorted_hovertool],
                 title="Components sorted by Kappa")
    fig.line(x=np.arange(1, Nc + 1),
             y=CDS_CompTable.data['kappa'].sort_values(ascending=False).values,
             color='black')
    fig.circle('kappa_rank', 'kappa', source=CDS_comp_table,
               size=5, color='color', alpha=0.7)
    fig.xaxis.axis_label = 'Kappa Rank'
    fig.yaxis.axis_label = 'Kappa'
    fig.x_range = Range1d(-1, Nc + 1)
    fig.toolbar.logo = None
    fig.js_on_event(Tap, tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft,
                                      CDS_TSplot, CDS_FFTplot, ts_line_glyph,
                                      fft_line_glyph, div))

    return fig


# %%
def create_rho_sortedPlot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot,
                          CDS_FFTplot, ts_line_glyph, fft_line_glyph, Nc, div):
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
    fig.line(x=np.arange(1, Nc + 1),
             y=CDS_CompTable.data['rho'].sort_values(ascending=False).values,
             color='black')
    fig.circle('rho_rank', 'rho', source=CDS_comp_table,
               size=5, color='color', alpha=0.7)
    fig.xaxis.axis_label = 'Rho Rank'
    fig.yaxis.axis_label = 'Rho'
    fig.x_range = Range1d(-1, Nc + 1)
    fig.toolbar.logo = None
    fig.js_on_event(Tap, tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft,
                    CDS_TSplot, CDS_FFTplot, ts_line_glyph,
                    fft_line_glyph, div))

    return fig


# %%
def create_varexp_sortedPlot(CDS_comp_table, CDS_meica_ts, CDS_meica_fft, CDS_TSplot,
                             CDS_FFTplot, ts_line_glyph, fft_line_glyph, Nc, div):
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
    fig.line(x=np.arange(1, Nc + 1),
             y=CDS_CompTable.data['varexp'].sort_values(ascending=False).values,
             color='black')
    fig.circle('varexp_rank', 'varexp', source=CDS_comp_table,
               size=5, color='color', alpha=.7)
    fig.xaxis.axis_label = 'Variance Rank'
    fig.yaxis.axis_label = 'Variance'
    fig.x_range = Range1d(-1, Nc + 1)
    fig.toolbar.logo = None
    fig.js_on_event(Tap, tap_callback(CDS_comp_table, CDS_meica_ts, CDS_meica_fft,
                    CDS_TSplot, CDS_FFTplot, ts_line_glyph,
                    fft_line_glyph, div))

    return fig


# %%
def create_ts_plot(CDS_TSplot, Nt):
    """
    Generates a bokeh line plot for the time series of a given component

    Parameters
    ----------
    CDS_FFTplot: bokeh.models.ColumnDataSource
        Contains only the data to be plotted at a given moment.

    Nt: float
        Number of acquisitions

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot to show a given component time series
    """
    fig = figure(plot_width=800, plot_height=200,
                 tools=["wheel_zoom,reset,pan,crosshair",
                        HoverTool(tooltips=[('Volume', '@x'), ('Signal', '@y')])],
                 title="Component Time Series")
    line_glyph = Line(x='x', y='y', line_color='#000000', line_width=3)
    fig.add_glyph(CDS_TSplot, line_glyph)
    fig.xaxis.axis_label = 'Time [Volume]'
    fig.yaxis.axis_label = 'Signal'
    fig.toolbar.logo = None
    fig.toolbar_location = 'above'
    fig.x_range = Range1d(0, Nt)
    return fig, line_glyph


# %%
def create_fft_plot(CDS_FFTplot, max_freq):
    """
    Generates a bokeh line plot for the spectrum of a given component

    Parameters
    ----------
    CDS_FFTplot: bokeh.models.ColumnDataSource
        Contains only the data to be plotted at a given moment.

    max_freq: float
        Maximum frequency for which there is information

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot to show a given component spectrum
    """
    fig = figure(plot_width=800, plot_height=200,
                 tools=["wheel_zoom,reset,pan,crosshair",
                        HoverTool(tooltips=[('Freq.', '@x'), ('Power', '@y')])],
                 title="Component Spectrum")
    line_glyph = Line(x='x', y='y', line_color='#000000', line_width=3)
    fig.add_glyph(CDS_FFTplot, line_glyph)
    fig.xaxis.axis_label = 'Frequency [Hz]'
    fig.yaxis.axis_label = 'Power'
    fig.toolbar.logo = None
    fig.toolbar_location = 'above'
    fig.x_range = Range1d(0, max_freq)
    return fig, line_glyph


# %%
# LOAD ALL NECESSARY INFORMATION
# 1) Load the Comp_table file into a bokeh CDS
[CDS_CompTable, Nc] = load_comp_table(OUTDIR)
# 2) Load the Component Timeseries into a bokeh CDS
[CDS_meica_mix, Nt, Nc] = load_comp_ts(OUTDIR)
# 3) Generate the Component Spectrum and store it into a bokeh CDS
[CDS_meica_fft, Nf] = generate_spectrum_CDS(CDS_meica_mix, TR, Nc)

# GENERATE CDS NECESSARY FOR LIVE UPDATE OF PLOTS
# 4) Save flat Line into a bokeh CDS (this is what get plotted in the TS graph)
CDS_TSplot = ColumnDataSource(data=dict(x=np.arange(Nt), y=np.zeros(Nt,)))
# 5) Save flat Line into a bokeh CDS (this is what get plotted in the FFT graph)
CDS_FFTplot = ColumnDataSource(data=dict(x=CDS_meica_fft.data['Freq'], y=np.zeros(Nf,)))

# CREATE ALL GRAPHIC ELEMENTS
# 6) Create a DIV element
div_content = Div(width=600, height=900, height_policy='fixed')
# 7) Create the Component Timeseries Plot
[ts_plot, ts_line_glyph] = create_ts_plot(CDS_TSplot, Nt)
# 8) Create the Component FFT Plot
[fft_plot, fft_line_glyph] = create_fft_plot(CDS_FFTplot, np.max(CDS_meica_fft.data['Freq']))
# 9) Create the Kappa/Rho Scatter Plot
kappa_rho_plot = create_krPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft,
                               CDS_TSplot, CDS_FFTplot, ts_line_glyph, fft_line_glyph, div_content)
# 10) Create the Ranked Kappa Plot
kappa_sorted_plot = create_ksortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft,
                                       CDS_TSplot, CDS_FFTplot, ts_line_glyph,
                                       fft_line_glyph, Nc, div_content)
# 11) Create the Ranked Rho Plot
rho_sorted_plot = create_rho_sortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft,
                                        CDS_TSplot, CDS_FFTplot, ts_line_glyph,
                                        fft_line_glyph, Nc, div_content)
# 12) Create the Ranked Variance Explained Plot
varexp_sorted_plot = create_varexp_sortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft,
                                              CDS_TSplot, CDS_FFTplot, ts_line_glyph,
                                              fft_line_glyph, Nc, div_content)
# 13) Create a layout
app = column(row(kappa_rho_plot, kappa_sorted_plot, rho_sorted_plot, varexp_sorted_plot),
             row(ts_plot, fft_plot),
             div_content)

# CREATE EMBEDDING MATERIALS
# 14) Create Script and Div
(kr_script, kr_div) = components(app)
# 15) Embed into Report Template
generate_report(kr_div, kr_script, file_path='/opt/report_v2.html')


# %%
