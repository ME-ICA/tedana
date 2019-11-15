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
hv.extension('bokeh')

# %%
import pandas as pd
import numpy as np
from os.path import join as opj
from math import pi
from sklearn.preprocessing import MinMaxScaler
from bokeh import (embed, events, layouts, models, plotting, transform)

from tedana import (io, reporting, utils)
from tedana.io import load_comptable

color_mapping = {'accepted': '#2ecc71', 
                 'rejected': '#e74c3c', 
                 'ignored': '#3498db'}

# %%
OUTDIR = '/opt/tedana-hack/tedana/docs/DynReports/data/TED.five-echo/'
TR = 2


# %%
def _link_figures(fig, comptable_ds,
                  ts_src, fft_src, ts_plot, fft_plot, 
                  ts_line_glyph, fft_line_glyph,
                  div_content, out_dir):
    """
    Links figures and adds interaction on mouse-click.

    Parameters
    ----------
    fig : bokeh.plotting.figure
        Figure containing a given plot

    comptable_ds : bokeh.models.ColumnDataSource
        Data structure with a limited version of the comptable
        suitable for dynamic plot purposes

    ts_src : bokeh.models.ColumnDataSource
        Data structure containing the timeseries for all
        components

    fft_src : bokeh.models.ColumnDataSource
        Data structure containing the fft for all components

    ts_plot : bokeh.models.ColumnDataSource
        Data structure that contains the timeseries being
        plotted at a given moment.

    fft_plot : bokeh.models.ColumnDataSource
        Data structure that contains the fft being plotted
        at a given moment in time
        
    ts_line_glyph : bokeh.models.Line
        Link to the line element of the time series plot.
        Needed to update the color of the timeseries trace.
    
    fft_line_glyph : bokeh.models.Line
        Link to the line element of the fft plot.
        Needed to update the color of the fft trace.

    div_content : bokeh.models.Div
        Div element for additional HTML content.

    out_dir : str
        Output directory of tedana results.

    Returns
    -------
    fig : bokeh.plotting.figure
        Same as input figure, but with a linked method to
        its Tap event.
    
    """
    fig.js_on_event(events.Tap,
                    tap_callback(comptable_ds,
                                ts_src,
                                fft_src,
                                ts_plot,
                                fft_plot,
                                ts_line_glyph,
                                fft_line_glyph,
                                out_dir))
    return fig


# %%
def _create_ts_plot(n_vol):
    """
    Generates a bokeh line plot for the time series of a given component

    Parameters
    ----------
    n_vols: float
        Number of acquisitions

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot to show a given component time series
    
    line_glyph: bokeh.models.Line
        Link to the line element of the time series plot.
    
    ts_cds: bokeh.models.ColumnDataSource
        Data structure that contains the timeseries being
        plotted at a given moment.
    """
    
    ts_cds = models.ColumnDataSource(data=dict(x=np.arange(n_vol),
                                               y=np.zeros(n_vol,)))
    
    fig = plotting.figure(plot_width=800, plot_height=200,
                 tools=["wheel_zoom,box_zoom,reset,pan,crosshair,save",
                        models.HoverTool(tooltips=[('Volume', '@x{0.00}'), ('Signal', '@y{0.00}')])],
                 title="Component Time Series")
    line_glyph = models.Line(x='x', y='y', line_color='#000000', line_width=3)
    fig.add_glyph(ts_cds, line_glyph)
    fig.xaxis.axis_label = 'Time [Volume]'
    fig.yaxis.axis_label = 'Signal'
    fig.toolbar.logo = None
    fig.toolbar_location = 'above'
    fig.x_range = models.Range1d(0, n_vol)
    return fig, line_glyph, ts_cds


# %%
def _create_fft_plot(freqs):
    """
    Generates a bokeh line plot for the spectrum of a given component

    Parameters
    ----------
    freqs: np.darray
        List of frequencies for which spectral amplitude values are
        available

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot to show a given component spectrum
        
    line_glyph: bokeh.models.Line
        Link to the line element of the fft plot.
    
    ts_cds: bokeh.models.ColumnDataSource
        Data structure that contains the fft being
        plotted at a given moment.
    """
    Nf = len(freqs)
    max_freq = np.max(freqs)
    fft_cds = models.ColumnDataSource(data=dict(x=freqs, y=np.zeros(Nf,)))
    fig = plotting.figure(plot_width=800, plot_height=200,
                 tools=["wheel_zoom,box_zoom,reset,pan,crosshair,save",
                        models.HoverTool(tooltips=[('Freq.', '@x{0.000} Hz'), ('Power', '@y{0.00}')])],
                 title="Component Spectrum")
    line_glyph = models.Line(x='x', y='y', line_color='#000000', line_width=3)
    fig.add_glyph(fft_cds, line_glyph)
    fig.xaxis.axis_label = 'Frequency [Hz]'
    fig.yaxis.axis_label = 'Power'
    fig.toolbar.logo = None
    fig.toolbar_location = 'above'
    fig.x_range = models.Range1d(0, max_freq)
    return fig, line_glyph, fft_cds


# %%
def _spectrum_data_src(CDS_meica_mix, tr, n_comps):
    """
    Computes FFT for all time series and puts them on a bokeh.models.ColumnDataSource

    Parameters
    ----------
    CDS_meica_mix: bokeh.models.ColumnDataSource
        Contains the time series for all components
    tr: int
        Data Repetition Time in seconds
    n_comps: int
        Number of components

    Returns
    -------
    cds: bokeh.models.ColumnDataSource
        Contains the spectrum for all time series
    freqs: np.darray
        List of frequencies for which spectral amplitude values are
        available
    """
    spectrum, freqs = utils.get_spectrum(CDS_meica_mix.data['ica_00'], tr)
    n_freqs = spectrum.shape[0]
    df = pd.DataFrame(columns=['ica_' + str(c).zfill(2) for c in np.arange(n_comps)], 
                      index=np.arange(n_freqs))
    for c in np.arange(Nc):
        cid = 'ica_' + str(c).zfill(2)
        ts = CDS_meica_mix.data[cid]
        spectrum, freqs = utils.get_spectrum(ts, tr)
        df[cid] = spectrum
    cds = models.ColumnDataSource(df)
    return cds, freqs


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
    file_path = opj(OUTDIR,'ica_mixing.tsv')
    DF = pd.read_csv(file_path, sep='\t')
    [Nt,Nc] = DF.shape
    DF['Volume'] = np.arange(Nt)
    CDS = models.ColumnDataSource(DF)
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
    comptable_path = opj(OUTDIR,'ica_decomposition.json')
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
    DF['color'] = [color_mapping[i] for i in DF['classification']]
    DF['component'] = np.arange(Nc)
    # Re-sort for Pie
    DF['angle']=DF['var_exp']/DF['var_exp'].sum() * 2*pi
    DF.sort_values(by=['classification','var_exp'], inplace=True)
    CDS = models.ColumnDataSource(data=dict(
        kappa=DF['kappa'],
        rho=DF['rho'],
        varexp=DF['var_exp'],
        kappa_rank=DF['kappa_rank'],
        rho_rank=DF['rho_rank'],
        varexp_rank=DF['var_exp_rank'],
        component=[str(i) for i in DF['component']],
        color=DF['color'],
        size=DF['var_exp_size'],
        classif=DF['classification'],
        angle=DF['angle']))
    return CDS, Nc


# %%
tap_callback_jscode = """
    // Accessing the selected component ID
    var data          = source_comp_table.data;
    var selected_idx = source_comp_table.selected.indices;
    if(selected_idx > 0) {
        // A component has been selected
        // -----------------------------
        var components = data['component']
        var selected = components[selected_idx]
        var selected_padded = '' + selected;
        while (selected_padded.length < 2) {
            selected_padded = '0' + selected_padded;
        }
        var selected_padded_forIMG = '0' + selected_padded
        var selected_padded_C = 'ica_' + selected_padded
        
        // Find color for selected component
        var colors = data['color']
        var this_component_color = colors[selected_idx]
        
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
        
        // Image Below Plots
        div.text = ""
        var line = "<span><img src='" + outdir + "/figures/comp_"+selected_padded_forIMG+".png'" +
            " alt='Component Map' height=1000 width=900><span>\\n";
        console.log('Linea: ' + line)
        var text = div.text.concat(line);
        var lines = text.split("\\n")
            if (lines.length > 35)
                lines.shift();
        div.text = lines.join("\\n");
    
    } else {
        // No component has been selected
        // ------------------------------
        // Set Component color to Black
        var this_component_color = '#000000'
        
        // Update time series line color
        ts_line.line_color = this_component_color;

        // Update spectrum line color
        fft_line.line_color = this_component_color;
        
        // Updating TS Plot
        var Plot_TS = source_tsplot.data;
        var TS_x    = Plot_TS['x']
        var TS_y    = Plot_TS['y']
        for (var i = 0; i < TS_x.length; i ++) {
            TS_y[i] = 0
        }
        source_tsplot.change.emit();
        
        // Updating FFT Plot
        var Plot_FFT = source_fftplot.data;
        var FFT_x = Plot_FFT['x']
        var FFT_y = Plot_FFT['y']
        for (var i = 0; i < FFT_x.length; i ++) {
            FFT_y[i] = 0
        }
        source_fftplot.change.emit();
        
        // Image Below Plots
        div.text = "\\n"

    }
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
    return models.CustomJS(args=dict(source_comp_table=CDS_comp_table,
                              source_meica_ts=CDS_meica_ts,
                              source_tsplot=CDS_TSplot,
                              source_meica_fft=CDS_meica_fft,
                              source_fftplot=CDS_FFTplot,
                              div=div_content,
                              ts_line=ts_line_glyph,
                              fft_line=fft_line_glyph,
                              outdir=OUTDIR), code=tap_callback_jscode)


# %%
def create_krPlot(CDS_comp_table):
    """
    Create Dymamic Kappa/Rho Scatter Plot

    Parameters
    ----------
    CDS_comp_table: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh scatter plot of kappa vs. rho
    """
    # Create Panel for the Kappa - Rho Scatter
    kr_hovertool = models.HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa{0.00}'),
                                       ('Rho', '@rho{0.00}'), ('Var. Expl.', '@varexp{0.00}%')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                 tools=["tap,wheel_zoom,reset,pan,crosshair,save", kr_hovertool],
                 title="Kappa / Rho Plot")
    fig.circle('kappa', 'rho', size='size', color='color', alpha=0.5, source=CDS_comp_table,
               legend_group='classif')
    fig.xaxis.axis_label = 'Kappa'
    fig.yaxis.axis_label = 'Rho'
    fig.toolbar.logo = None
    fig.legend.background_fill_alpha = 0.5
    fig.legend.orientation = 'horizontal'
    fig.legend.location = 'bottom_right'
    return fig


# %%
def create_ksortedPlot(CDS_comp_table, Nc):
    """
    Create Dymamic Sorted Kappa Plot

    Parameters
    ----------
    CDS_comp_table: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by kappa
    """
    # Create Panel for the Ranked Kappa Plot
    ksorted_hovertool = models.HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa{0.00}'),
                                            ('Rho', '@rho{0.00}'), ('Var. Expl.', '@varexp{0.00}%')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                 tools=["tap,wheel_zoom,reset,pan,crosshair,save", ksorted_hovertool],
                 title="Components sorted by Kappa")
    fig.line(x=np.arange(1, Nc + 1),
             y=CDS_CompTable.data['kappa'].sort_values(ascending=False).values,
             color='black')
    fig.circle('kappa_rank', 'kappa', source=CDS_comp_table,
               size=5, color='color', alpha=0.7)
    fig.xaxis.axis_label = 'Kappa Rank'
    fig.yaxis.axis_label = 'Kappa'
    fig.x_range = models.Range1d(-1, Nc + 1)
    fig.toolbar.logo = None

    return fig


# %%
def create_rho_sortedPlot(CDS_comp_table, Nc):
    """
    Create Dymamic Sorted Kappa Plot

    Parameters
    ----------
    CDS_comp_table: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by kappa
    """
    # Create Panel for the Ranked Kappa Plot
    hovertool = models.HoverTool(tooltips=[('Component ID', '@component'), ('Kappa', '@kappa{0.00}'),
                                    ('Rho', '@rho{0.00}'), ('Var. Expl.', '@varexp{0.00}%')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                 tools=["tap,wheel_zoom,reset,pan,crosshair,save", hovertool],
                 title="Components sorted by Rho")
    fig.line(x=np.arange(1, Nc + 1),
             y=CDS_CompTable.data['rho'].sort_values(ascending=False).values,
             color='black')
    fig.circle('rho_rank', 'rho', source=CDS_comp_table,
               size=5, color='color', alpha=0.7)
    fig.xaxis.axis_label = 'Rho Rank'
    fig.yaxis.axis_label = 'Rho'
    fig.x_range = models.Range1d(-1, Nc + 1)
    fig.toolbar.logo = None

    return fig


# %%
def create_varexp_piePlot(CDS_comp_table, Nc):
    fig = plotting.figure(plot_width=400, plot_height=400, title='Variance Explained View', 
                 tools=['hover,tap,save'], 
                 tooltips=[('Component ID','@component'),
                           ('Kappa','@kappa{0.00}'),
                           ('Rho','@rho{0.00}'),
                           ('Var. Exp.','@varexp{0.00}%')])
    fig.wedge(x=0,y=1,radius=.9,
              start_angle=transform.cumsum('angle', include_zero=True),
              end_angle=transform.cumsum('angle'),
              line_color="white",
              fill_color='color', source=CDS_CompTable, fill_alpha=0.7)
    fig.axis.visible=False
    fig.grid.visible=False
    fig.toolbar.logo=None    
        
    circle = models.Circle(x=0,y=1,size=150, fill_color='white', line_color='white')
    fig.add_glyph(circle)

    return fig


# %%
# LOAD ALL NECESSARY INFORMATION
# 1) Load the Comp_table file into a bokeh CDS
[CDS_CompTable, Nc] = load_comp_table(OUTDIR)
# 2) Load the Component Timeseries into a bokeh CDS
[CDS_meica_mix, n_vols, Nc] = load_comp_ts(OUTDIR)
# 3) Generate the Component Spectrum and store it into a bokeh CDS
[CDS_meica_fft, freqs] = _spectrum_data_src(CDS_meica_mix, TR, Nc)

# GENERATE CDS NECESSARY FOR LIVE UPDATE OF PLOTS
# 4) Save flat Line into a bokeh CDS (this is what get plotted in the TS graph)
#CDS_TSplot = models.ColumnDataSource(data=dict(x=np.arange(Nt), y=np.zeros(Nt,)))
# 5) Save flat Line into a bokeh CDS (this is what get plotted in the FFT graph)
#CDS_FFTplot = models.ColumnDataSource(data=dict(x=CDS_meica_fft.data['Freq'], y=np.zeros(Nf,)))

# CREATE ALL GRAPHIC ELEMENTS
# 6) Create a DIV element
div_content = models.Div(width=600, height=900, height_policy='fixed')
# 7) Create the Component Timeseries Plot
[ts_plot, ts_line_glyph, CDS_TSplot] = _create_ts_plot(n_vols)
# 8) Create the Component FFT Plot
[fft_plot, fft_line_glyph, CDS_FFTplot] = _create_fft_plot(freqs)
# 9) Create the Kappa/Rho Scatter Plot
kappa_rho_plot = create_krPlot(CDS_CompTable)
# 10) Create the Ranked Kappa Plot
kappa_sorted_plot = create_ksortedPlot(CDS_CompTable, Nc)
# 11) Create the Ranked Rho Plot
rho_sorted_plot = create_rho_sortedPlot(CDS_CompTable, Nc)
# 12) Create the Ranked Variance Explained Plot
varexp_pie_plot = create_varexp_piePlot(CDS_CompTable, Nc)

for fig in [kappa_rho_plot,kappa_sorted_plot,rho_sorted_plot,varexp_pie_plot]:
    _link_figures(fig,
                  CDS_CompTable,
                  CDS_meica_mix,
                  CDS_meica_fft,
                  CDS_TSplot,
                  CDS_FFTplot, ts_line_glyph, fft_line_glyph,
                  div_content, OUTDIR)

# 13) Create a layout
app = layouts.column(layouts.row(kappa_rho_plot, kappa_sorted_plot, rho_sorted_plot, varexp_pie_plot),
             layouts.row(ts_plot, fft_plot),
             div_content)

# CREATE EMBEDDING MATERIALS
# 14) Create Script and Div
(kr_script, kr_div) = embed.components(app)
# 15) Embed into Report Template
reporting.generate_report(kr_div, kr_script, file_path='/opt/report_v3.html')


# %%
CDS_meica_mix.data

# %%

# %%

# %%

# %%
