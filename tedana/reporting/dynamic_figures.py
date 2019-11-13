import pandas as pd
import numpy as np
from os.path import join as opj
from sklearn.preprocessing import MinMaxScaler
from bokeh import (embed, events, layouts, models, plotting)

from tedana.io import load_comp_ts
from tedana.utils import get_spectrum
from tedana.reporting import generate_report

TR = 2

color_mapping = {'accepted': '#00ff00',
                 'rejected': '#ff0000',
                 'ignored': '#0000ff'}


def _create_ts_plot(n_vol):
    """
    Creates timeseries Bokeh plot.

    Parameters
    ----------
    n_vol : int

    Returns
    -------
    fig
    """
    ts_cds = models.ColumnDataSource(data=dict(x=np.arange(n_vol),
                                               y=np.zeros(n_vol,)))
    fig = plotting.figure(plot_width=800, plot_height=200,
                          tools=["tap,wheel_zoom,reset,pan,crosshair",
                                 models.HoverTool(tooltips=[('x', '@x'), ('y', '@y')])],
                          title="Component Time Series")
    fig.line('x', 'y', source=ts_cds, line_color='black', line_width=3)
    fig.xaxis.axis_label = 'Time [Volume]'
    fig.yaxis.axis_label = 'Signal'
    fig.toolbar.logo = None
    fig.toolbar_location = 'above'
    fig.x_range = models.Range1d(0, n_vol)
    return fig


def _create_fft_plot(n_vol, Nf):
    """
    Creates FFT Bokeh plot.

    Parameters
    ----------
    n_vol : int
        Number of volumes in the time series
    Nf :

    Returns
    -------
    fig
    """
    fft_cds = models.ColumnDataSource(data=dict(x=np.arange(Nt),
                                                y=np.zeros(Nt,)))
    fig = plotting.figure(plot_width=800, plot_height=200,
                          tools=["tap,wheel_zoom,reset,pan,crosshair",
                                 models.HoverTool(tooltips=[('x', '@x'), ('y', '@y')])],
                          title="Component Spectrum")
    fig.line('x', 'y', source=fft_cds, line_color='black', line_width=3)
    fig.xaxis.axis_label = 'Frequency [Hz]'
    fig.yaxis.axis_label = 'Power'
    fig.toolbar.logo = None
    fig.toolbar_location = 'above'
    fig.x_range = models.Range1d(0, Nf)
    return fig


def create_data_struct(out_dir, color_mapping=color_mapping):
    """
    Create Bokeh ColumnDataSource with all relevant info
    needed for dynamic plots.

    Parameters
    ----------
    out_dir: str
        tedana output directory
    
    Returns
    -------
    comptable_ds: bokeh.models.ColumnDataSource
        Data structure with all the fields to plot or hover over
    """
    unused_cols = ['normalized variance explained',
                   'countsigFR2', 'countsigFS0',
                   'dice_FS0', 'countnoise', 'dice_FR2',
                   'signal-noise_t', 'signal-noise_p',
                   'd_table_score', 'kappa ratio',
                   'rationale', 'd_table_score_scrub']

    comptable = opj(out_dir, 'comp_table_ica.tsv')
    df = pd.read_csv(comptable, sep='\t')
    # remove space from column names
    df.rename(columns={'variance explained': 'var_exp'}, inplace=True)

    # For providing sizes based on var_exp that are visible
    mm_scaler = MinMaxScaler(feature_range=(4, 20))
    df['var_exp_size'] = mm_scaler.fit_transform(
        df[['var_exp', 'normalized variance explained']])[:, 0]

    # Calculate kappa and rho ranks
    df['rho_rank'] = df['rho'].rank(ascending=False).values
    df['kappa_rank'] = df['kappa'].rank(ascending=False).values
    df['var_exp_rank'] = df['var_exp'].rank(ascending=False).values
    
    # Remove unsed columns to decrease size of final HTML
    df.drop(unused_cols, axis=1, inplace=True)

    # Create additional column with colors based on classification
    df['color'] = [color_mapping[i] for i in df['classification']]

    comptable_ds = models.ColumnDataSource(data=dict(
        kappa=df['kappa'],
        rho=df['rho'],
        varexp=df['var_exp'],
        kappa_rank = df['kappa_rank'],
        rho_rank = df['rho_rank'],
        varexp_rank = df['var_exp_rank'],
        component=[str(i) for i in df['component']],
        color=df['color'],
        size=df['var_exp_size'],
        classif=df['classification']))

    return comptable_ds


def create_krPlot(comptable_ds, CDS_meica_ts, CDS_meica_fft,
                  CDS_TSplot, CDS_FFTplot, div, out_dir):
    """
    Create Dymamic Kappa/Rho Scatter Plot

    Parameters
    ----------
    comptable_ds: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    div: bokeh.models.Div
        Target Div element where component images will be loaded
    
    Returns
    -------
    krFig: bokeh.plotting.figure.Figure
        Bokeh scatter plot of kappa vs. rho
    """
    # Create Panel for the Kappa - Rho Scatter
    hovertool = models.HoverTool(tooltips=[('Component ID', '@component'),
                                              ('Kappa', '@kappa'),
                                              ('Rho', '@rho'),
                                              ('Var. Expl.', '@varexp')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                   tools=["tap,wheel_zoom,reset,pan,crosshair", hovertool],
                   title="Kappa / Rho Plot")
    fig.circle('kappa', 'rho', size='size', color='color',
                 alpha=0.5, source=comptable_ds,
                 legend_group='classif')
    fig.xaxis.axis_label = 'Kappa'
    fig.yaxis.axis_label = 'Rho'
    fig.toolbar.logo = None
    fig.legend.background_fill_alpha = 0.5
    fig.legend.orientation = 'horizontal'
    fig.legend.location = 'bottom_right'
    return fig


def create_ksortedPlot(comptable_ds):
    """
    Create Dymamic Sorted Kappa Plot

    Parameters
    ----------
    comptable_ds: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    div: bokeh.models.Div
        Target Div element where component images will be loaded
    
    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by kappa
    """
    # Create Panel for the Ranked Kappa Plot
    hovertool = models.HoverTool(tooltips=[('Component ID', '@component'),
                                           ('Kappa', '@kappa'),
                                           ('Rho', '@rho'),
                                           ('Var. Expl.', '@varexp')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                          tools=["tap,wheel_zoom,reset,pan,crosshair", hovertool],
                          title="Components sorted by Kappa")
    fig.circle('kappa_rank', 'kappa', source=comptable_ds, size=3, color='color')
    fig.xaxis.axis_label = 'Kappa Rank'
    fig.yaxis.axis_label = 'Kappa'
    fig.x_range = models.Range1d(-1, Nc + 1)
    fig.toolbar.logo = None
    return fig


def create_rho_sortedPlot(comptable_ds):
    """
    Create Dymamic Sorted Rho Plot

    Parameters
    ----------
    comptable_ds: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    
    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by rho
    """
    hovertool = models.HoverTool(tooltips=[('Component ID', '@component'),
                                           ('Kappa', '@kappa'),
                                           ('Rho', '@rho'),
                                           ('Var. Expl.', '@varexp')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                         tools=["tap,wheel_zoom,reset,pan,crosshair", hovertool],
                         title="Components sorted by Rho")
    fig.circle('rho_rank', 'rho', source=comptable_ds, size=3, color='color')
    fig.xaxis.axis_label = 'Rho Rank'
    fig.yaxis.axis_label = 'Rho'
    fig.x_range = models.Range1d(-1, Nc + 1)
    fig.toolbar.logo = None

    return fig


def create_varexp_sortedPlot(comptable_ds):
    """
    Create Dynamic Sorted VarExp Plot

    Parameters
    ----------
    comptable_ds: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    
    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by VarExp
    """
    # Create Panel for the Ranked Kappa Plot
    hovertool = models.HoverTool(tooltips=[('Component ID', '@component'),
                                           ('Kappa', '@kappa'),
                                           ('Rho', '@rho'),
                                           ('Var. Expl.', '@varexp')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                         tools=["tap,wheel_zoom,reset,pan,crosshair", hovertool],
                         title="Components sorted by Variance Explained")
    fig.circle('varexp_rank', 'varexp', source=comptable_ds, size=3, color='color')
    fig.xaxis.axis_label = 'Variance Rank'
    fig.yaxis.axis_label = 'Variance'
    fig.x_range = models.Range1d(-1, Nc + 1)
    fig.toolbar.logo = None
    return fig


def tap_callback(comptable_ds, CDS_meica_ts, CDS_meica_fft,
                 CDS_TSplot, CDS_FFTplot, div_content, out_dir):
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
    return models.CustomJS(
        args=dict(source_comp_table=comptable_ds,
                  source_meica_ts=CDS_meica_ts,
                  source_tsplot=CDS_TSplot,
                  source_meica_fft=CDS_meica_fft,
                  source_fftplot=CDS_FFTplot,
                  div=div_content, outdir=out_dir),
        code="""
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


def generate_spectrum_CDS(CDS_meica_mix, tr, n_comps):
    """
    """
    spectrum, freqs = get_spectrum(CDS_meica_mix.data['C000'], TR)
    Nf = spectrum.shape[0]
    DF = pd.DataFrame(columns=['C'+str(c).zfill(3) for c in np.arange(Nc)],index=np.arange(Nf))
    for c in np.arange(Nc):
        cid = 'C' + str(c).zfill(3)
        ts = CDS_meica_mix.data[cid]
        spectrum, freqs = get_spectrum(ts, 2)
        DF[cid] = spectrum
    DF['Freq'] = freqs
    CDS = models.ColumnDataSource(DF)
    return CDS, Nf


# Page Genearation
# 1) Load the Comp_table file into a bokeh CDS
CDS_CompTable = create_data_struct(out_dir)
# 2) Load the Component Timeseries into a bokeh CDS
[CDS_meica_mix, Nt, Nc] = load_comp_ts(out_dir)
# 3) Generate the Component Spectrum and store it into a bokeh CDS
[CDS_meica_fft, Nf] = generate_spectrum_CDS(CDS_meica_mix,TR,Nc)
# 6) DIV
div_content = models.Div(width=600, height=900, height_policy='fixed')
# -------------------------------------------------------------------------------------
# 7) Create the Kappa/Rho Scatter Plot
kappa_rho_plot = create_krPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, div_content, out_dir)
# 8) Create the Ranked Kappa Plot
kappa_sorted_plot = create_ksortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div_content)
# 9) Create the Ranked Rho Plot
rho_sorted_plot = create_rho_sortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div_content)
# 10) Create the Ranked Variance Explained Plot
varexp_sorted_plot = create_varexp_sortedPlot(CDS_CompTable, CDS_meica_mix, CDS_meica_fft, CDS_TSplot, CDS_FFTplot, Nc, div_content)
# 11) Create the Component Timeseries Plot
ts_plot = _create_ts_plot(Nt)
# 12) Create the Component FFT Plot
fft_plot = _create_fft_plot(Nt,Nf)
# 13) Create a layout
app = layouts.column(layouts.row(kappa_rho_plot, kappa_sorted_plot, rho_sorted_plot, varexp_sorted_plot), layouts.row(ts_plot,fft_plot), div_content)
# 14) Create Script and Div
(kr_script, kr_div) = embed.components(app)
# 15) Embed into Report Template
generate_report(kr_div, kr_script, file_path='/opt/report_v2.html')

