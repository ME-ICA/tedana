# %%
import pandas as pd
import numpy as np
from os.path import join as opj
from sklearn.preprocessing import MinMaxScaler
from bokeh import (embed, events, layouts, models, plotting)

from tedana import (io, reporting, utils)

color_mapping = {'accepted': '#00ff00',
                 'rejected': '#ff0000',
                 'ignored': '#0000ff'}


def _link_figures(fig, comptable_ds,
                  ts_src, fft_src, ts_plot, fft_plot,
                  div_content, out_dir):
    """
    Links figures and adds interaction on mouse-click.

    Parameters
    ----------
    fig :

    comptable_ds :

    ts_src :

    fft_src :

    ts_plot :

    fft_plot :

    div_content :

    out_dir :

    Returns
    -------
    fig :

    """
    fig.js_on_event(events.Tap,
                    tap_callback(comptable_ds, ts_src, fft_src,
                                 ts_plot, fft_plot,
                                 div_content, out_dir))
    return fig


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


def _create_fft_plot(n_vols, Nf):
    """
    Creates FFT Bokeh plot.

    Parameters
    ----------
    n_vols : int
        Number of volumes in the time series
    Nf :

    Returns
    -------
    fig
    """
    fft_cds = models.ColumnDataSource(data=dict(x=np.arange(n_vols),
                                                y=np.zeros(n_vols,)))
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


def _spectrum_data_src(df, tr):
    """
    Parameters
    ----------
    df :
        The component time series as a Pandas DataFrame
    tr : float
        Repetition time of the acquired ME data set

    Returns
    -------
    data_src :
    Nf :
    """
    n_comps, n_vols = df.shape
    data_src = models.ColumnDataSource(df)
    spectrum, _ = utils.get_spectrum(data_src.data['ica_00'], tr)
    Nf = spectrum.shape[0]
    return data_src, Nf


def create_data_struct(comptable, color_mapping=color_mapping):
    """
    Create Bokeh ColumnDataSource with all relevant info
    needed for dynamic plots.

    Parameters
    ----------
    comptable: str
        file path to component table, JSON format

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

    df = io.load_comptable(comptable)
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
        kappa_rank=df['kappa_rank'],
        rho_rank=df['rho_rank'],
        varexp_rank=df['var_exp_rank'],
        component=[str(i) for i in df.index],
        color=df['color'],
        size=df['var_exp_size'],
        classif=df['classification']))

    return comptable_ds


def create_kr_plt(comptable_ds):
    """
    Create dynamic Kappa vs Rho scatter plot

    Parameters
    ----------
    comptable_ds: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    div: bokeh.models.Div
        Target Div element where component images will be loaded

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh scatter plot of kappa vs. rho
    """
    # Create Panel for the Kappa - Rho Scatter
    hovertool = models.HoverTool(tooltips=[('Component ID', '@component'),
                                           ('Kappa', '@kappa'),
                                           ('Rho', '@rho'),
                                           ('Var. Expl.', '@varexp')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                          tools=["tap,wheel_zoom,reset,pan,crosshair", hovertool],
                          title="Kappa vs Rho Plot")
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


def create_sorted_plt(comptable_ds, n_comps,
                      x_var, y_var,
                      title=None,
                      x_label=None, y_label=None):
    """
    Create dynamic sorted plot

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
                          title=title)
    fig.circle(x_var, y_var, source=comptable_ds, size=3, color='color')
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = y_label
    fig.x_range = models.Range1d(-1, n_comps + 1)
    fig.toolbar.logo = None

    return fig


def tap_callback(comptable_ds, ts_src, fft_src,
                 ts_plot, fft_plot,
                 div_content, out_dir):
    """
    Javacript function to animate tap events and show component info on the right

    Parameters
    ----------
    comptable_ds: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table
    ts_src :
    fft_src :
    ts_plot :
    fft_plot :
    div : bokeh.models.Div
        Target Div element where component images will be loaded

    Returns
    -------
    CustomJS: bokeh.models.CustomJS
        Javascript function that adds the tapping functionality
    """
    return models.CustomJS(
        args=dict(source_comp_table=comptable_ds,
                  source_meica_ts=ts_src,
                  source_tsplot=ts_plot,
                  source_meica_fft=fft_src,
                  source_fftplot=fft_plot,
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


# %%
# ----------------------------------------------------------------
# What not to do:

out_dir = '/home/emdupre/Desktop/tedest'
tr = 2

# Load the component time series
comp_ts = opj(out_dir, 'ica_mixing.tsv')
comptable = opj(out_dir, 'ica_decomposition.json')
df = pd.read_csv(comp_ts, sep='\t', encoding='utf-8')
n_comps, n_vols = df.shape

# Load the component table
comptable_ds = create_data_struct(comptable)
# generate the component spectrum
data_src, Nf = _spectrum_data_src(df, tr)

# create fft, ts plots
fft_plot = _create_fft_plot(n_vols, Nf)
ts_plot = _create_ts_plot(n_vols)

# %%

# create kapp rho plot
kappa_rho_plot = create_kr_plt(comptable_ds)

# create all sorted plots
kappa_sorted_plot = create_sorted_plt()
rho_sorted_plot = create_sorted_plt()
varexp_sorted_plot = create_sorted_plt()

# link all dynamic figures
figs = [kappa_rho_plot, kappa_sorted_plot,
        rho_sorted_plot, varexp_sorted_plot]
for f in figs:
    _link_figures(f)

# Create a layout
div_content = models.Div(width=600, height=900, height_policy='fixed')
app = layouts.column(layouts.row(kappa_rho_plot, kappa_sorted_plot,
                                 rho_sorted_plot, varexp_sorted_plot),
                     layouts.row(ts_plot, fft_plot),
                     div_content)

# Embed for reporting
kr_script, kr_div = embed.components(app)
reporting.generate_report(kr_div, kr_script,
                          file_path='/opt/report_v2.html')
