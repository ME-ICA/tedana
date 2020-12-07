import numpy as np
from math import pi
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from bokeh import (events, models, plotting, transform)

color_mapping = {'accepted': '#2ecc71',
                 'rejected': '#e74c3c',
                 'ignored': '#3498db'}

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

        // Image Below Plots
        div.text = ""
        var line = "<span><img src='./figures/comp_"+selected_padded_forIMG+".png'" +
            " alt='Component Map'><span>\\n";
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

        // Image Below Plots
        div.text = ""
        var line = "<p>Please select an individual component to view it in more detail</p>\\n"
        var text = div.text.concat(line);

    }
    """


def _create_data_struct(comptable_path, color_mapping=color_mapping):
    """
    Create Bokeh ColumnDataSource with all info dynamic plots need

    Parameters
    ----------
    comptable: str
        file path to component table, JSON format

    Returns
    -------
    cds: bokeh.models.ColumnDataSource
        Data structure with all the fields to plot or hover over
    """
    unused_cols = ['normalized variance explained',
                   'countsigFR2', 'countsigFS0',
                   'dice_FS0', 'countnoise', 'dice_FR2',
                   'signal-noise_t', 'signal-noise_p',
                   'd_table_score', 'kappa ratio',
                   'rationale', 'd_table_score_scrub']

    df = pd.read_json(comptable_path)
    df.drop('Description', axis=0, inplace=True)
    df.drop('Method', axis=1, inplace=True)
    df = df.T
    n_comps = df.shape[0]

    # remove space from column name
    df.rename(columns={'variance explained': 'var_exp'}, inplace=True)

    # For providing sizes based on Var Explained that are visible
    mm_scaler = MinMaxScaler(feature_range=(4, 20))
    df['var_exp_size'] = mm_scaler.fit_transform(
        df[['var_exp', 'normalized variance explained']])[:, 0]

    # Calculate Kappa and Rho ranks
    df['rho_rank'] = df['rho'].rank(ascending=False).values
    df['kappa_rank'] = df['kappa'].rank(ascending=False).values
    df['var_exp_rank'] = df['var_exp'].rank(ascending=False).values

    # Remove unused columns to decrease size of final HTML
    # set errors to 'ignore' in case some columns do not exist in
    # a given data frame
    df.drop(unused_cols, axis=1, inplace=True, errors='ignore')

    # Create additional Column with colors based on final classification
    df['color'] = [color_mapping[i] for i in df['classification']]

    # Create additional column with component ID
    df['component'] = np.arange(n_comps)

    # Compute angle and re-sort data for Pie plots
    df['angle'] = df['var_exp'] / df['var_exp'].sum() * 2 * pi
    df.sort_values(by=['classification', 'var_exp'], inplace=True)

    cds = models.ColumnDataSource(data=dict(
        kappa=df['kappa'],
        rho=df['rho'],
        varexp=df['var_exp'],
        kappa_rank=df['kappa_rank'],
        rho_rank=df['rho_rank'],
        varexp_rank=df['var_exp_rank'],
        component=[str(i) for i in df['component']],
        color=df['color'],
        size=df['var_exp_size'],
        classif=df['classification'],
        angle=df['angle']))

    return cds


def _create_kr_plt(comptable_cds):
    """
    Create Dymamic Kappa/Rho Scatter Plot

    Parameters
    ----------
    comptable_cds: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh scatter plot of kappa vs. rho
    """
    # Create Panel for the Kappa - Rho Scatter
    kr_hovertool = models.HoverTool(tooltips=[('Component ID', '@component'),
                                              ('Kappa', '@kappa{0.00}'),
                                              ('Rho', '@rho{0.00}'),
                                              ('Var. Expl.', '@varexp{0.00}%')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                          tools=["tap,wheel_zoom,reset,pan,crosshair,save", kr_hovertool],
                          title="Kappa / Rho Plot")
    diagonal = models.Slope(gradient=1, y_intercept=0, line_color='#D3D3D3')
    fig.add_layout(diagonal)
    fig.circle('kappa', 'rho', size='size', color='color', alpha=0.5, source=comptable_cds,
               legend_group='classif')
    fig.xaxis.axis_label = 'Kappa'
    fig.yaxis.axis_label = 'Rho'
    fig.toolbar.logo = None
    fig.legend.background_fill_alpha = 0.5
    fig.legend.orientation = 'horizontal'
    fig.legend.location = 'bottom_right'
    return fig


def _create_sorted_plt(comptable_cds, n_comps, x_var, y_var, title=None,
                       x_label=None, y_label=None):
    """
    Create dynamic sorted plots

    Parameters
    ----------
    comptable_ds: bokeh.models.ColumnDataSource
        Data structure containing a limited set of columns from the comp_table

    x_var: str
        Name of variable for the x-axis

    y_var: str
        Name of variable for the y-axis

    title: str
        Plot title

    x_label: str
        X-axis label

    y_label: str
        Y-axis label

    Returns
    -------
    fig: bokeh.plotting.figure.Figure
        Bokeh plot of components ranked by a given feature
    """
    hovertool = models.HoverTool(tooltips=[('Component ID', '@component'),
                                           ('Kappa', '@kappa{0.00}'),
                                           ('Rho', '@rho{0.00}'),
                                           ('Var. Expl.', '@varexp{0.00}%')])
    fig = plotting.figure(plot_width=400, plot_height=400,
                          tools=["tap,wheel_zoom,reset,pan,crosshair,save", hovertool],
                          title=title)
    fig.line(x=np.arange(1, n_comps + 1),
             y=comptable_cds.data[y_var].sort_values(ascending=False).values,
             color='black')
    fig.circle(x_var, y_var, source=comptable_cds,
               size=5, color='color', alpha=0.7)
    fig.xaxis.axis_label = x_label
    fig.yaxis.axis_label = y_label
    fig.x_range = models.Range1d(-1, n_comps + 1)
    fig.toolbar.logo = None

    return fig


def _create_varexp_pie_plt(comptable_cds, n_comps):
    fig = plotting.figure(plot_width=400, plot_height=400, title='Variance Explained View',
                          tools=['hover,tap,save'],
                          tooltips=[('Component ID', ' @component'),
                                    ('Kappa', '@kappa{0.00}'),
                                    ('Rho', '@rho{0.00}'),
                                    ('Var. Exp.', '@varexp{0.00}%')])
    fig.wedge(x=0, y=1, radius=.9,
              start_angle=transform.cumsum('angle', include_zero=True),
              end_angle=transform.cumsum('angle'),
              line_color="white",
              fill_color='color', source=comptable_cds, fill_alpha=0.7)
    fig.axis.visible = False
    fig.grid.visible = False
    fig.toolbar.logo = None

    circle = models.Circle(x=0, y=1, size=150, fill_color='white', line_color='white')
    fig.add_glyph(circle)

    return fig


def _tap_callback(comptable_cds, div_content, out_dir):
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
    return models.CustomJS(args=dict(source_comp_table=comptable_cds,
                                     div=div_content,
                                     outdir=out_dir), code=tap_callback_jscode)


def _link_figures(fig, comptable_ds, div_content, out_dir):
    """
    Links figures and adds interaction on mouse-click.

    Parameters
    ----------
    fig : bokeh.plotting.figure
        Figure containing a given plot

    comptable_ds : bokeh.models.ColumnDataSource
        Data structure with a limited version of the comptable
        suitable for dynamic plot purposes

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
                    _tap_callback(comptable_ds,
                                  div_content,
                                  out_dir))
    return fig
