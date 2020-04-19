import pandas as pd
from bokeh import (embed, layouts, models)
from pathlib import Path
from os.path import join as opj
from string import Template
from tedana.info import __version__
from tedana.reporting import dynamic_figures as df


def _update_template_bokeh(bokeh_id, about, bokeh_js):
    """
    Populate a report with content.

    Parameters
    ----------
    bokeh_id : str
        HTML div created by bokeh.embed.components
    about : str
        Reporting information for a given run
    bokeh_js : str
        Javascript created by bokeh.embed.components
    Returns
    -------
    HTMLReport : an instance of a populated HTML report
    """
    resource_path = Path(__file__).resolve().parent.joinpath('data', 'html')

    body_template_name = 'report_body_template.html'
    body_template_path = resource_path.joinpath(body_template_name)
    with open(str(body_template_path), 'r') as body_file:
        body_tpl = Template(body_file.read())
    body = body_tpl.substitute(content=bokeh_id,
                               about=about,
                               javascript=bokeh_js)
    return body


def _save_as_html(body):
    """
    Save an HTML report out to a file.

    Parameters
    ----------
    body : str
        Body for HTML report with embedded figures
    """
    resource_path = Path(__file__).resolve().parent.joinpath('data', 'html')
    head_template_name = 'report_head_template.html'
    head_template_path = resource_path.joinpath(head_template_name)
    with open(str(head_template_path), 'r') as head_file:
        head_tpl = Template(head_file.read())

    html = head_tpl.substitute(version=__version__, body=body)
    return html


def generate_report(out_dir, tr):
    """
    Parameters
    ----------
    out_dir : str
        File path to a completed tedana output directory
    tr : float
        The repetition time (TR) for the collected multi-echo
        sequence

    Returns
    -------
    HTML : file
        A generated HTML report
    """
    # Load the component time series
    comp_ts_path = opj(out_dir, 'ica_mixing.tsv')
    comp_ts_df = pd.read_csv(comp_ts_path, sep='\t', encoding='utf=8')
    n_vols, n_comps = comp_ts_df.shape

    # Load the component table
    comptable_path = opj(out_dir, 'ica_decomposition.json')
    comptable_cds = df._create_data_struct(comptable_path)

    # Create kappa rho plot
    kappa_rho_plot = df._create_kr_plt(comptable_cds)

    # Create sorted plots
    kappa_sorted_plot = df._create_sorted_plt(comptable_cds, n_comps,
                                              'kappa_rank', 'kappa',
                                              'Kappa Rank', 'Kappa')
    rho_sorted_plot = df._create_sorted_plt(comptable_cds, n_comps,
                                            'rho_rank', 'rho',
                                            'Rho Rank', 'Rho')
    varexp_pie_plot = df._create_varexp_pie_plt(comptable_cds, n_comps)

    # link all dynamic figures
    figs = [kappa_rho_plot, kappa_sorted_plot,
            rho_sorted_plot, varexp_pie_plot]

    div_content = models.Div(width=500, height=750, height_policy='fixed')

    for fig in figs:
        df._link_figures(fig, comptable_cds, div_content, out_dir=out_dir)

    # Create a layout
    app = layouts.gridplot([[
        layouts.column(layouts.row(kappa_rho_plot, varexp_pie_plot),
                       layouts.row(rho_sorted_plot, kappa_sorted_plot)),
        layouts.column(div_content)]],
        toolbar_location='left')

    # Embed for reporting and save out HTML
    kr_script, kr_div = embed.components(app)

    # Read in relevant methods
    with open(opj(out_dir, 'report.txt'), 'r+') as f:
        about = f.read()

    body = _update_template_bokeh(kr_div, about, kr_script)
    html = _save_as_html(body)
    with open(opj(out_dir, 'tedana_report.html'), 'wb') as f:
        f.write(html.encode('utf-8'))
