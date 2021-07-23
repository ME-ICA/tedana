import os
from os.path import join as opj
from pathlib import Path
from string import Template

import pandas as pd
from bokeh import __version__ as bokehversion
from bokeh import embed, layouts, models
from tedana.info import __version__
from tedana.reporting import dynamic_figures as df


def _generate_buttons(out_dir):
    resource_path = Path(__file__).resolve().parent.joinpath("data", "html")

    images_list = [img for img in os.listdir(out_dir) if ".svg" in img]
    optcom_nogsr_disp = "none"
    optcom_name = ""
    if "carpet_optcom_nogsr.svg" in images_list:
        optcom_nogsr_disp = "block"
        optcom_name = "before MIR"

    denoised_mir_disp = "none"
    denoised_name = ""
    if "carpet_denoised_mir.svg" in images_list:
        denoised_mir_disp = "block"
        denoised_name = "before MIR"

    accepted_mir_disp = "none"
    accepted_name = ""
    if "carpet_accepted_mir.svg" in images_list:
        accepted_mir_disp = "block"
        accepted_name = "before MIR"

    buttons_template_name = "report_carpet_buttons_template.html"
    buttons_template_path = resource_path.joinpath(buttons_template_name)
    with open(str(buttons_template_path), "r") as buttons_file:
        buttons_tpl = Template(buttons_file.read())

    buttons_html = buttons_tpl.substitute(optcomdisp=optcom_nogsr_disp,
                                          denoiseddisp=denoised_mir_disp,
                                          accepteddisp=accepted_mir_disp,
                                          optcomname=optcom_name,
                                          denoisedname=denoised_name,
                                          acceptedname=accepted_name)

    return buttons_html


def _update_template_bokeh(bokeh_id, about, bokeh_js, buttons):
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
    resource_path = Path(__file__).resolve().parent.joinpath("data", "html")

    body_template_name = "report_body_template.html"
    body_template_path = resource_path.joinpath(body_template_name)
    with open(str(body_template_path), "r") as body_file:
        body_tpl = Template(body_file.read())
    body = body_tpl.substitute(content=bokeh_id, about=about, javascript=bokeh_js, buttons=buttons)
    return body


def _save_as_html(body):
    """
    Save an HTML report out to a file.

    Parameters
    ----------
    body : str
        Body for HTML report with embedded figures
    """
    resource_path = Path(__file__).resolve().parent.joinpath("data", "html")
    head_template_name = "report_head_template.html"
    head_template_path = resource_path.joinpath(head_template_name)
    with open(str(head_template_path), "r") as head_file:
        head_tpl = Template(head_file.read())

    html = head_tpl.substitute(
        version=__version__, bokehversion=bokehversion, body=body
    )
    return html


def generate_report(io_generator, tr):
    """
    Parameters
    ----------
    io_generator : tedana.io.OutputGenerator
        io_generator object for this workflow's output
    tr : float
        The repetition time (TR) for the collected multi-echo
        sequence

    Returns
    -------
    HTML : file
        A generated HTML report
    """
    # Load the component time series
    comp_ts_path = io_generator.get_name("ICA mixing tsv")
    comp_ts_df = pd.read_csv(comp_ts_path, sep="\t", encoding="utf=8")
    n_vols, n_comps = comp_ts_df.shape

    # Load the component table
    comptable_path = io_generator.get_name("ICA metrics tsv")
    comptable_cds = df._create_data_struct(comptable_path)

    # Create kappa rho plot
    kappa_rho_plot = df._create_kr_plt(comptable_cds)

    # Create sorted plots
    kappa_sorted_plot = df._create_sorted_plt(
        comptable_cds, n_comps, "kappa_rank", "kappa", "Kappa Rank", "Kappa"
    )
    rho_sorted_plot = df._create_sorted_plt(
        comptable_cds, n_comps, "rho_rank", "rho", "Rho Rank", "Rho"
    )
    varexp_pie_plot = df._create_varexp_pie_plt(comptable_cds, n_comps)

    # link all dynamic figures
    figs = [kappa_rho_plot, kappa_sorted_plot, rho_sorted_plot, varexp_pie_plot]

    div_content = models.Div(width=500, height=750, height_policy="fixed")

    for fig in figs:
        df._link_figures(fig, comptable_cds, div_content, io_generator)

    # Create a layout
    app = layouts.gridplot(
        [
            [
                layouts.row(
                    layouts.column(
                        layouts.row(kappa_rho_plot, varexp_pie_plot),
                        layouts.row(rho_sorted_plot, kappa_sorted_plot),
                    ),
                    layouts.column(div_content),
                )
            ]
        ],
        toolbar_location="left",
    )

    # Embed for reporting and save out HTML
    kr_script, kr_div = embed.components(app)

    # Generate html of buttons (only for images that were generated)
    buttons_html = _generate_buttons(opj(io_generator.out_dir, "figures"))

    # Read in relevant methods
    with open(opj(io_generator.out_dir, "report.txt"), "r+") as f:
        about = f.read()

    body = _update_template_bokeh(kr_div, about, kr_script, buttons_html)
    html = _save_as_html(body)
    with open(opj(io_generator.out_dir, "tedana_report.html"), "wb") as f:
        f.write(html.encode("utf-8"))
