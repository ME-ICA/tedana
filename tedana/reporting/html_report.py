import logging
import os
from os.path import join as opj
from pathlib import Path
from string import Template

import pandas as pd
from bokeh import __version__ as bokehversion
from bokeh import embed, layouts, models

from tedana import __version__
from tedana.io import load_json
from tedana.reporting import dynamic_figures as df

LGR = logging.getLogger("GENERAL")


def _generate_buttons(out_dir, io_generator):
    resource_path = Path(__file__).resolve().parent.joinpath("data", "html")

    images_list = [img for img in os.listdir(out_dir) if ".svg" in img]
    optcom_nogsr_disp = "none"
    optcom_name = ""
    if f"{io_generator.prefix}carpet_optcom_nogsr.svg" in images_list:
        optcom_nogsr_disp = "block"
        optcom_name = "before MIR"

    denoised_mir_disp = "none"
    denoised_name = ""
    if f"{io_generator.prefix}carpet_denoised_mir.svg" in images_list:
        denoised_mir_disp = "block"
        denoised_name = "before MIR"

    accepted_mir_disp = "none"
    accepted_name = ""
    if f"{io_generator.prefix}carpet_accepted_mir.svg" in images_list:
        accepted_mir_disp = "block"
        accepted_name = "before MIR"

    buttons_template_name = "report_carpet_buttons_template.html"
    buttons_template_path = resource_path.joinpath(buttons_template_name)
    with open(str(buttons_template_path), "r") as buttons_file:
        buttons_tpl = Template(buttons_file.read())

    buttons_html = buttons_tpl.substitute(
        optcomdisp=optcom_nogsr_disp,
        denoiseddisp=denoised_mir_disp,
        accepteddisp=accepted_mir_disp,
        optcomname=optcom_name,
        denoisedname=denoised_name,
        acceptedname=accepted_name,
    )

    return buttons_html


def _update_template_bokeh(bokeh_id, info_table, about, prefix, references, bokeh_js, buttons):
    """
    Populate a report with content.

    Parameters
    ----------
    bokeh_id : str
        HTML div created by bokeh.embed.components
    info_table : str
        HTML div created by _generate_info_table()
    about : str
        Reporting information for a given run
    prefix : str
        Prefix for the outputted figures
    references : str
        BibTeX references associated with the reporting information
    bokeh_js : str
        Javascript created by bokeh.embed.components
    buttons : str
        HTML div created by _generate_buttons()

    Returns
    -------
    HTMLReport : an instance of a populated HTML report
    """
    resource_path = Path(__file__).resolve().parent.joinpath("data", "html")

    # Initial carpet plot (default one)
    initial_carpet = f"./figures/{prefix}carpet_optcom.svg"

    body_template_name = "report_body_template.html"
    body_template_path = resource_path.joinpath(body_template_name)
    with open(str(body_template_path), "r") as body_file:
        body_tpl = Template(body_file.read())
    body = body_tpl.substitute(
        content=bokeh_id,
        info=info_table,
        about=about,
        prefix=prefix,
        initialCarpet=initial_carpet,
        references=references,
        javascript=bokeh_js,
        buttons=buttons,
    )
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

    html = head_tpl.substitute(version=__version__, bokehversion=bokehversion, body=body)
    return html


def _generate_info_table(info_dict):
    """Generate a table with relevant information about the
    system and tedana.
    """
    resource_path = Path(__file__).resolve().parent.joinpath("data", "html")

    info_template_name = "report_info_table_template.html"
    info_template_path = resource_path.joinpath(info_template_name)
    with open(str(info_template_path), "r") as info_file:
        info_tpl = Template(info_file.read())

    info_dict = info_dict["GeneratedBy"][0]
    command = info_dict["Command"]
    version_python = info_dict["Python"]
    info_dict = info_dict["Node"]

    info_html = info_tpl.substitute(
        command=command,
        system=info_dict["System"],
        node=info_dict["Name"],
        release=info_dict["Release"],
        sysversion=info_dict["Version"],
        machine=info_dict["Machine"],
        processor=info_dict["Processor"],
        python=version_python,
        tedana=__version__,
    )
    return info_html


def generate_report(io_generator, tr):
    """Generate an HTML report.

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

    # Load the cross component metrics, including the kappa & rho elbows
    cross_component_metrics_path = io_generator.get_name("ICA cross component metrics json")
    cross_comp_metrics_dict = load_json(cross_component_metrics_path)

    def get_elbow_val(elbow_prefix):
        """
        Find cross component metrics that begin with elbow_prefix and output the value
        Current prefixes are kappa_elbow_kundu and rho_elbow_kundu.

        This flexability
        means anything that begins [kappa/rho]_elbow will be found and used regardless
        of the suffix. If more than one metric has the prefix then the alphabetically
        first one will be used and a warning will be logged.
        """

        elbow_keys = [k for k in cross_comp_metrics_dict.keys() if elbow_prefix in k]
        elbow_keys.sort()
        if len(elbow_keys) == 0:
            LGR.warning(
                f"No {elbow_prefix} saved in cross_component_metrics so not displaying in report"
            )
            return None
        elif len(elbow_keys) == 1:
            return cross_comp_metrics_dict[elbow_keys[0]]
        else:
            printed_key = elbow_keys[0]
            unprinted_keys = elbow_keys[1:]

            LGR.warning(
                "More than one key saved in cross_component_metrics begins with "
                f"{elbow_prefix}. The lines on the plots will be for {printed_key} "
                f"NOT {unprinted_keys}"
            )
            return cross_comp_metrics_dict[printed_key]

    kappa_elbow = get_elbow_val("kappa_elbow")
    rho_elbow = get_elbow_val("rho_elbow")

    # Create kappa rho plot
    kappa_rho_plot = df._create_kr_plt(comptable_cds, kappa_elbow=kappa_elbow, rho_elbow=rho_elbow)

    # Create sorted plots
    kappa_sorted_plot = df._create_sorted_plt(
        comptable_cds,
        n_comps,
        "kappa_rank",
        "kappa",
        title="Kappa Rank",
        x_label="Components sorted by Kappa",
        y_label="Kappa",
        elbow=kappa_elbow,
    )
    rho_sorted_plot = df._create_sorted_plt(
        comptable_cds,
        n_comps,
        "rho_rank",
        "rho",
        title="Rho Rank",
        x_label="Components sorted by Rho",
        y_label="Rho",
        elbow=rho_elbow,
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
    buttons_html = _generate_buttons(opj(io_generator.out_dir, "figures"), io_generator)

    # Read in relevant methods
    with open(opj(io_generator.out_dir, f"{io_generator.prefix}report.txt"), "r+") as f:
        about = f.read()

    with open(opj(io_generator.out_dir, f"{io_generator.prefix}references.bib"), "r") as f:
        references = f.read()

    # Read info table
    data_descr_path = io_generator.get_name("data description json")
    data_descr_dict = load_json(data_descr_path)

    # Create info table
    info_table = _generate_info_table(data_descr_dict)

    body = _update_template_bokeh(
        bokeh_id=kr_div,
        info_table=info_table,
        about=about,
        references=references,
        prefix=io_generator.prefix,
        bokeh_js=kr_script,
        buttons=buttons_html,
    )
    html = _save_as_html(body)
    with open(opj(io_generator.out_dir, f"{io_generator.prefix}tedana_report.html"), "wb") as f:
        f.write(html.encode("utf-8"))
