import os
from os.path import join as opj
from pathlib import Path
from string import Template
import shutil
import subprocess

import pandas as pd
from bokeh import __version__ as bokehversion
from bokeh import embed, layouts, models
from tedana.info import __version__
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import pi

color_mapping = {"accepted": "#2ecc71", "rejected": "#e74c3c", "ignored": "#3498db"}


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
    unused_cols = [
        "normalized variance explained",
        "countsigFT2",
        "countsigFS0",
        "dice_FS0",
        "countnoise",
        "dice_FT2",
        "signal-noise_t",
        "signal-noise_p",
        "d_table_score",
        "kappa ratio",
        "rationale",
        "d_table_score_scrub",
    ]

    df = pd.read_table(comptable_path)
    n_comps = df.shape[0]

    # remove space from column name
    df.rename(columns={"variance explained": "var_exp"}, inplace=True)

    # For providing sizes based on Var Explained that are visible
    mm_scaler = MinMaxScaler(feature_range=(4, 20))
    df["var_exp_size"] = mm_scaler.fit_transform(
        df[["var_exp", "normalized variance explained"]]
    )[:, 0]

    # Calculate Kappa and Rho ranks
    df["rho_rank"] = df["rho"].rank(ascending=False).values
    df["kappa_rank"] = df["kappa"].rank(ascending=False).values
    df["var_exp_rank"] = df["var_exp"].rank(ascending=False).values

    # Remove unused columns to decrease size of final HTML
    # set errors to 'ignore' in case some columns do not exist in
    # a given data frame
    df.drop(unused_cols, axis=1, inplace=True, errors="ignore")

    # Create additional Column with colors based on final classification
    df["color"] = [color_mapping[i] for i in df["classification"]]

    # Create additional column with component ID
    df["component"] = np.arange(n_comps)

    # Compute angle and re-sort data for Pie plots
    df["angle"] = df["var_exp"] / df["var_exp"].sum() * 2 * pi
    df.sort_values(by=["classification", "var_exp"], inplace=True)

    return df


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

    buttons_html = buttons_tpl.substitute(
        optcomdisp=optcom_nogsr_disp,
        denoiseddisp=denoised_mir_disp,
        accepteddisp=accepted_mir_disp,
        optcomname=optcom_name,
        denoisedname=denoised_name,
        acceptedname=accepted_name,
    )

    return buttons_html


def _save_as_html(body, mode):
    """
    Save an HTML report out to a file.

    Parameters
    ----------
    body : str
        Body for HTML report with embedded figures
    """
    resource_path = Path(__file__).resolve().parent.joinpath("data", "src")

    if mode == "data":
        path = str(resource_path.joinpath("Plots.js"))
        with open(path, "r") as head_file:
            head_tpl = Template(head_file.read())
        html = head_tpl.substitute(componentsData=body)

    elif mode == "about":
        path = str(resource_path.joinpath("index.js"))
        with open(path, "r") as head_file:
            head_tpl = Template(head_file.read())
        html = head_tpl.substitute(about=body)

    return html


def react_structure(df):
    """
    {
        x: "IC 1",
        var: 30,
        kappa: 40,
        rho: 10,
        status: "accepted",
        color: acceptedColor,
    },
    """
    data = ""
    for index, row in df.iterrows():
        row_data = f'x: "Component {row["component"]}", var: {row["var_exp"]}, kappa: {row["kappa"]}, kappa_rank: {row["kappa_rank"]}, rho: {row["rho"]}, rho_rank: {row["rho_rank"]}, classification: "{row["classification"]}", color: "{row["color"]}",'
        data += f"{{ {row_data} }},"
    return data


def copy_report_files(out_dir):
    resource_path = Path(__file__).resolve().parent.joinpath("data")
    shutil.copytree(resource_path, opj(out_dir, "report"))


def build_report(out_dir):
    subprocess.run("npm install", cwd=out_dir, shell=True)
    subprocess.run(
        "npm run build",
        cwd=out_dir,
        shell=True,
    )
    subprocess.run("npx gulp", cwd=out_dir, shell=True)


def clean_report_directory(out_dir):
    file_names = os.listdir(out_dir)
    for file_name in file_names:
        file_name_path = opj(out_dir, file_name)
        if file_name != "build":
            if os.path.isdir(file_name_path):
                shutil.rmtree(file_name_path)
            else:
                os.remove(file_name_path)

    build_dir = opj(out_dir, "build")
    file_names = os.listdir(build_dir)

    for file_name in file_names:
        shutil.move(
            os.path.join(build_dir, file_name), os.path.join(out_dir, file_name)
        )

    shutil.rmtree(build_dir)
    os.rename(opj(out_dir, "index.html"), opj(out_dir, "tedana_report.html"))


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

    # Load the component table
    comptable_path = io_generator.get_name("ICA metrics tsv")
    comptable_cds = _create_data_struct(comptable_path)

    # Read in relevant methods
    with open(opj(io_generator.out_dir, "report.txt"), "r+") as f:
        about = f.read()

    about = about.replace("\n", "<br/>")
    about = about.replace("References:", "<h3>References:</h3>")

    react_data = react_structure(comptable_cds)

    html = _save_as_html(react_data, "data")
    out_dir = opj(io_generator.out_dir, "report", "src")
    with open(opj(out_dir, "Plots.js"), "wb") as f:
        f.write(html.encode("utf-8"))

    html = _save_as_html(about, "about")
    with open(opj(out_dir, "index.js"), "wb") as f:
        f.write(html.encode("utf-8"))
