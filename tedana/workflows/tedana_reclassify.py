"""
Run the reclassification workflow for a previous tedana run
"""
import argparse
import datetime
import logging
import os
import os.path as op
import sys
from glob import glob

import numpy as np
import pandas as pd

import tedana.gscontrol as gsc
from tedana import __version__, io, reporting, selection, utils
from tedana.bibtex import get_description_references

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def main():
    from tedana import __version__

    verstr = "tedana_reclassify v{}".format(__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "registry",
        help="File registry from a previous tedana run",
    )
    parser.add_argument(
        "--manacc",
        dest="manual_accept",
        nargs="+",
        type=int,
        help="Component indices to accept (zero-indexed).",
    )
    parser.add_argument(
        "--manrej",
        dest="manual_reject",
        nargs="+",
        type=int,
        help="Component indices to reject (zero-indexed).",
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="File naming configuration. Default auto (prepackaged).",
        default="auto",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default=".",
    )
    parser.add_argument(
        "--prefix", dest="prefix", type=str, help="Prefix for filenames generated.", default=""
    )
    parser.add_argument(
        "--convention",
        dest="convention",
        action="store",
        choices=["orig", "bids"],
        help=("Filenaming convention. bids will use the latest BIDS derivatives version."),
        default="bids",
    )
    parser.add_argument(
        "--tedort",
        dest="tedort",
        action="store_true",
        help=("Orthogonalize rejected components w.r.t. accepted components prior to denoising."),
        default=False,
    )
    parser.add_argument(
        "--mir",
        dest="mir",
        action="store_true",
        help="Run minimum image regression.",
    )
    parser.add_argument(
        "--no-reports",
        dest="no_reports",
        action="store_true",
        help=(
            "Creates a figures folder with static component "
            "maps, timecourse plots and other diagnostic "
            "images and displays these in an interactive "
            "reporting framework"
        ),
        default=False,
    )
    parser.add_argument(
        "--png-cmap", dest="png_cmap", type=str, help="Colormap for figures", default="coolwarm"
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=(
            "Logs in the terminal will have increased "
            "verbosity, and will also be written into "
            "a .tsv file in the output directory."
        ),
        default=False,
    )
    parser.add_argument(
        "--force",
        "-f",
        dest="force",
        action="store_true",
        help="Force overwriting of files. Default False.",
    )
    parser.add_argument(
        "--quiet", dest="quiet", help=argparse.SUPPRESS, action="store_true", default=False
    )
    parser.add_argument("-v", "--version", action="version", version=verstr)

    args = parser.parse_args()

    # Run post-tedana
    post_tedana(
        args.registry,
        accept=args.manual_accept,
        reject=args.manual_reject,
        out_dir=args.out_dir,
        config=args.config,
        convention=args.convention,
        tedort=args.tedort,
        mir=args.mir,
        no_reports=args.no_reports,
        png_cmap=args.png_cmap,
        force=args.force,
        debug=args.debug,
        quiet=args.quiet,
    )


def post_tedana(
    registry,
    accept=[],
    reject=[],
    out_dir=".",
    config="auto",
    convention="bids",
    prefix="",
    tedort=False,
    mir=False,
    no_reports=False,
    png_cmap="coolwarm",
    force=False,
    debug=False,
    quiet=False,
):
    """
    Run the post-tedana manual classification workflow.

    Please remember to cite [1]_.

    Parameters
    ----------
    registry: :obj:`str`
        The previously run registry as a JSON file.
    accept: :obj: `list`
        A list of integer values of components to accept in this workflow.
    reject: :obj: `list`
        A list of integer values of components to reject in this workflow.
    out_dir : :obj:`str`, optional
        Output directory.
    tedort : :obj:`bool`, optional
        Orthogonalize rejected components w.r.t. accepted ones prior to
        denoising. Default is False.
    mir : :obj:`bool`, optional
        Run minimum image regression after denoising. Default is False.
    no_reports : obj:'bool', optional
        Do not generate .html reports and .png plots. Default is false such
        that reports are generated.
    png_cmap : obj:'str', optional
        Name of a matplotlib colormap to be used when generating figures.
        Cannot be used with --no-png. Default is 'coolwarm'.
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    force : :obj:`bool`, optional
        Whether to force file overwrites. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppresses logging/printing of messages. Default is False.

    Notes
    -----
    This workflow writes out several files. For a complete list of the files
    generated by this workflow, please visit
    https://tedana.readthedocs.io/en/latest/outputs.html

    References
    ----------
    .. [1] DuPre, E. M., Salo, T., Ahmed, Z., Bandettini, P. A., Bottenhorn, K. L.,
           Caballero-Gaudes, C., Dowdle, L. T., Gonzalez-Castillo, J., Heunis, S.,
           Kundu, P., Laird, A. R., Markello, R., Markiewicz, C. J., Moia, S.,
           Staden, I., Teves, J. B., Uruñuela, E., Vaziri-Pashkam, M.,
           Whitaker, K., & Handwerker, D. A. (2021).
           TE-dependent analysis of multi-echo fMRI with tedana.
           Journal of Open Source Software, 6(66), 3669. doi:10.21105/joss.03669.
    """
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    # Check that there is no overlap in accepted/rejected components
    if accept:
        acc = set(accept)
    else:
        acc = ()
    if reject:
        rej = set(reject)
    else:
        rej = ()

    if (not accept) and (not reject):
        raise ValueError("Must manually accept or reject at least one component")

    in_both = []
    for a in acc:
        if a in rej:
            in_both.append(a)
    for r in rej:
        if r in acc and r not in rej:
            in_both.append(r)
    if len(in_both) != 0:
        raise ValueError("The following components were both accepted and rejected: " f"{in_both}")

    # boilerplate
    basename = "report"
    extension = "txt"
    repname = op.join(out_dir, (basename + "." + extension))
    bibtex_file = op.join(out_dir, "references.bib")
    repex = op.join(out_dir, (basename + "*"))
    previousreps = glob(repex)
    previousreps.sort(reverse=True)
    for f in previousreps:
        previousparts = op.splitext(f)
        newname = previousparts[0] + "_old" + previousparts[1]
        os.rename(f, newname)

    # create logfile name
    basename = "tedana_"
    extension = "tsv"
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, (basename + start_time + "." + extension))
    utils.setup_loggers(logname=logname, repname=repname, quiet=quiet, debug=debug)

    LGR.info("Using output directory: {}".format(out_dir))

    ioh = io.InputHarvester(registry)
    comptable = ioh.get_file_contents("ICA metrics tsv")
    xcomp = ioh.get_file_contents("ICA cross component metrics json")
    status_table = ioh.get_file_contents("ICA status table tsv")
    previous_tree_fname = ioh.get_file_path("ICA decision tree json")
    mmix = np.asarray(ioh.get_file_contents("ICA mixing tsv"))
    mask_denoise = ioh.get_file_contents("adaptive mask img")
    # If global signal was removed in the previous run, we can assume that
    # the user wants to use that file again. If not, use the default of
    # optimally combined data.
    gskey = "removed gs combined img"
    if ioh.get_file_path(gskey):
        data_oc = ioh.get_file_contents(gskey)
        used_gs = True
    else:
        data_oc = ioh.get_file_contents("combined img")
        used_gs = False
    io_generator = io.OutputGenerator(
        data_oc,
        convention=convention,
        prefix=prefix,
        config=config,
        force=force,
        verbose=False,
        out_dir=out_dir,
    )

    # Make a new selector with the added files
    selector = selection.ComponentSelector.ComponentSelector(
        previous_tree_fname, comptable, cross_component_metrics=xcomp, status_table=status_table
    )

    if accept:
        selector.add_manual(accept, "accepted")
    if reject:
        selector.add_manual(reject, "rejected")
    selector.select()
    comptable = selector.component_table

    # NOTE: most of these will be identical to previous, but this makes
    # things easier for programs which will view the data after running.
    # First, make the output generator
    comp_names = comptable["Component"].values
    mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
    to_copy = [
        "z-scored ICA components img",
        "ICA mixing tsv",
        "ICA decomposition json",
        "ICA metrics json",
    ]
    if used_gs:
        to_copy.append(gskey)
        to_copy.append("has gs combined img")

    for tc in to_copy:
        print(tc)
        io_generator.save_file(ioh.get_file_contents(tc), tc)

    # Save component selector and tree
    selector.to_files(io_generator)

    if selector.n_accepted_comps == 0:
        LGR.warning("No BOLD components detected! Please check data and results!")

    mmix_orig = mmix.copy()
    # TODO: make this a function
    if tedort:
        comps_accepted = selector.accepted_comps
        comps_rejected = selector.rejected_comps
        acc_ts = mmix[:, comps_accepted]
        rej_ts = mmix[:, comps_rejected]
        betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
        pred_rej_ts = np.dot(acc_ts, betas)
        resid = rej_ts - pred_rej_ts
        rej_idx = comps_accepted[comps_accepted].index
        mmix[:, rej_idx] = resid
        comp_names = [
            io.add_decomp_prefix(comp, prefix="ica", max_value=comptable.index.max())
            for comp in range(selector.n_comps)
        ]
        mixing_df = pd.DataFrame(data=mmix, columns=comp_names)
        io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")
        RepLGR.info(
            "Rejected components' time series were then "
            "orthogonalized with respect to accepted components' time "
            "series."
        )

    n_vols = data_oc.shape[3]
    img_t_r = io_generator.reference_img.header.get_zooms()[-1]
    mask_denoise = utils.reshape_niimg(mask_denoise).astype(bool)
    data_oc = utils.reshape_niimg(data_oc)

    # TODO: make a better result-writing function
    # #############################################!!!!
    # TODO: make a better time series creation function
    #       - get_ts_fit_tag(include=[], exclude=[])
    #       - get_ts_regress/residual_tag(include=[], exclude=[])
    #       How to handle [acc/rej] + tag ?
    io.writeresults(
        data_oc,
        mask=mask_denoise,
        comptable=comptable,
        mmix=mmix,
        n_vols=n_vols,
        io_generator=io_generator,
    )

    if mir:
        io_generator.force = True
        gsc.minimum_image_regression(data_oc, mmix, mask_denoise, comptable, io_generator)
        io_generator.force = False

    # Write out BIDS-compatible description file
    derivative_metadata = {
        "Name": "tedana Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "tedana_reclassify",
                "Version": __version__,
                "Description": (
                    "A denoising pipeline for the identification and removal "
                    "of non-BOLD noise from multi-echo fMRI data."
                ),
                "CodeURL": "https://github.com/ME-ICA/tedana",
            }
        ],
    }
    io_generator.save_file(derivative_metadata, "data description json")

    with open(repname, "r") as fo:
        report = [line.rstrip() for line in fo.readlines()]
        report = " ".join(report)
    with open(repname, "w") as fo:
        fo.write(report)

    # Collect BibTeX entries for cited papers
    references = get_description_references(report)

    with open(bibtex_file, "w") as fo:
        fo.write(references)

    if not no_reports:
        LGR.info("Making figures folder with static component maps and timecourse plots.")

        dn_ts, hikts, lowkts = io.denoise_ts(data_oc, mmix, mask_denoise, comptable)

        # Figure out which control methods were used
        gscontrol = []
        if used_gs:
            gscontrol.append("gsr")
        if mir:
            gscontrol.append("mir")
        gscontrol = None if gscontrol is [] else gscontrol

        reporting.static_figures.carpet_plot(
            optcom_ts=data_oc,
            denoised_ts=dn_ts,
            hikts=hikts,
            lowkts=lowkts,
            mask=mask_denoise,
            io_generator=io_generator,
            gscontrol=gscontrol,
        )
        reporting.static_figures.comp_figures(
            data_oc,
            mask=mask_denoise,
            comptable=comptable,
            mmix=mmix_orig,
            io_generator=io_generator,
            png_cmap=png_cmap,
        )

        if sys.version_info.major == 3 and sys.version_info.minor < 6:
            warn_msg = (
                "Reports requested but Python version is less than "
                "3.6.0. Dynamic reports will not be generated."
            )
            LGR.warn(warn_msg)
        else:
            LGR.info("Generating dynamic report")
            reporting.generate_report(io_generator, tr=img_t_r)

    io_generator.save_self()
    LGR.info("Workflow completed")
    utils.teardown_loggers()


if __name__ == "__main__":
    main()
