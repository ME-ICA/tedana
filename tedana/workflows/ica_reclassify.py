"""Run the reclassification workflow for a previous tedana run."""

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
from tedana.io import ALLOWED_COMPONENT_DELIMITERS
from tedana.workflows.parser_utils import parse_manual_list_int, parse_manual_list_str

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def _get_parser():
    """Parse command line inputs for ica_reclassify.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    from tedana import __version__

    verstr = f"ica_reclassify v{__version__}"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Argument parser follow template provided by RalphyZ
    # https://stackoverflow.com/a/43456577
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "registry",
        help="File registry from a previous tedana run",
    )
    optional.add_argument(
        "--manacc",
        dest="manual_accept",
        nargs="+",
        help=(
            "Component indices to accept (zero-indexed)."
            "Supply as a comma-delimited list with no spaces, "
            "as a csv file, or as a text file with an allowed "
            f"delimiter {repr(ALLOWED_COMPONENT_DELIMITERS)}."
        ),
        default=[],
    )
    optional.add_argument(
        "--manrej",
        dest="manual_reject",
        nargs="+",
        help=(
            "Component indices to reject (zero-indexed)."
            "Supply as a comma-delimited list with no spaces, "
            "as a csv file, or as a text file with an allowed "
            f"delimiter {repr(ALLOWED_COMPONENT_DELIMITERS)}."
        ),
        default=[],
    )
    optional.add_argument(
        "--tagacc",
        dest="tag_accept",
        nargs="+",
        help=(
            "Classification tag(s) to add to accepted components."
            "Will be applied to all listed accepted components, "
            "even if they were already accepted."
            "Supply a single tag or a comma-delimited list."
        ),
        default=[],
    )
    optional.add_argument(
        "--tagrej",
        dest="tag_reject",
        nargs="+",
        help=(
            "Classification tag(s) to add to rejected components."
            "Will be applied to all listed rejected components, "
            "even if they were already rejected."
            "Supply a single tag or a comma-delimited list."
        ),
        default=[],
    )
    optional.add_argument(
        "--config",
        dest="config",
        help="File naming configuration.",
        default="auto",
    )
    optional.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="PATH",
        help="Output directory.",
        default=".",
    )
    optional.add_argument(
        "--prefix", dest="prefix", type=str, help="Prefix for filenames generated.", default=""
    )
    optional.add_argument(
        "--convention",
        dest="convention",
        action="store",
        choices=["orig", "bids"],
        help=("Filenaming convention. bids will use the latest BIDS derivatives version."),
        default="bids",
    )
    optional.add_argument(
        "--tedort",
        dest="tedort",
        action="store_true",
        help=("Orthogonalize rejected components w.r.t. accepted components prior to denoising."),
        default=False,
    )
    optional.add_argument(
        "--mir",
        dest="mir",
        action="store_true",
        help="Run minimum image regression.",
        default=False,
    )
    optional.add_argument(
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
    optional.add_argument(
        "--png-cmap", dest="png_cmap", type=str, help="Colormap for figures", default="coolwarm"
    )
    optional.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Generate intermediate and additional files.",
        default=False,
    )
    optional.add_argument(
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
    optional.add_argument(
        "--overwrite",
        "-f",
        dest="overwrite",
        action="store_true",
        help="Force overwriting of files.",
    )
    optional.add_argument(
        "--quiet", dest="quiet", help=argparse.SUPPRESS, action="store_true", default=False
    )
    optional.add_argument("-v", "--version", action="version", version=verstr)

    parser._action_groups.append(optional)
    return parser


def _main(argv=None):
    """Run the ica_reclassify workflow."""
    if argv:
        # relevant for tests or if CLI called using ica_reclassify_cli._main(args)
        reclassify_command = "ica_reclassify " + " ".join(argv)
    else:
        reclassify_command = "ica_reclassify " + " ".join(sys.argv[1:])

    args = _get_parser().parse_args(argv)

    # Run ica_reclassify_workflow
    ica_reclassify_workflow(
        args.registry,
        accept=args.manual_accept,
        reject=args.manual_reject,
        tag_accept=args.tag_accept,
        tag_reject=args.tag_reject,
        out_dir=args.out_dir,
        config=args.config,
        prefix=args.prefix,
        convention=args.convention,
        tedort=args.tedort,
        mir=args.mir,
        no_reports=args.no_reports,
        png_cmap=args.png_cmap,
        overwrite=args.overwrite,
        verbose=args.verbose,
        debug=args.debug,
        quiet=args.quiet,
        reclassify_command=reclassify_command,
    )


def ica_reclassify_workflow(
    registry,
    accept=[],
    reject=[],
    tag_accept=[],
    tag_reject=[],
    out_dir=".",
    config="auto",
    convention="bids",
    prefix="",
    tedort=False,
    mir=False,
    no_reports=False,
    png_cmap="coolwarm",
    verbose=False,
    overwrite=False,
    debug=False,
    quiet=False,
    reclassify_command=None,
):
    """
    Run the post-tedana manual classification workflow.

    Please remember to cite [1]_.

    Parameters
    ----------
    registry : :obj:`str`
        The previously run registry as a JSON file.
    accept : :obj: `list`
        A list of integer values of components to accept in this workflow.
    reject : :obj: `list`
        A list of integer values of components to reject in this workflow.
    tag_accept : :obj: `list`
        A list of classification tags to add to accepted components.
        Will be applied to all listed accepted components, even if they were already accepted.
    tag_reject : :obj: `list`
        A list of classification tags to add to rejected components.
        Will be applied to all listed rejected components, even if they were already rejected.
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
    verbose : :obj:`bool`, optional
        Generate intermediate and additional files. Default is False.
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    overwrite : :obj:`bool`, optional
        Whether to force file overwrites. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppresses logging/printing of messages. Default is False.
    reclassify_command : :obj:`str`, optional
        The command used to run ica_reclassify. Default is None.

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

    # boilerplate
    prefix = io._infer_prefix(prefix)
    basename = f"{prefix}report"
    extension = "txt"
    repname = op.join(out_dir, (basename + "." + extension))
    bibtex_file = op.join(out_dir, f"{prefix}references.bib")
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

    # If accept and reject are a list of integers, they stay the same
    # If they are a filename, load numbers of from
    # If they are a string of values, convert to a list of ints
    accept = parse_manual_list_int(accept)
    reject = parse_manual_list_int(reject)

    # If classification tags are a list of strings without commas, they stay the same.
    # If classification tags are a single string,
    # convert to a list with a single string or a split into a comma delimited list of strings.
    tag_accept = parse_manual_list_str(tag_accept)
    tag_reject = parse_manual_list_str(tag_reject)

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

    if len(in_both) != 0:
        raise ValueError(f"The following components were both accepted and rejected: {in_both}")

    # Save command into sh file, if the command-line interface was used
    # TODO: use io_generator to save command
    if reclassify_command is not None:
        command_file = open(os.path.join(out_dir, "ica_reclassify_call.sh"), "w")
        command_file.write(reclassify_command)
        command_file.close()
    else:
        # Get variables passed to function if the tedana command is None
        variables = ", ".join(f"{name}={value}" for name, value in locals().items())
        # From variables, remove everything after ", tedana_command"
        variables = variables.split(", reclassify_command")[0]
        reclassify_command = f"ica_reclassify_workflow({variables})"

    # Save system info to json
    info_dict = utils.get_system_version_info()
    info_dict["Command"] = reclassify_command

    LGR.info(f"Using output directory: {out_dir}")

    ioh = io.InputHarvester(registry)
    component_table = ioh.get_file_contents("ICA metrics tsv")
    xcomp = ioh.get_file_contents("ICA cross component metrics json")
    status_table = ioh.get_file_contents("ICA status table tsv")
    previous_tree_fname = ioh.get_file_path("ICA decision tree json")
    mixing = np.asarray(ioh.get_file_contents("ICA mixing tsv"))
    adaptive_mask = ioh.get_file_contents("adaptive mask img")
    # If global signal was removed in the previous run, we can assume that
    # the user wants to use that file again. If not, use the default of
    # optimally combined data.
    gskey = "removed gs combined img"
    if ioh.get_file_path(gskey):
        data_optcom = ioh.get_file_contents(gskey)
        used_gs = True
    else:
        data_optcom = ioh.get_file_contents("combined img")
        used_gs = False

    if verbose:
        LGR.debug("Loading input 4D data")
        data_cat = ioh.get_file_contents("input img")
        # Extract the data from the nibabel objects
        data_cat, _ = io.load_data(data_cat, n_echos=len(data_cat))

    io_generator = io.OutputGenerator(
        data_optcom,
        convention=convention,
        prefix=prefix,
        config=config,
        overwrite=overwrite,
        verbose=verbose,
        out_dir=out_dir,
        old_registry=ioh.registry,
    )

    # Make a new selector with the added files
    selector = selection.component_selector.ComponentSelector(previous_tree_fname)

    if accept:
        selector.add_manual(accept, "accepted", classification_tags=tag_accept)

    if reject:
        selector.add_manual(reject, "rejected", classification_tags=tag_reject)

    selector.select(
        component_table,
        cross_component_metrics=xcomp,
        status_table=status_table,
    )
    component_table = selector.component_table_

    # NOTE: most of these will be identical to previous, but this makes
    # things easier for programs which will view the data after running.
    # First, make the output generator
    comp_names = component_table["Component"].values
    mixing_df = pd.DataFrame(data=mixing, columns=comp_names)
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

    # calculate the fit of rejected to accepted components to use as a quality measure
    # Note: This adds a column to component_table & needs to run before the table is saved
    reporting.quality_metrics.calculate_rejected_components_impact(selector, mixing)

    # Save component selector and tree
    selector.to_files(io_generator)

    if selector.n_accepted_comps_ == 0:
        LGR.warning(
            "No accepted components remaining after manual classification! "
            "Please check data and results!"
        )

    mixing_orig = mixing.copy()
    # TODO: make this a function
    if tedort:
        comps_accepted = selector.accepted_comps_
        comps_rejected = selector.rejected_comps_
        acc_ts = mixing[:, comps_accepted]
        rej_ts = mixing[:, comps_rejected]
        betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
        pred_rej_ts = np.dot(acc_ts, betas)
        resid = rej_ts - pred_rej_ts
        mixing[:, comps_rejected] = resid
        comp_names = [
            io.add_decomp_prefix(comp, prefix="ica", max_value=component_table.index.max())
            for comp in range(selector.n_comps_)
        ]
        mixing_df = pd.DataFrame(data=mixing, columns=comp_names)
        io_generator.save_file(mixing_df, "ICA orthogonalized mixing tsv")
        RepLGR.info(
            "Rejected components' time series were then "
            "orthogonalized with respect to accepted components' time "
            "series."
        )

    # img_t_r = io_generator.reference_img.header.get_zooms()[-1]
    adaptive_mask = utils.reshape_niimg(adaptive_mask)
    mask_denoise = adaptive_mask >= 1
    data_optcom = utils.reshape_niimg(data_optcom)

    # TODO: make a better result-writing function
    # #############################################!!!!
    # TODO: make a better time series creation function
    #       - get_ts_fit_tag(include=[], exclude=[])
    #       - get_ts_regress/residual_tag(include=[], exclude=[])
    #       How to handle [acc/rej] + tag ?
    io.writeresults(
        data_optcom,
        mask=mask_denoise,
        component_table=component_table,
        mixing=mixing,
        io_generator=io_generator,
    )

    if mir:
        io_generator.overwrite = True
        gsc.minimum_image_regression(
            data_optcom=data_optcom,
            mixing=mixing,
            mask=mask_denoise,
            component_table=component_table,
            classification_tags=selector.classification_tags,
            io_generator=io_generator,
        )
        io_generator.overwrite = False

    if verbose:
        LGR.debug("Writing out verbose data")
        io.writeresults_echoes(data_cat, mixing, mask_denoise, component_table, io_generator)

    # Write out BIDS-compatible description file
    derivative_metadata = {
        "Name": "tedana Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "ica_reclassify",
                "Version": __version__,
                "Description": (
                    "A denoising pipeline for the identification and removal "
                    "of non-BOLD noise from multi-echo fMRI data."
                ),
                "CodeURL": "https://github.com/ME-ICA/tedana",
                "Node": {
                    "Name": info_dict["Node"],
                    "System": info_dict["System"],
                    "Machine": info_dict["Machine"],
                    "Processor": info_dict["Processor"],
                    "Release": info_dict["Release"],
                    "Version": info_dict["Version"],
                },
                "Python": info_dict["Python"],
                "Python_Libraries": info_dict["Python_Libraries"],
                "Command": info_dict["Command"],
            }
        ],
    }
    io_generator.save_file(derivative_metadata, "data description json")

    with open(repname) as fo:
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

        dn_ts, hikts, lowkts = io.denoise_ts(data_optcom, mixing, mask_denoise, component_table)

        # Figure out which control methods were used
        gscontrol = []
        if used_gs:
            gscontrol.append("gsr")
        if mir:
            gscontrol.append("mir")
        gscontrol = None if gscontrol == [] else gscontrol

        reporting.static_figures.carpet_plot(
            optcom_ts=data_optcom,
            denoised_ts=dn_ts,
            hikts=hikts,
            lowkts=lowkts,
            mask=mask_denoise,
            io_generator=io_generator,
            gscontrol=gscontrol,
        )
        reporting.static_figures.comp_figures(
            data_optcom,
            mask=mask_denoise,
            component_table=component_table,
            mixing=mixing_orig,
            io_generator=io_generator,
            png_cmap=png_cmap,
        )

        LGR.info("Generating dynamic report")
        reporting.generate_report(io_generator, cluster_labels=None, similarity_t_sne=None)

    io_generator.save_self()
    LGR.info("Workflow completed")

    # Add newsletter info to the log
    utils.log_newsletter_info()

    utils.teardown_loggers()


if __name__ == "__main__":
    _main()
