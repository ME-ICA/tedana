"""Estimate T2 and S0, and optimally combine data across TEs."""

import argparse
import logging
import os
import os.path as op
import sys

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from tedana import __version__, combine, decay, io, utils
from tedana.workflows.parser_utils import is_valid_file

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def _get_parser():
    """Parse command line inputs for t2smap.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Argument parser follow template provided by RalphyZ
    # https://stackoverflow.com/a/43456577
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-d",
        dest="data",
        nargs="+",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Multi-echo dataset for analysis. May be a "
            "single file with spatially concatenated data "
            "or a set of echo-specific files, in the same "
            "order as the TEs are listed in the -e "
            "argument."
        ),
        required=True,
    )
    required.add_argument(
        "-e",
        dest="tes",
        nargs="+",
        metavar="TE",
        type=float,
        help="Echo times (in ms). E.g., 15.0 39.0 63.0",
        required=True,
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
        "--mask",
        dest="mask",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help=(
            "Binary mask of voxels to include in TE "
            "Dependent ANAlysis. Must be in the same "
            "space as `data`."
        ),
        default=None,
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
        "--fittype",
        dest="fittype",
        action="store",
        choices=["loglin", "curvefit"],
        help=(
            "Desired T2*/S0 fitting method. "
            '"loglin" means that a linear model is fit '
            "to the log of the data. "
            '"curvefit" means that a more computationally '
            "demanding monoexponential model is fit "
            "to the raw data. "
        ),
        default="loglin",
    )
    optional.add_argument(
        "--fitmode",
        dest="fitmode",
        action="store",
        choices=["all", "ts"],
        help=(
            "Monoexponential model fitting scheme. "
            '"all" means that the model is fit, per voxel, '
            "across all timepoints. "
            '"ts" means that the model is fit, per voxel '
            "and per timepoint."
        ),
        default="all",
    )
    optional.add_argument(
        "--combmode",
        dest="combmode",
        action="store",
        choices=["t2s", "paid"],
        help=("Combination scheme for TEs: t2s (Posse 1999), paid (Poser)"),
        default="t2s",
    )
    optional.add_argument(
        "--n-threads",
        dest="n_threads",
        type=int,
        action="store",
        help=(
            "Number of threads to use. Used by "
            "threadpoolctl to set the parameter outside "
            "of the workflow function. Higher numbers of "
            "threads tend to slow down performance on "
            "typical datasets."
        ),
        default=1,
    )
    optional.add_argument(
        "--debug", dest="debug", help=argparse.SUPPRESS, action="store_true", default=False
    )
    optional.add_argument(
        "--quiet", dest="quiet", help=argparse.SUPPRESS, action="store_true", default=False
    )
    parser._action_groups.append(optional)
    return parser


def t2smap_workflow(
    data,
    tes,
    out_dir=".",
    mask=None,
    prefix="",
    convention="bids",
    fittype="loglin",
    fitmode="all",
    combmode="t2s",
    debug=False,
    quiet=False,
    t2smap_command=None,
):
    """
    Estimate T2 and S0, and optimally combine data across TEs.

    Please remember to cite :footcite:t:`dupre2021te`.

    Parameters
    ----------
    data : :obj:`str` or :obj:`list` of :obj:`str`
        Either a single z-concatenated file (single-entry list or str) or a
        list of echo-specific files, in ascending order.
    tes : :obj:`list`
        List of echo times associated with data in milliseconds.
    out_dir : :obj:`str`, optional
        Output directory.
    mask : :obj:`str`, optional
        Binary mask of voxels to include in TE Dependent ANAlysis. Must be spatially
        aligned with `data`.
    fittype : {'loglin', 'curvefit'}, optional
        Monoexponential fitting method.
        'loglin' means to use the the default linear fit to the log of
        the data.
        'curvefit' means to use a monoexponential fit to the raw data,
        which is slightly slower but may be more accurate.
    fitmode : {'all', 'ts'}, optional
        Monoexponential model fitting scheme.
        'all' means that the model is fit, per voxel, across all timepoints.
        'ts' means that the model is fit, per voxel and per timepoint.
        Default is 'all'.
    combmode : {'t2s', 'paid'}, optional
        Combination scheme for TEs: 't2s' (Posse 1999, default), 'paid' (Poser).
    t2smap_command : :obj:`str`, optional
        The command used to run t2smap. Default is None.

    Other Parameters
    ----------------
    debug : :obj:`bool`, optional
        Whether to run in debugging mode or not. Default is False.
    quiet : :obj:`bool`, optional
        If True, suppress logging/printing of messages. Default is False.

    Notes
    -----
    This workflow writes out several files, which are described below:

    ============================= =================================================
    Filename                      Content
    ============================= =================================================
    T2starmap.nii.gz              Estimated T2* 3D map or 4D timeseries.
                                  Will be a 3D map if ``fitmode`` is 'all' and a
                                  4D timeseries if it is 'ts'.
    S0map.nii.gz                  S0 3D map or 4D timeseries.
    desc-limited_T2starmap.nii.gz Limited T2* map/timeseries. The difference between
                                  the limited and full maps is that, for voxels
                                  affected by dropout where only one echo contains
                                  good data, the full map uses the T2* estimate
                                  from the first two echos, while the limited map
                                  will have a NaN.
    desc-limited_S0map.nii.gz     Limited S0 map/timeseries. The difference between
                                  the limited and full maps is that, for voxels
                                  affected by dropout where only one echo contains
                                  good data, the full map uses the S0 estimate
                                  from the first two echos, while the limited map
                                  will have a NaN.
    desc-optcom_bold.nii.gz       Optimally combined timeseries.
    ============================= =================================================

    References
    ----------
    .. footbibliography::
    """
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    utils.setup_loggers(quiet=quiet, debug=debug)

    LGR.info(f"Using output directory: {out_dir}")

    # Save command into sh file, if the command-line interface was used
    if t2smap_command is not None:
        command_file = open(os.path.join(out_dir, "t2smap_call.sh"), "w")
        command_file.write(t2smap_command)
        command_file.close()
    else:
        # Get variables passed to function if the tedana command is None
        variables = ", ".join(f"{name}={value}" for name, value in locals().items())
        # From variables, remove everything after ", tedana_command"
        variables = variables.split(", t2smap_command")[0]
        t2smap_command = f"t2smap_workflow({variables})"

    # Save system info to json
    info_dict = utils.get_system_version_info()
    info_dict["Command"] = t2smap_command

    # ensure tes are in appropriate format
    tes = [float(te) for te in tes]
    n_echos = len(tes)

    # coerce data to samples x echos x time array
    if isinstance(data, str):
        data = [data]

    LGR.info(f"Loading input data: {[f for f in data]}")
    catd, ref_img = io.load_data(data, n_echos=n_echos)
    io_generator = io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        make_figures=False,
    )
    n_samp, n_echos, n_vols = catd.shape
    LGR.debug(f"Resulting data shape: {catd.shape}")

    if mask is None:
        LGR.info("Computing adaptive mask")
    else:
        LGR.info("Using user-defined mask")
    mask, masksum = utils.make_adaptive_mask(catd, mask=mask, getsum=True, threshold=1)

    LGR.info("Computing adaptive T2* map")
    if fitmode == "all":
        (t2s_limited, s0_limited, t2s_full, s0_full) = decay.fit_decay(
            catd, tes, mask, masksum, fittype
        )
    else:
        (t2s_limited, s0_limited, t2s_full, s0_full) = decay.fit_decay_ts(
            catd, tes, mask, masksum, fittype
        )

    # set a hard cap for the T2* map/timeseries
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s_full.flatten(), 99.5, interpolation_method="lower")
    cap_t2s_sec = utils.millisec2sec(cap_t2s * 10.0)
    LGR.debug(f"Setting cap on T2* map at {cap_t2s_sec:.5f}s")
    t2s_full[t2s_full > cap_t2s * 10] = cap_t2s

    LGR.info("Computing optimal combination")
    # optimally combine data
    data_oc = combine.make_optcom(catd, tes, masksum, t2s=t2s_full, combmode=combmode)

    # clean up numerical errors
    for arr in (data_oc, s0_full, t2s_full):
        np.nan_to_num(arr, copy=False)

    s0_full[s0_full < 0] = 0
    t2s_full[t2s_full < 0] = 0

    io_generator.save_file(
        utils.millisec2sec(t2s_full),
        "t2star img",
    )
    io_generator.save_file(s0_full, "s0 img")
    io_generator.save_file(
        utils.millisec2sec(t2s_limited),
        "limited t2star img",
    )
    io_generator.save_file(
        s0_limited,
        "limited s0 img",
    )
    io_generator.save_file(data_oc, "combined img")

    # Write out BIDS-compatible description file
    derivative_metadata = {
        "Name": "t2smap Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "t2smap",
                "Version": __version__,
                "Description": (
                    "A pipeline estimating T2* from multi-echo fMRI data and "
                    "combining data across echoes."
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
    io_generator.save_self()

    LGR.info("Workflow completed")
    utils.teardown_loggers()


def _main(argv=None):
    """Run the t2smap workflow."""
    if argv:
        # relevant for tests when CLI called with t2smap_cli._main(args)
        t2smap_command = "t2smap " + " ".join(argv)
    else:
        t2smap_command = "t2smap " + " ".join(sys.argv[1:])
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    n_threads = kwargs.pop("n_threads")
    n_threads = None if n_threads == -1 else n_threads
    with threadpool_limits(limits=n_threads, user_api=None):
        t2smap_workflow(**kwargs, t2smap_command=t2smap_command)


if __name__ == "__main__":
    _main()
