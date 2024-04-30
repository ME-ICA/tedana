"""Estimate T2 and S0, and optimally combine data across TEs."""

import argparse
import logging
import os
import os.path as op
import sys

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
from tqdm import trange

from tedana import __version__, combine, decay, io, utils
from tedana.metrics.collect import generate_metrics
from tedana.workflows.parser_utils import is_valid_file

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def _get_parser():
    """Parse command line inputs for denoise_echoes.

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
    required.add_argument(
        "--confounds",
        dest="confounds",
        nargs="+",
        metavar="FILE",
        type=lambda x: is_valid_file(parser, x),
        help="Files defining confounds to regress from the echo-wise data.",
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
        "--masktype",
        dest="masktype",
        required=False,
        action="store",
        nargs="+",
        help="Method(s) by which to define the adaptive mask.",
        choices=["dropout", "decay", "none"],
        default=["dropout"],
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


def denoise_echoes_workflow(
    data,
    tes,
    out_dir=".",
    confounds=None,
    mask=None,
    prefix="",
    convention="bids",
    masktype=["dropout"],
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
    confounds : :obj:`list` of :obj:`str`
        Files defining confounds to regress from the echo-wise data.
    mask : :obj:`str`, optional
        Binary mask of voxels to include in TE Dependent ANAlysis. Must be spatially
        aligned with `data`.
    masktype : :obj:`list` with 'dropout' and/or 'decay' or None, optional
        Method(s) by which to define the adaptive mask. Default is ["dropout"].
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
    desc-optcom_bold.nii.gz       Optimally combined timeseries with the confounds
                                  regressed out.
    echo-<n>_bold.nii.gz          Echo-specific timeseries with the confounds
                                  regressed out.
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
    data_cat, ref_img = io.load_data(data, n_echos=n_echos)
    io_generator = io.OutputGenerator(
        ref_img,
        convention=convention,
        out_dir=out_dir,
        prefix=prefix,
        config="auto",
        make_figures=False,
    )
    n_samp, n_echos, n_vols = data_cat.shape
    LGR.debug(f"Resulting data shape: {data_cat.shape}")

    if mask is None:
        LGR.info("Computing adaptive mask")
    else:
        LGR.info("Using user-defined mask")

    mask, adaptive_mask = utils.make_adaptive_mask(
        data_cat,
        mask=mask,
        threshold=1,
        methods=masktype,
    )

    LGR.info("Computing adaptive T2* map")
    if fitmode == "all":
        (t2s_limited, s0_limited, t2s_full, s0_full) = decay.fit_decay(
            data_cat, tes, mask, adaptive_mask, fittype
        )
    else:
        (t2s_limited, s0_limited, t2s_full, s0_full) = decay.fit_decay_ts(
            data_cat, tes, mask, adaptive_mask, fittype
        )

    # set a hard cap for the T2* map/timeseries
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s_full.flatten(), 99.5, interpolation_method="lower")
    cap_t2s_sec = utils.millisec2sec(cap_t2s * 10.0)
    LGR.debug(f"Setting cap on T2* map at {cap_t2s_sec:.5f}s")
    t2s_full[t2s_full > cap_t2s * 10] = cap_t2s

    LGR.info("Computing optimal combination")
    # optimally combine data
    data_optcom = combine.make_optcom(
        data_cat,
        tes,
        adaptive_mask,
        t2s=t2s_full,
        combmode=combmode,
    )

    # clean up numerical errors
    for arr in (data_optcom, s0_full, t2s_full):
        np.nan_to_num(arr, copy=False)

    s0_full[s0_full < 0] = 0
    t2s_full[t2s_full < 0] = 0

    io_generator.save_file(utils.millisec2sec(t2s_full), "t2star img")
    io_generator.save_file(s0_full, "s0 img")
    io_generator.save_file(utils.millisec2sec(t2s_limited), "limited t2star img")
    io_generator.save_file(s0_limited, "limited s0 img")
    io_generator.save_file(data_optcom, "combined img")

    if confounds is not None:
        data_cat_denoised, data_optcom_denoised, metrics = denoise_echoes(
            data_cat=data_cat,
            data_optcom=data_optcom,
            mask=mask,
            confounds=confounds,
        )
        io_generator.save_file(data_cat_denoised, "echo img")
        io_generator.save_file(data_optcom_denoised, "optcom img")
        name = os.path.join(
            io_generator.out_dir,
            io_generator.prefix + "desc-external_metrics.tsv",
        )
        io_generator.save_tsv(metrics, name)

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
        denoise_echoes_workflow(**kwargs, t2smap_command=t2smap_command)


if __name__ == "__main__":
    _main()


def denoise_echoes(
    *,
    data_cat: np.ndarray,
    tes: np.ndarray,
    data_optcom: np.ndarray,
    adaptive_mask: np.ndarray,
    confounds: dict[str, str],
    io_generator: io.OutputGenerator,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Denoise echoes using external regressors.

    TODO: Calculate confound-wise echo-dependence metrics.

    Parameters
    ----------
    data_cat : :obj:`numpy.ndarray` of shape (n_samples, n_echos, n_volumes)
        Concatenated data across echoes.
    tes : :obj:`numpy.ndarray` of shape (n_echos,)
        Echo times in milliseconds.
    data_optcom : :obj:`numpy.ndarray` of shape (n_samples, n_volumes)
        Optimally combined data across echoes.
    adaptive_mask : :obj:`numpy.ndarray` of shape (n_samples,)
        Adaptive mask of voxels to include in TE Dependent ANAlysis.
    confounds : :obj:`dict`
        Files defining confounds to regress from the echo-wise data.
        Keys indicate the names to use for the confounds.
    io_generator : :obj:`~tedana.io.OutputGenerator`
        Output generator.

    Returns
    -------
    data_cat_denoised : :obj:`numpy.ndarray` of shape (n_samples, n_echos, n_volumes)
        Denoised concatenated data across echoes.
    data_optcom_denoised : :obj:`numpy.ndarray` of shape (n_samples, n_volumes)
        Denoised optimally combined data.
    metrics : :obj:`pandas.DataFrame` of shape (n_volumes, n_confounds)
        Metrics of confound-wise echo-dependence.
    """
    LGR.info("Applying a priori confound regression")
    RepLGR.info("Confounds were regressed out of the multi-echo and optimally combined datasets.")
    n_samples, n_echos, n_volumes = data_cat.shape
    mask = adaptive_mask >= 1
    if data_cat.shape[0] != data_optcom.shape[0]:
        raise ValueError(
            f"First dimensions of data_cat ({data_cat.shape[0]}) and data_optcom ({data_optcom.shape[0]}) "
            "do not match"
        )
    elif data_cat.shape[2] != data_optcom.shape[1]:
        raise ValueError(
            f"Third dimension of data_cat ({data_cat.shape[2]}) does not match second dimension "
            f"of data_optcom ({data_optcom.shape[1]})"
        )

    confound_dfs = []
    voxelwise_confounds = {}
    for confound_name, confound_file in confounds.items():
        if confound_file.endswith(".tsv"):
            # One or more time series
            confound_df = pd.read_table(confound_file)
            confound_df = confound_df.add_suffix(f"_{confound_name}")
            assert confound_df.shape[0] == n_volumes
            confound_dfs.append(confound_df)
        elif confound_file.endswith(".nii.gz"):
            confound_arr = utils.reshape_niimg(confound_file)
            if confound_arr.ndim == 1:
                # Spatial map that must be regressed to produce a single time series
                assert confound_arr.shape[0] == n_samples

                # Mask out out-of-brain voxels
                confound_arr = confound_arr[mask]

                # Find the time course for the spatial map
                confound_timeseries = np.linalg.lstsq(
                    np.atleast_2d(confound_arr).T,
                    data_optcom,
                    rcond=None,
                )[0]
                confound_timeseries = stats.zscore(confound_timeseries, axis=None)
                confound_df = pd.DataFrame(confound_timeseries.T, columns=[confound_name])
                assert confound_df.shape[0] == n_volumes
                confound_dfs.append(confound_df)

            elif confound_arr.ndim == 2:
                # Voxel-wise regressors
                assert confound_arr.shape[0] == n_samples
                assert confound_arr.shape[1] == n_volumes
                # Mask out out-of-brain voxels
                confound_arr = confound_arr[mask, ...]
                voxelwise_confounds[confound_name] = confound_arr

            else:
                raise ValueError(f"Unknown shape for confound array: {confound_arr.shape}")
        else:
            raise ValueError(f"Unknown file type: {confound_file}")

    if confound_dfs:
        confounds_df = pd.concat(confound_dfs, axis=1)
        confounds_df = confounds_df.fillna(0)

        # Calculate dependence metrics for non-voxel-wise confounds
        # TODO: Support metrics for voxel-wise confounds too
        required_metrics = [
            "kappa",
            "rho",
            "countnoise",
            "countsigFT2",
            "countsigFS0",
            "dice_FT2",
            "dice_FS0",
            "signal-noise_t",
            "variance explained",
            "normalized variance explained",
            "d_table_score",
        ]
        metrics = generate_metrics(
            data_cat=data_cat,
            data_optcom=data_optcom,
            mixing=confounds_df.values,
            adaptive_mask=adaptive_mask,
            tes=tes,
            io_generator=io_generator,
            label="external",
            metrics=required_metrics,
        )
    else:
        confounds_df = pd.DataFrame(index=range(n_volumes))

    # Project confounds out of optimally combined data
    temporal_mean = data_optcom.mean(axis=-1)  # temporal mean
    data_optcom_denoised = data_optcom[mask] - temporal_mean[mask, np.newaxis]
    if voxelwise_confounds:
        for i_voxel in trange(data_optcom_denoised.shape[0], desc="Denoise optimally combined"):
            design_matrix = confounds_df.copy()
            for confound_name, confound_arr in voxelwise_confounds.items():
                design_matrix[confound_name] = confound_arr[i_voxel, :]

            betas = np.linalg.lstsq(design_matrix.values, data_optcom_denoised.T, rcond=None)[0]
            data_optcom_denoised[i_voxel, :] -= np.dot(
                np.atleast_2d(betas).T,
                design_matrix.values,
            )
    else:
        betas = np.linalg.lstsq(confounds_df.values, data_optcom_denoised.T, rcond=None)[0]
        data_optcom_denoised -= np.dot(np.atleast_2d(betas).T, confounds_df.values)

    # Add the temporal mean back
    data_optcom_denoised += temporal_mean[mask, np.newaxis]

    # io_generator.save_file(data_optcom, "has gs combined img")
    data_optcom_denoised = utils.unmask(data_optcom_denoised, mask)
    # io_generator.save_file(data_optcom_denoised, "removed regressors combined img")

    # Project confounds out of each echo
    data_cat_denoised = data_cat.copy()  # don't overwrite data_cat
    for echo in range(n_echos):
        echo_denoised = data_cat_denoised[:, echo, :][mask]
        # Remove the temporal mean
        temporal_mean = echo_denoised.mean(axis=-1)
        echo_denoised -= temporal_mean
        if voxelwise_confounds:
            for i_voxel in trange(echo_denoised.shape[0], desc=f"Denoise echo {echo + 1}"):
                design_matrix = confounds_df.copy()
                for confound_name, confound_arr in voxelwise_confounds.items():
                    design_matrix[confound_name] = confound_arr[i_voxel, :]

                betas = np.linalg.lstsq(design_matrix.values, echo_denoised.T, rcond=None)[0]
                echo_denoised[i_voxel, :] -= np.dot(np.atleast_2d(betas).T, design_matrix.values)
        else:
            betas = np.linalg.lstsq(confounds_df.values, echo_denoised.T, rcond=None)[0]
            echo_denoised -= np.dot(np.atleast_2d(betas).T, confounds_df.values)

        # Add the temporal mean back
        echo_denoised += temporal_mean

        data_cat_denoised[:, echo, :] = utils.unmask(echo_denoised, mask)

    return data_cat_denoised, data_optcom_denoised, metrics
