"""Setup and teardown utilities for tedana workflows.

This module provides common setup and teardown functions used by
all tedana workflows.
"""

import datetime
import logging
import os
import os.path as op
from glob import glob
from typing import Tuple

from tedana import io, utils

LGR = logging.getLogger("GENERAL")


def setup_output_directory(out_dir: str) -> str:
    """Create and return absolute path to output directory.

    Parameters
    ----------
    out_dir : str
        Path to output directory.

    Returns
    -------
    str
        Absolute path to output directory.
    """
    out_dir = op.abspath(out_dir)
    if not op.isdir(out_dir):
        os.mkdir(out_dir)
    return out_dir


def rename_previous_reports(out_dir: str, prefix: str) -> Tuple[str, str]:
    """Rename any previous report files and return new report paths.

    Parameters
    ----------
    out_dir : str
        Output directory path.
    prefix : str
        Prefix for filenames.

    Returns
    -------
    tuple of (str, str)
        Tuple containing (repname, bibtex_file) paths.
    """
    prefix = io._infer_prefix(prefix)
    basename = f"{prefix}report"
    extension = "txt"
    repname = op.join(out_dir, f"{basename}.{extension}")
    bibtex_file = op.join(out_dir, f"{prefix}references.bib")

    # Rename previous report files
    repex = op.join(out_dir, f"{basename}*")
    previousreps = glob(repex)
    previousreps.sort(reverse=True)
    for f in previousreps:
        previousparts = op.splitext(f)
        newname = previousparts[0] + "_old" + previousparts[1]
        os.rename(f, newname)

    return repname, bibtex_file


def setup_logging(
    out_dir: str,
    repname: str,
    quiet: bool = False,
    debug: bool = False,
) -> str:
    """Set up logging for the workflow.

    Parameters
    ----------
    out_dir : str
        Output directory path.
    repname : str
        Report filename for the report logger.
    quiet : bool, optional
        Whether to suppress logging output. Default is False.
    debug : bool, optional
        Whether to enable debug logging. Default is False.

    Returns
    -------
    str
        Path to the log file.
    """
    start_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    logname = op.join(out_dir, f"tedana_{start_time}.tsv")
    utils.setup_loggers(logname, repname, quiet=quiet, debug=debug)
    return logname


def save_workflow_command(
    out_dir: str,
    command: str,
    filename: str = "tedana_call.sh",
) -> None:
    """Save the workflow command to a shell script file.

    Parameters
    ----------
    out_dir : str
        Output directory path.
    command : str
        Command string to save.
    filename : str, optional
        Name of the output file. Default is "tedana_call.sh".
    """
    with open(os.path.join(out_dir, filename), "w") as f:
        f.write(command)


def teardown_workflow() -> None:
    """Clean up after workflow completion.

    Logs completion message, newsletter info, and tears down loggers.
    """
    LGR.info("Workflow completed")
    utils.log_newsletter_info()
    utils.teardown_loggers()
