"""Output and result writing utilities for tedana workflows.

This module provides functions for writing workflow outputs.
"""

import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import tedana.gscontrol as gsc
from tedana import __version__, io
from tedana.bibtex import get_description_references

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def apply_tedort(
    mixing: np.ndarray,
    accepted_comps: List[int],
    rejected_comps: List[int],
) -> np.ndarray:
    """Orthogonalize rejected components with respect to accepted components.

    Parameters
    ----------
    mixing : np.ndarray
        Mixing matrix (T x C).
    accepted_comps : list of int
        Indices of accepted components.
    rejected_comps : list of int
        Indices of rejected components.

    Returns
    -------
    np.ndarray
        Modified mixing matrix with orthogonalized rejected components.
    """
    acc_ts = mixing[:, accepted_comps]
    rej_ts = mixing[:, rejected_comps]
    betas = np.linalg.lstsq(acc_ts, rej_ts, rcond=None)[0]
    pred_rej_ts = np.dot(acc_ts, betas)
    resid = rej_ts - pred_rej_ts

    mixing_orth = mixing.copy()
    mixing_orth[:, rejected_comps] = resid

    RepLGR.info(
        "Rejected components' time series were then "
        "orthogonalized with respect to accepted components' time "
        "series."
    )

    return mixing_orth


def write_denoised_results(
    data_optcom: np.ndarray,
    mask: np.ndarray,
    component_table: pd.DataFrame,
    mixing: np.ndarray,
    io_generator: Any,
) -> None:
    """Write denoised timeseries and component results.

    Parameters
    ----------
    data_optcom : np.ndarray
        Optimally combined data (S x T).
    mask : np.ndarray
        Mask for denoising.
    component_table : pd.DataFrame
        Component metrics table.
    mixing : np.ndarray
        Mixing matrix (T x C).
    io_generator : OutputGenerator
        Output generator for saving files.
    """
    io.writeresults(
        data_optcom,
        mask=mask,
        component_table=component_table,
        mixing=mixing,
        io_generator=io_generator,
    )


def apply_mir(
    data_optcom: np.ndarray,
    mixing: np.ndarray,
    mask: np.ndarray,
    component_table: pd.DataFrame,
    classification_tags: List[str],
    io_generator: Any,
) -> None:
    """Apply minimum image regression.

    Parameters
    ----------
    data_optcom : np.ndarray
        Optimally combined data (S x T).
    mixing : np.ndarray
        Mixing matrix (T x C).
    mask : np.ndarray
        Mask array.
    component_table : pd.DataFrame
        Component metrics table.
    classification_tags : list of str
        Classification tags from selector.
    io_generator : OutputGenerator
        Output generator for saving files.
    """
    gsc.minimum_image_regression(
        data_optcom=data_optcom,
        mixing=mixing,
        mask=mask,
        component_table=component_table,
        classification_tags=classification_tags,
        io_generator=io_generator,
    )


def write_echo_results(
    data_cat: np.ndarray,
    mixing: np.ndarray,
    mask: np.ndarray,
    component_table: pd.DataFrame,
    io_generator: Any,
) -> None:
    """Write per-echo results (verbose mode).

    Parameters
    ----------
    data_cat : np.ndarray
        Multi-echo data (S x E x T).
    mixing : np.ndarray
        Mixing matrix (T x C).
    mask : np.ndarray
        Mask array.
    component_table : pd.DataFrame
        Component metrics table.
    io_generator : OutputGenerator
        Output generator for saving files.
    """
    io.writeresults_echoes(data_cat, mixing, mask, component_table, io_generator)


def save_derivative_metadata(
    io_generator: Any,
    info_dict: Dict[str, Any],
    workflow_name: str,
    workflow_description: str,
) -> None:
    """Save BIDS-compatible derivative metadata.

    Parameters
    ----------
    io_generator : OutputGenerator
        Output generator for saving files.
    info_dict : dict
        System and command info dictionary.
    workflow_name : str
        Name of the workflow (e.g., 'tedana', 't2smap', 'ica_reclassify').
    workflow_description : str
        Description of the workflow.
    """
    derivative_metadata = {
        "Name": f"{workflow_name} Outputs",
        "BIDSVersion": "1.5.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": workflow_name,
                "Version": __version__,
                "Description": workflow_description,
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
    with open(io_generator.get_name("data description json"), "w") as fo:
        json.dump(derivative_metadata, fo, sort_keys=True, indent=4)


def finalize_report_text(repname: str, bibtex_file: str) -> None:
    """Finalize report text and write BibTeX references.

    Parameters
    ----------
    repname : str
        Path to report text file.
    bibtex_file : str
        Path to BibTeX output file.
    """
    RepLGR.info(
        "\n\nThis workflow used numpy \\citep{van2011numpy}, scipy \\citep{virtanen2020scipy}, "
        "pandas \\citep{mckinney2010data,reback2020pandas}, "
        "scikit-learn \\citep{pedregosa2011scikit}, "
        "nilearn, bokeh \\citep{bokehmanual}, matplotlib \\citep{Hunter2007}, "
        "and nibabel \\citep{brett_matthew_2019_3233118}."
    )

    RepLGR.info(
        "This workflow also used the Dice similarity index "
        "\\citep{dice1945measures,sorensen1948method}."
    )

    with open(repname) as fo:
        report = [line.rstrip() for line in fo.readlines()]
        report = " ".join(report)
        # Double-spaces reflect new paragraphs
        report = report.replace("  ", "\n\n")

    with open(repname, "w") as fo:
        fo.write(report)

    # Collect BibTeX entries
    references = get_description_references(report)

    with open(bibtex_file, "w") as fo:
        fo.write(references)
