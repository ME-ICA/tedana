"""Optimal combination utilities for tedana workflows.

This module provides functions for optimally combining multi-echo data.
"""

import logging
from typing import Any, List, Optional

import numpy as np

import tedana.gscontrol as gsc
from tedana import combine
from tedana.workflows.shared.containers import (
    DecayMaps,
    MaskData,
    MultiEchoData,
    OptcomData,
)

LGR = logging.getLogger("GENERAL")


def compute_optimal_combination(
    data: MultiEchoData,
    masks: MaskData,
    decay_maps: DecayMaps,
    combmode: str,
    io_generator: Any,
    gscontrol: Optional[List[str]] = None,
) -> OptcomData:
    """Compute optimally combined data.

    Parameters
    ----------
    data : MultiEchoData
        Loaded multi-echo data.
    masks : MaskData
        Mask data.
    decay_maps : DecayMaps
        T2*/S0 maps from decay fitting.
    combmode : str
        Combination method (currently only 't2s').
    io_generator : OutputGenerator
        Output generator for saving files.
    gscontrol : list of str, optional
        Global signal control methods. Default is None.

    Returns
    -------
    OptcomData
        Container with optimally combined data.

    Notes
    -----
    If 'gsr' is in gscontrol, global signal regression is applied
    to both the multi-echo data and the optimally combined data.
    """
    if gscontrol is None:
        gscontrol = []

    data_optcom = combine.make_optcom(
        data.data_cat,
        data.tes,
        masks.masksum_denoise,
        t2s=decay_maps.t2s_full,
        combmode=combmode,
    )

    # Apply global signal regression if requested
    if "gsr" in gscontrol:
        # Note: This modifies data.data_cat in place
        data.data_cat, data_optcom = gsc.gscontrol_raw(
            data_cat=data.data_cat,
            data_optcom=data_optcom,
            n_echos=data.n_echos,
            io_generator=io_generator,
        )

    fout = io_generator.save_file(data_optcom, "combined img")
    LGR.info(f"Writing optimally combined data set: {fout}")

    return OptcomData(data_optcom=data_optcom)


def compute_optimal_combination_simple(
    data_cat: np.ndarray,
    tes: List[float],
    masksum: np.ndarray,
    t2s: np.ndarray,
    combmode: str,
) -> np.ndarray:
    """Compute optimally combined data without saving (for t2smap workflow).

    Parameters
    ----------
    data_cat : np.ndarray
        Multi-echo data array (S x E x T).
    tes : list of float
        Echo times in milliseconds.
    masksum : np.ndarray
        Masksum array.
    t2s : np.ndarray
        T2* map.
    combmode : str
        Combination method.

    Returns
    -------
    np.ndarray
        Optimally combined data.
    """
    LGR.info("Computing optimal combination")
    return combine.make_optcom(data_cat, tes, masksum, t2s=t2s, combmode=combmode)
