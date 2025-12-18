"""T2*/S0 decay fitting utilities for tedana workflows.

This module provides functions for fitting the T2* decay model.
"""

import logging
from typing import Any

from scipy import stats

from tedana import decay, utils
from tedana.workflows.shared.containers import DecayMaps, MaskData, MultiEchoData

LGR = logging.getLogger("GENERAL")


def fit_decay_model(
    data: MultiEchoData,
    masks: MaskData,
    fittype: str,
    io_generator: Any,
    verbose: bool = False,
    n_threads: int = 1,
) -> DecayMaps:
    """Fit T2*/S0 decay model and save outputs.

    Parameters
    ----------
    data : MultiEchoData
        Loaded multi-echo data.
    masks : MaskData
        Mask data for fitting.
    fittype : str
        Fitting method ('loglin' or 'curvefit').
    io_generator : OutputGenerator
        Output generator for saving files.
    verbose : bool, optional
        Whether to save verbose outputs. Default is False.
    n_threads : int, optional
        Number of threads to use for parallel processing. Default is 1.

    Returns
    -------
    DecayMaps
        Container with T2* and S0 maps.
    """
    LGR.info("Computing T2* map")
    t2s_limited, s0_limited, t2s_full, s0_full = decay.fit_decay(
        data=data.data_cat,
        tes=data.tes,
        mask=masks.mask_denoise,
        adaptive_mask=masks.masksum_denoise,
        fittype=fittype,
        n_threads=n_threads,
    )

    # Set a hard cap for the T2* map
    # Anything 10x higher than the 99.5 percentile is reset
    cap_t2s = stats.scoreatpercentile(t2s_full.flatten(), 99.5, interpolation_method="lower")
    LGR.debug(f"Setting cap on T2* map at {utils.millisec2sec(cap_t2s):.5f}s")
    t2s_full[t2s_full > cap_t2s * 10] = cap_t2s

    # Save outputs
    io_generator.save_file(utils.millisec2sec(t2s_full), "t2star img")
    io_generator.save_file(s0_full, "s0 img")

    if verbose:
        io_generator.save_file(utils.millisec2sec(t2s_limited), "limited t2star img")
        io_generator.save_file(s0_limited, "limited s0 img")

    # Calculate RMSE
    rmse_map, rmse_df = decay.rmse_of_fit_decay_ts(
        data=data.data_cat,
        tes=data.tes,
        adaptive_mask=masks.masksum_denoise,
        t2s=t2s_limited,
        s0=s0_limited,
        fitmode="all",
    )
    io_generator.save_file(rmse_map, "rmse img")
    io_generator.add_df_to_file(rmse_df, "confounds tsv")

    return DecayMaps(
        t2s_limited=t2s_limited,
        t2s_full=t2s_full,
        s0_limited=s0_limited,
        s0_full=s0_full,
    )


def fit_decay_model_simple(
    data_cat,
    tes,
    mask,
    masksum,
    fittype: str,
    fitmode: str = "all",
    n_threads: int = 1,
):
    """Fit T2*/S0 decay model without saving (for t2smap workflow).

    Parameters
    ----------
    data_cat : np.ndarray
        Multi-echo data array (S x E x T).
    tes : list of float
        Echo times in milliseconds.
    mask : np.ndarray
        Mask array.
    masksum : np.ndarray
        Masksum array.
    fittype : str
        Fitting method ('loglin' or 'curvefit').
    fitmode : str, optional
        Fitting mode ('all' or 'ts'). Default is 'all'.
    n_threads : int, optional
        Number of threads to use for parallel processing. Default is 1.

    Returns
    -------
    tuple
        Tuple of (t2s_limited, s0_limited, t2s_full, s0_full).
    """
    LGR.info("Computing adaptive T2* map")
    decay_function = decay.fit_decay if fitmode == "all" else decay.fit_decay_ts
    return decay_function(
        data=data_cat,
        tes=tes,
        mask=mask,
        adaptive_mask=masksum,
        fittype=fittype,
        n_threads=n_threads,
    )
