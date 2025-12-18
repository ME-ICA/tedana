"""Report generation utilities for tedana workflows.

This module provides functions for generating HTML reports and static figures.
"""

import logging
import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from tedana import io, reporting

LGR = logging.getLogger("GENERAL")


def generate_static_figures(
    data_optcom: np.ndarray,
    mask_denoise: np.ndarray,
    base_mask: np.ndarray,
    component_table: pd.DataFrame,
    mixing: np.ndarray,
    io_generator: Any,
    png_cmap: str,
    gscontrol: List[str],
    masksum_denoise: Optional[np.ndarray] = None,
    external_regressors: Optional[pd.DataFrame] = None,
    t2smap_provided: bool = False,
) -> None:
    """Generate static figures for the HTML report.

    Parameters
    ----------
    data_optcom : np.ndarray
        Optimally combined data (S x T).
    mask_denoise : np.ndarray
        Denoising mask.
    base_mask : np.ndarray
        Base mask for adaptive mask plot.
    component_table : pd.DataFrame
        Component metrics table.
    mixing : np.ndarray
        Mixing matrix (T x C) - original, not tedort-modified.
    io_generator : OutputGenerator
        Output generator for saving files.
    png_cmap : str
        Colormap for figures.
    gscontrol : list of str
        Global signal control methods used.
    masksum_denoise : np.ndarray, optional
        Masksum for RMSE plot.
    external_regressors : pd.DataFrame, optional
        External regressors for correlation heatmap.
    t2smap_provided : bool, optional
        Whether T2* map was provided (skip RMSE plot if True).
    """
    LGR.info("Making figures folder with static component maps and timecourse plots.")

    # Generate denoised timeseries for carpet plot
    data_denoised, data_accepted, data_rejected = io.denoise_ts(
        data_optcom,
        mixing,
        mask_denoise,
        component_table,
    )

    # Adaptive mask plot
    reporting.static_figures.plot_adaptive_mask(
        optcom=data_optcom,
        base_mask=base_mask,
        io_generator=io_generator,
    )

    # Carpet plot
    reporting.static_figures.carpet_plot(
        optcom_ts=data_optcom,
        denoised_ts=data_denoised,
        hikts=data_accepted,
        lowkts=data_rejected,
        mask=mask_denoise,
        io_generator=io_generator,
        gscontrol=gscontrol,
    )

    # Component figures
    reporting.static_figures.comp_figures(
        data_optcom,
        mask=mask_denoise,
        component_table=component_table,
        mixing=mixing,
        io_generator=io_generator,
        png_cmap=png_cmap,
    )

    # T2* and S0 plots
    reporting.static_figures.plot_t2star_and_s0(io_generator=io_generator, mask=mask_denoise)

    # RMSE plot (only if T2* map was computed, not provided)
    if not t2smap_provided and masksum_denoise is not None:
        reporting.static_figures.plot_rmse(
            io_generator=io_generator,
            adaptive_mask=masksum_denoise,
        )

    # Global signal control plots
    if gscontrol:
        reporting.static_figures.plot_gscontrol(
            io_generator=io_generator,
            gscontrol=gscontrol,
        )

    # External regressors correlation heatmap
    if external_regressors is not None:
        comp_names = component_table["Component"].values
        mixing_df = pd.DataFrame(data=mixing, columns=comp_names)
        reporting.static_figures.plot_heatmap(
            mixing=mixing_df,
            external_regressors=external_regressors,
            component_table=component_table,
            out_file=os.path.join(
                io_generator.out_dir,
                "figures",
                f"{io_generator.prefix}confound_correlations.svg",
            ),
        )


def generate_dynamic_report(
    io_generator: Any,
    cluster_labels: Optional[np.ndarray] = None,
    similarity_t_sne: Optional[np.ndarray] = None,
) -> None:
    """Generate interactive HTML report.

    Parameters
    ----------
    io_generator : OutputGenerator
        Output generator for saving files.
    cluster_labels : np.ndarray, optional
        Cluster labels from robustica.
    similarity_t_sne : np.ndarray, optional
        t-SNE similarity from robustica.
    """
    LGR.info("Generating dynamic report")
    reporting.generate_report(io_generator, cluster_labels, similarity_t_sne)


def generate_reclassify_figures(
    data_optcom: np.ndarray,
    mask_denoise: np.ndarray,
    component_table: pd.DataFrame,
    mixing: np.ndarray,
    io_generator: Any,
    png_cmap: str,
    gscontrol: List[str],
) -> None:
    """Generate figures for ica_reclassify workflow.

    This is a simpler version that doesn't include all the plots
    from the full tedana workflow.

    Parameters
    ----------
    data_optcom : np.ndarray
        Optimally combined data (S x T).
    mask_denoise : np.ndarray
        Denoising mask.
    component_table : pd.DataFrame
        Component metrics table.
    mixing : np.ndarray
        Mixing matrix (T x C).
    io_generator : OutputGenerator
        Output generator for saving files.
    png_cmap : str
        Colormap for figures.
    gscontrol : list of str
        Global signal control methods used.
    """
    LGR.info("Making figures folder with static component maps and timecourse plots.")

    # Generate denoised timeseries for carpet plot
    dn_ts, hikts, lowkts = io.denoise_ts(data_optcom, mixing, mask_denoise, component_table)

    # Carpet plot
    reporting.static_figures.carpet_plot(
        optcom_ts=data_optcom,
        denoised_ts=dn_ts,
        hikts=hikts,
        lowkts=lowkts,
        mask=mask_denoise,
        io_generator=io_generator,
        gscontrol=gscontrol,
    )

    # Component figures
    reporting.static_figures.comp_figures(
        data_optcom,
        mask=mask_denoise,
        component_table=component_table,
        mixing=mixing,
        io_generator=io_generator,
        png_cmap=png_cmap,
    )
