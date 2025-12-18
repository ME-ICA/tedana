"""Masking utilities for tedana workflows.

This module provides functions for creating adaptive masks.
"""

import logging
from typing import Any, List, Optional

import numpy as np
from nilearn.masking import compute_epi_mask

from tedana import io, utils
from tedana.workflows.shared.containers import MaskData, MultiEchoData

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def create_adaptive_masks(
    data: MultiEchoData,
    mask_file: Optional[str],
    masktype: List[str],
    io_generator: Any,
    t2smap_file: Optional[str] = None,
    n_independent_echos: Optional[int] = None,
) -> MaskData:
    """Create adaptive masks for denoising and classification.

    Creates both a liberal mask (for denoising, requiring at least 1 good echo)
    and a conservative mask (for classification, requiring at least 3 good echoes).

    Parameters
    ----------
    data : MultiEchoData
        Loaded multi-echo data.
    mask_file : str or None
        Path to user-provided mask file, or None to compute from data.
    masktype : list of str
        Methods for adaptive mask generation ('dropout', 'decay', 'none').
    io_generator : OutputGenerator
        Output generator for saving mask files.
    t2smap_file : str, optional
        Path to pre-computed T2* map for mask generation.
    n_independent_echos : int, optional
        Number of independent echoes for F-stat calculation.

    Returns
    -------
    MaskData
        Container with all mask arrays.
    """
    t2s_limited = None

    if mask_file and not t2smap_file:
        LGR.info("Using user-defined mask")
        RepLGR.info("A user-defined mask was applied to the data.")
        base_mask = utils.reshape_niimg(mask_file).astype(int)
    elif t2smap_file and not mask_file:
        LGR.info("Assuming user-defined T2* map is masked and using it to generate mask")
        t2s_limited_sec = utils.reshape_niimg(t2smap_file)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        base_mask = (t2s_limited != 0).astype(int)
    elif t2smap_file and mask_file:
        LGR.info("Combining user-defined mask and T2* map to generate mask")
        t2s_limited_sec = utils.reshape_niimg(t2smap_file)
        t2s_limited = utils.sec2millisec(t2s_limited_sec)
        base_mask = utils.reshape_niimg(mask_file).astype(int)
        base_mask[t2s_limited == 0] = 0
    else:
        LGR.warning(
            "Computing EPI mask from first echo using nilearn's compute_epi_mask function. "
            "Most external pipelines include more reliable masking functions. "
            "It is strongly recommended to provide an external mask, "
            "and to visually confirm that mask accurately conforms to data boundaries."
        )
        first_echo_img = io.new_nii_like(io_generator.reference_img, data.data_cat[:, 0, :])
        base_mask = compute_epi_mask(first_echo_img).get_fdata()
        base_mask = utils.reshape_niimg(base_mask).astype(int)
        RepLGR.info(
            "An initial mask was generated from the first echo using "
            "nilearn's compute_epi_mask function."
        )

    # Create adaptive mask with at least 1 good echo, for denoising
    mask_denoise, masksum_denoise = utils.make_adaptive_mask(
        data.data_cat,
        mask=base_mask,
        n_independent_echos=n_independent_echos,
        threshold=1,
        methods=masktype,
    )
    LGR.debug(f"Retaining {mask_denoise.sum()}/{data.n_samp} samples for denoising")
    io_generator.save_file(masksum_denoise, "adaptive mask img")

    # Create adaptive mask with at least 3 good echoes, for classification
    masksum_clf = masksum_denoise.copy()
    masksum_clf[masksum_clf < 3] = 0
    mask_clf = masksum_clf.astype(bool)
    RepLGR.info(
        "A two-stage masking procedure was applied, in which a liberal mask "
        "(including voxels with good data in at least the first echo) was used for "
        "optimal combination, T2*/S0 estimation, and denoising, while a more conservative mask "
        "(restricted to voxels with good data in at least the first three echoes) was used for "
        "the component classification procedure."
    )
    LGR.debug(f"Retaining {mask_clf.sum()}/{data.n_samp} samples for classification")

    return MaskData(
        base_mask=base_mask,
        mask_denoise=mask_denoise,
        mask_clf=mask_clf,
        masksum_denoise=masksum_denoise,
        masksum_clf=masksum_clf,
    )


def create_simple_adaptive_mask(
    data_cat: np.ndarray,
    mask: Optional[np.ndarray],
    masktype: List[str],
    n_independent_echos: Optional[int] = None,
) -> tuple:
    """Create a simple adaptive mask (for t2smap workflow).

    Parameters
    ----------
    data_cat : np.ndarray
        Multi-echo data array (S x E x T).
    mask : np.ndarray or None
        Initial mask or None to compute from data.
    masktype : list of str
        Methods for adaptive mask generation.
    n_independent_echos : int, optional
        Number of independent echoes.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Tuple of (mask, masksum).
    """
    return utils.make_adaptive_mask(
        data_cat,
        mask=mask,
        n_independent_echos=n_independent_echos,
        threshold=1,
        methods=masktype,
    )
