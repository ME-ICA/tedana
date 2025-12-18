"""Data loading utilities for tedana workflows.

This module provides functions for loading multi-echo fMRI data.
"""

import logging
from typing import List, Union

from tedana import io, utils
from tedana.workflows.shared.containers import MultiEchoData

LGR = logging.getLogger("GENERAL")


def load_multiecho_data(
    data: Union[str, List[str]],
    tes: List[float],
    dummy_scans: int = 0,
) -> MultiEchoData:
    """Load and validate multi-echo fMRI data.

    Parameters
    ----------
    data : str or list of str
        Either a single z-concatenated file or a list of echo-specific files.
    tes : list of float
        Echo times in milliseconds.
    dummy_scans : int, optional
        Number of dummy scans to remove. Default is 0.

    Returns
    -------
    MultiEchoData
        Container with loaded data.
    """
    # Ensure tes are floats and validate
    tes = [float(te) for te in tes]
    tes = utils.check_te_values(tes)
    n_echos = len(tes)

    # Ensure data is a list
    if isinstance(data, str):
        data = [data]

    LGR.info(f"Loading input data: {[f for f in data]}")
    data_cat, ref_img = io.load_data(data, n_echos=n_echos, dummy_scans=dummy_scans)

    n_samp, n_echos_loaded, n_vols = data_cat.shape
    LGR.debug(f"Resulting data shape: {data_cat.shape}")

    return MultiEchoData(
        data_cat=data_cat,
        ref_img=ref_img,
        tes=tes,
        n_samp=n_samp,
        n_echos=n_echos_loaded,
        n_vols=n_vols,
    )


def validate_tr(ref_img) -> float:
    """Validate that TR is non-zero and return it.

    Parameters
    ----------
    ref_img : nibabel image
        Reference image to get TR from.

    Returns
    -------
    float
        TR value from image header.

    Raises
    ------
    OSError
        If TR is 0.
    """
    img_t_r = ref_img.header.get_zooms()[-1]
    if img_t_r == 0:
        raise OSError(
            "Dataset has a TR of 0. This indicates incorrect"
            " header information. To correct this, we recommend"
            " using this snippet:"
            "\n"
            "https://gist.github.com/jbteves/032c87aeb080dd8de8861cb151bff5d6"
            "\n"
            "to correct your TR to the value it should be."
        )
    return img_t_r
