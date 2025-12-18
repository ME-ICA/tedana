"""Data containers for tedana workflows.

This module provides focused dataclasses for passing data between
workflow stages. Each container holds related data and provides
a clear interface for what each stage needs and produces.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd


@dataclass
class WorkflowConfig:
    """Common workflow configuration parameters.

    Parameters
    ----------
    out_dir : str
        Absolute path to output directory.
    prefix : str
        Prefix for output filenames.
    convention : str
        Filenaming convention ('bids' or 'orig').
    verbose : bool
        Whether to generate verbose output.
    debug : bool
        Whether to run in debug mode.
    quiet : bool
        Whether to suppress logging.
    overwrite : bool
        Whether to overwrite existing files.
    """

    out_dir: str
    prefix: str
    convention: str = "bids"
    verbose: bool = False
    debug: bool = False
    quiet: bool = False
    overwrite: bool = False


@dataclass
class MultiEchoData:
    """Container for loaded multi-echo fMRI data.

    Parameters
    ----------
    data_cat : np.ndarray
        Concatenated multi-echo data array with shape (S x E x T),
        where S is samples (voxels), E is echoes, T is timepoints.
    ref_img : nibabel image
        Reference image for output generation.
    tes : list of float
        Echo times in milliseconds.
    n_samp : int
        Number of samples (voxels).
    n_echos : int
        Number of echoes.
    n_vols : int
        Number of volumes (timepoints).
    """

    data_cat: np.ndarray = field(repr=False)
    ref_img: Any = field(repr=False)
    tes: List[float]
    n_samp: int
    n_echos: int
    n_vols: int

    @classmethod
    def from_data_array(
        cls, data_cat: np.ndarray, ref_img: Any, tes: List[float]
    ) -> "MultiEchoData":
        """Create MultiEchoData from a data array.

        Parameters
        ----------
        data_cat : np.ndarray
            Concatenated multi-echo data (S x E x T).
        ref_img : nibabel image
            Reference image.
        tes : list of float
            Echo times in milliseconds.

        Returns
        -------
        MultiEchoData
            Initialized container.
        """
        n_samp, n_echos, n_vols = data_cat.shape
        return cls(
            data_cat=data_cat,
            ref_img=ref_img,
            tes=tes,
            n_samp=n_samp,
            n_echos=n_echos,
            n_vols=n_vols,
        )


@dataclass
class MaskData:
    """Container for mask arrays.

    Parameters
    ----------
    base_mask : np.ndarray
        Initial binary mask (from user or computed).
    mask_denoise : np.ndarray
        Liberal mask for denoising (voxels with >= 1 good echo).
    mask_clf : np.ndarray
        Conservative mask for classification (voxels with >= 3 good echoes).
    masksum_denoise : np.ndarray
        Per-voxel count of good echoes for denoising mask.
    masksum_clf : np.ndarray
        Per-voxel count of good echoes for classification mask.
    """

    base_mask: np.ndarray = field(repr=False)
    mask_denoise: np.ndarray = field(repr=False)
    mask_clf: np.ndarray = field(repr=False)
    masksum_denoise: np.ndarray = field(repr=False)
    masksum_clf: np.ndarray = field(repr=False)

    @property
    def n_voxels_denoise(self) -> int:
        """Number of voxels in denoising mask."""
        return int(self.mask_denoise.sum())

    @property
    def n_voxels_clf(self) -> int:
        """Number of voxels in classification mask."""
        return int(self.mask_clf.sum())


@dataclass
class DecayMaps:
    """Container for T2* and S0 maps.

    Parameters
    ----------
    t2s_limited : np.ndarray
        T2* map limited to mask_denoise voxels (in milliseconds).
    t2s_full : np.ndarray
        Full T2* map with extrapolation for dropout voxels (in milliseconds).
    s0_limited : np.ndarray
        S0 map limited to mask_denoise voxels.
    s0_full : np.ndarray
        Full S0 map with extrapolation for dropout voxels.
    """

    t2s_limited: np.ndarray = field(repr=False)
    t2s_full: np.ndarray = field(repr=False)
    s0_limited: np.ndarray = field(repr=False)
    s0_full: np.ndarray = field(repr=False)


@dataclass
class OptcomData:
    """Container for optimally combined data.

    Parameters
    ----------
    data_optcom : np.ndarray
        Optimally combined timeseries with shape (S x T).
    """

    data_optcom: np.ndarray = field(repr=False)

    @property
    def n_samp(self) -> int:
        """Number of samples (voxels)."""
        return self.data_optcom.shape[0]

    @property
    def n_vols(self) -> int:
        """Number of volumes (timepoints)."""
        return self.data_optcom.shape[1]


@dataclass
class DecompositionResult:
    """Container for PCA/ICA decomposition results.

    Parameters
    ----------
    mixing : np.ndarray
        ICA mixing matrix with shape (T x C), where T is timepoints
        and C is components.
    n_components : int
        Number of components.
    data_reduced : np.ndarray, optional
        PCA-reduced data for ICA input.
    cluster_labels : np.ndarray, optional
        Cluster labels from robustica (if used).
    similarity_t_sne : np.ndarray, optional
        t-SNE similarity from robustica (if used).
    convergence_warning_count : int, optional
        Number of convergence warnings from FastICA.
    index_quality : float, optional
        Mean index quality from robustica.
    """

    mixing: np.ndarray = field(repr=False)
    n_components: int
    data_reduced: Optional[np.ndarray] = field(default=None, repr=False)
    cluster_labels: Optional[np.ndarray] = field(default=None, repr=False)
    similarity_t_sne: Optional[np.ndarray] = field(default=None, repr=False)
    convergence_warning_count: Optional[int] = None
    index_quality: Optional[float] = None

    @property
    def n_vols(self) -> int:
        """Number of volumes (timepoints)."""
        return self.mixing.shape[0]


@dataclass
class SelectionResult:
    """Container for component selection results.

    Parameters
    ----------
    selector : ComponentSelector
        Component selector object with selection results.
    component_table : pd.DataFrame
        Component metrics table with classifications.
    n_accepted : int
        Number of accepted components.
    n_rejected : int
        Number of rejected components.
    """

    selector: Any
    component_table: pd.DataFrame = field(repr=False)
    n_accepted: int
    n_rejected: int

    @classmethod
    def from_selector(cls, selector: Any) -> "SelectionResult":
        """Create SelectionResult from a ComponentSelector.

        Parameters
        ----------
        selector : ComponentSelector
            Selector after running selection.

        Returns
        -------
        SelectionResult
            Initialized container.
        """
        return cls(
            selector=selector,
            component_table=selector.component_table_,
            n_accepted=selector.n_accepted_comps_,
            n_rejected=selector.n_rejected_comps_,
        )
