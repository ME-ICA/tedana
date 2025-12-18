"""Pipeline context for tedana workflow state management.

This module provides a dataclass that holds all state needed throughout the
tedana workflow, reducing parameter passing and enabling better memory management.
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LGR = logging.getLogger("GENERAL")


@dataclass
class PipelineContext:
    """Container for all tedana workflow state.

    This class centralizes all data and configuration needed throughout
    the tedana workflow, enabling cleaner function interfaces and
    explicit memory management between stages.

    Parameters
    ----------
    data : list of str
        Input data file paths.
    tes : list of float
        Echo times in milliseconds.
    out_dir : str
        Output directory path.

    Attributes
    ----------
    data_cat : np.ndarray or None
        Concatenated multi-echo data array (S x E x T).
    data_optcom : np.ndarray or None
        Optimally combined data array (S x T).
    ref_img : nibabel image or None
        Reference image for output generation.
    mask : np.ndarray or None
        Binary mask array.
    mask_denoise : np.ndarray or None
        Adaptive mask for denoising (at least 1 good echo).
    mask_clf : np.ndarray or None
        Conservative mask for classification (at least 3 good echoes).
    masksum_denoise : np.ndarray or None
        Per-voxel count of good echoes for denoising.
    masksum_clf : np.ndarray or None
        Per-voxel count of good echoes for classification.
    t2s_limited : np.ndarray or None
        T2* map (limited to mask_denoise voxels).
    t2s_full : np.ndarray or None
        Full T2* map.
    s0_limited : np.ndarray or None
        S0 map (limited to mask_denoise voxels).
    s0_full : np.ndarray or None
        Full S0 map.
    mixing : np.ndarray or None
        ICA mixing matrix (T x C).
    mixing_orig : np.ndarray or None
        Original mixing matrix before tedort.
    component_table : pd.DataFrame or None
        Component metrics table.
    selector : ComponentSelector or None
        Component selection object.
    external_regressors : pd.DataFrame or None
        External regressors for component classification.
    io_generator : OutputGenerator or None
        Output file generator.
    """

    # Required inputs
    data: List[str]
    tes: List[float]
    out_dir: str

    # Configuration options
    mask_file: Optional[str] = None
    convention: str = "bids"
    prefix: str = ""
    dummy_scans: int = 0
    masktype: List[str] = field(default_factory=lambda: ["dropout"])
    fittype: str = "loglin"
    combmode: str = "t2s"
    n_independent_echos: Optional[int] = None
    tree: str = "tedana_orig"
    external_regressors_file: Optional[str] = None
    ica_method: str = "fastica"
    n_robust_runs: int = 30
    tedpca: Any = "aic"  # Can be str, float, or int
    fixed_seed: int = 42
    maxit: int = 500
    maxrestart: int = 10
    tedort: bool = False
    gscontrol: Optional[List[str]] = None
    no_reports: bool = False
    png_cmap: str = "coolwarm"
    verbose: bool = False
    low_mem: bool = False
    debug: bool = False
    quiet: bool = False
    overwrite: bool = False
    t2smap_file: Optional[str] = None
    mixing_file: Optional[str] = None
    tedana_command: Optional[str] = None

    # Computed data arrays (set during workflow)
    data_cat: Optional[np.ndarray] = field(default=None, repr=False)
    data_optcom: Optional[np.ndarray] = field(default=None, repr=False)
    data_reduced: Optional[np.ndarray] = field(default=None, repr=False)
    ref_img: Optional[Any] = field(default=None, repr=False)

    # Masks
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    mask_denoise: Optional[np.ndarray] = field(default=None, repr=False)
    mask_clf: Optional[np.ndarray] = field(default=None, repr=False)
    masksum_denoise: Optional[np.ndarray] = field(default=None, repr=False)
    masksum_clf: Optional[np.ndarray] = field(default=None, repr=False)

    # T2*/S0 maps
    t2s_limited: Optional[np.ndarray] = field(default=None, repr=False)
    t2s_full: Optional[np.ndarray] = field(default=None, repr=False)
    s0_limited: Optional[np.ndarray] = field(default=None, repr=False)
    s0_full: Optional[np.ndarray] = field(default=None, repr=False)

    # ICA/decomposition results
    mixing: Optional[np.ndarray] = field(default=None, repr=False)
    mixing_orig: Optional[np.ndarray] = field(default=None, repr=False)
    n_components: Optional[int] = None
    cluster_labels: Optional[np.ndarray] = field(default=None, repr=False)
    similarity_t_sne: Optional[np.ndarray] = field(default=None, repr=False)
    fastica_convergence_warning_count: Optional[int] = None

    # Component selection
    component_table: Optional[pd.DataFrame] = field(default=None, repr=False)
    selector: Optional[Any] = field(default=None, repr=False)
    external_regressors: Optional[pd.DataFrame] = field(default=None, repr=False)

    # I/O management
    io_generator: Optional[Any] = field(default=None, repr=False)
    info_dict: Optional[Dict[str, Any]] = field(default=None, repr=False)
    repname: Optional[str] = None
    bibtex_file: Optional[str] = None

    # Derived properties
    n_echos: Optional[int] = None
    n_samp: Optional[int] = None
    n_vols: Optional[int] = None
    img_t_r: Optional[float] = None

    def __post_init__(self):
        """Initialize derived attributes."""
        # Ensure tes are floats
        self.tes = [float(te) for te in self.tes]
        self.n_echos = len(self.tes)

        # Ensure gscontrol is a list
        if self.gscontrol is None:
            self.gscontrol = []
        elif not isinstance(self.gscontrol, list):
            self.gscontrol = [self.gscontrol]

    @property
    def data_shape(self) -> Optional[Tuple[int, int, int]]:
        """Return shape of data_cat if available."""
        if self.data_cat is not None:
            return self.data_cat.shape
        return None

    def clear_intermediate_data(self, stage: str = "all") -> None:
        """Clear intermediate data to free memory.

        Parameters
        ----------
        stage : str
            Which stage's data to clear. Options:
            - "decomposition": Clear data_reduced after ICA
            - "all": Clear all intermediate arrays
        """
        if stage in ("decomposition", "all"):
            if self.data_reduced is not None:
                LGR.debug("Clearing data_reduced to free memory")
                del self.data_reduced
                self.data_reduced = None

        if stage == "all":
            # Clear large arrays that are no longer needed
            arrays_to_clear = [
                "t2s_limited",
                "s0_limited",
            ]
            for attr in arrays_to_clear:
                if getattr(self, attr, None) is not None:
                    LGR.debug(f"Clearing {attr} to free memory")
                    delattr(self, attr)
                    setattr(self, attr, None)

        # Force garbage collection
        gc.collect()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage of main arrays in MB.

        Returns
        -------
        dict
            Dictionary mapping array names to their memory usage in MB.
        """
        arrays = [
            "data_cat",
            "data_optcom",
            "data_reduced",
            "mask",
            "mask_denoise",
            "mask_clf",
            "masksum_denoise",
            "masksum_clf",
            "t2s_limited",
            "t2s_full",
            "s0_limited",
            "s0_full",
            "mixing",
            "mixing_orig",
        ]

        usage = {}
        for name in arrays:
            arr = getattr(self, name, None)
            if arr is not None and isinstance(arr, np.ndarray):
                usage[name] = arr.nbytes / (1024 * 1024)  # Convert to MB

        return usage

    def log_memory_usage(self) -> None:
        """Log current memory usage of main arrays."""
        usage = self.get_memory_usage()
        total = sum(usage.values())
        LGR.debug(f"Pipeline context memory usage: {total:.2f} MB total")
        for name, mb in sorted(usage.items(), key=lambda x: -x[1]):
            if mb > 1:  # Only log arrays > 1 MB
                LGR.debug(f"  {name}: {mb:.2f} MB")

    def validate_data_loaded(self) -> None:
        """Validate that data has been loaded."""
        if self.data_cat is None:
            raise ValueError("Data not loaded. Run load_data stage first.")
        if self.ref_img is None:
            raise ValueError("Reference image not set. Run load_data stage first.")

    def validate_masks_created(self) -> None:
        """Validate that masks have been created."""
        if self.mask_denoise is None or self.mask_clf is None:
            raise ValueError("Masks not created. Run create_masks stage first.")

    def validate_decay_fitted(self) -> None:
        """Validate that decay fitting has been done."""
        if self.t2s_full is None:
            raise ValueError("T2* map not computed. Run fit_decay stage first.")

    def validate_optcom_computed(self) -> None:
        """Validate that optimal combination has been computed."""
        if self.data_optcom is None:
            raise ValueError(
                "Optimally combined data not computed. Run optimal_combination stage first."
            )

    def validate_decomposition_done(self) -> None:
        """Validate that decomposition has been performed."""
        if self.mixing is None:
            raise ValueError("Mixing matrix not computed. Run decomposition stage first.")

    def validate_selection_done(self) -> None:
        """Validate that component selection has been performed."""
        if self.selector is None or self.component_table is None:
            raise ValueError("Component selection not done. Run component_selection stage first.")


def create_context_from_args(
    data: List[str],
    tes: List[float],
    out_dir: str = ".",
    mask: Optional[str] = None,
    convention: str = "bids",
    prefix: str = "",
    dummy_scans: int = 0,
    masktype: List[str] = None,
    fittype: str = "loglin",
    combmode: str = "t2s",
    n_independent_echos: Optional[int] = None,
    tree: str = "tedana_orig",
    external_regressors: Optional[str] = None,
    ica_method: str = "fastica",
    n_robust_runs: int = 30,
    tedpca: Any = "aic",
    fixed_seed: int = 42,
    maxit: int = 500,
    maxrestart: int = 10,
    tedort: bool = False,
    gscontrol: Optional[List[str]] = None,
    no_reports: bool = False,
    png_cmap: str = "coolwarm",
    verbose: bool = False,
    low_mem: bool = False,
    debug: bool = False,
    quiet: bool = False,
    overwrite: bool = False,
    t2smap: Optional[str] = None,
    mixing_file: Optional[str] = None,
    tedana_command: Optional[str] = None,
) -> PipelineContext:
    """Create a PipelineContext from workflow arguments.

    This factory function creates a PipelineContext with the same signature
    as tedana_workflow for easy migration.

    Parameters
    ----------
    data : list of str
        Input data file paths.
    tes : list of float
        Echo times in milliseconds.
    out_dir : str
        Output directory path.
    mask : str or None
        Path to mask file.
    convention : str
        Filenaming convention ('bids' or 'orig').
    prefix : str
        Prefix for output filenames.
    dummy_scans : int
        Number of dummy scans to remove.
    masktype : list of str
        Methods for adaptive mask generation.
    fittype : str
        T2* fitting method ('loglin' or 'curvefit').
    combmode : str
        Echo combination method.
    n_independent_echos : int or None
        Number of independent echoes for F-stat.
    tree : str
        Decision tree for component classification.
    external_regressors : str or None
        Path to external regressors file.
    ica_method : str
        ICA method ('fastica' or 'robustica').
    n_robust_runs : int
        Number of robust ICA runs.
    tedpca : str, float, or int
        PCA component selection method.
    fixed_seed : int
        Random seed for reproducibility.
    maxit : int
        Maximum ICA iterations.
    maxrestart : int
        Maximum ICA restarts.
    tedort : bool
        Whether to orthogonalize rejected components.
    gscontrol : list of str or None
        Global signal control methods.
    no_reports : bool
        Whether to skip report generation.
    png_cmap : str
        Colormap for figures.
    verbose : bool
        Whether to generate verbose output.
    low_mem : bool
        Whether to use low-memory processing.
    debug : bool
        Whether to enable debug mode.
    quiet : bool
        Whether to suppress logging.
    overwrite : bool
        Whether to overwrite existing files.
    t2smap : str or None
        Path to pre-computed T2* map.
    mixing_file : str or None
        Path to pre-computed mixing matrix.
    tedana_command : str or None
        Command string for provenance.

    Returns
    -------
    PipelineContext
        Initialized pipeline context.
    """
    if masktype is None:
        masktype = ["dropout"]

    # Handle single file vs list
    if isinstance(data, str):
        data = [data]

    return PipelineContext(
        data=data,
        tes=tes,
        out_dir=out_dir,
        mask_file=mask,
        convention=convention,
        prefix=prefix,
        dummy_scans=dummy_scans,
        masktype=masktype,
        fittype=fittype,
        combmode=combmode,
        n_independent_echos=n_independent_echos,
        tree=tree,
        external_regressors_file=external_regressors,
        ica_method=ica_method,
        n_robust_runs=n_robust_runs,
        tedpca=tedpca,
        fixed_seed=fixed_seed,
        maxit=maxit,
        maxrestart=maxrestart,
        tedort=tedort,
        gscontrol=gscontrol,
        no_reports=no_reports,
        png_cmap=png_cmap,
        verbose=verbose,
        low_mem=low_mem,
        debug=debug,
        quiet=quiet,
        overwrite=overwrite,
        t2smap_file=t2smap,
        mixing_file=mixing_file,
        tedana_command=tedana_command,
    )
