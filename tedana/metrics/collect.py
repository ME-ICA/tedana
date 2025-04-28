"""Tools to collect and generate metrics."""

import logging
import os.path as op
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from tedana import io, utils
from tedana.metrics import dependence, external
from tedana.metrics._utils import (
    add_external_dependencies,
    dependency_resolver,
    determine_signs,
    flip_components,
)
from tedana.stats import getfbounds

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")


def generate_metrics(
    *,
    data_cat: npt.NDArray,
    data_optcom: npt.NDArray,
    mixing: npt.NDArray,
    adaptive_mask: npt.NDArray,
    tes: Union[List[int], List[float], npt.NDArray],
    n_independent_echos: int = None,
    io_generator: io.OutputGenerator,
    label: str,
    external_regressors: Union[pd.DataFrame, None] = None,
    external_regressor_config: Union[List[Dict], None] = None,
    metrics: Union[List[str], None] = None,
) -> Tuple[pd.DataFrame, npt.NDArray]:
    """Fit TE-dependence and -independence models to components.

    Parameters
    ----------
    data_cat : (S x E x T) array_like
        Input data, where `S` is samples, `E` is echos, and `T` is time
    data_optcom : (S x T) array_like
        Optimally combined data
    mixing : (T x C) array_like
        Mixing matrix for converting input data to component space,
        where `C` is components and `T` is the same as in `data_cat`
    adaptive_mask : (S) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel.
        This mask may be thresholded; for example, with values less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    tes : list
        List of echo times associated with `data_cat`, in milliseconds
    n_independent_echos : int
        Number of independent echoes to use in goodness of fit metrics (fstat).
        Primarily used for EPTI acquisitions.
        If None, number of echoes will be used. Default is None.
    io_generator : tedana.io.OutputGenerator
        The output generator object for this workflow
    label : str in ['ICA', 'PCA']
        The label for this metric generation type
    external_regressors : None or :obj:`pandas.DataFrame`, optional
        External regressors (e.g., motion parameters, physiological noise)
        to correlate with ICA components.
        If None, no external regressor metrics will be calculated.
    external_regressor_config : :obj:`list[dic]t`
        A list of dictionaries defining how to fit external regressors to component time series
    metrics : list
        List of metrics to return

    Returns
    -------
    component_table : (C x X) :obj:`pandas.DataFrame`
        Component metric table. One row for each component, with a column for each metric.
        The index is the component number.
    mixing : (T x C) array_like
        Mixing matrix for converting input data to component space,
        where `C` is components and `T` is the same as in `data_cat`
        The signs of a component time series are flipped so that the components spatial map
        has more positive voxel weights
    """
    # Load metric dependency tree from json file
    dependency_config = op.join(utils.get_resource_path(), "config", "metrics.json")
    dependency_config = io.load_json(dependency_config)

    if metrics is None:
        metrics = ["map weight"]

    if external_regressors is not None:
        if external_regressor_config is None:
            raise ValueError(
                "If external_regressors is defined, then "
                "external_regressor_config also needs to be defined."
            )
        dependency_config = add_external_dependencies(dependency_config, external_regressor_config)
        dependency_config["inputs"].append("external regressors")

    RepLGR.info(f"The following metrics were calculated: {', '.join(metrics)}.")

    if not (data_cat.shape[0] == data_optcom.shape[0] == adaptive_mask.shape[0]):
        raise ValueError(
            f"First dimensions (number of samples) of data_cat ({data_cat.shape[0]}), "
            f"data_optcom ({data_optcom.shape[0]}), and adaptive_mask "
            f"({adaptive_mask.shape[0]}) do not match"
        )
    elif data_cat.shape[1] != len(tes):
        raise ValueError(
            f"Second dimension of data_cat ({data_cat.shape[1]}) does not match "
            f"number of echoes provided ({tes}; {len(tes)})"
        )
    elif not (data_cat.shape[2] == data_optcom.shape[1] == mixing.shape[0]):
        raise ValueError(
            f"Number of volumes in data_cat ({data_cat.shape[2]}), "
            f"data_optcom ({data_optcom.shape[1]}), and mixing ({mixing.shape[0]}) do not "
            "match."
        )

    # Derive mask from thresholded adaptive mask
    mask = adaptive_mask >= 3

    # Apply masks before anything else
    data_cat = data_cat[mask, ...]
    data_optcom = data_optcom[mask, :]
    adaptive_mask = adaptive_mask[mask]

    # Ensure that echo times are in an array, rather than a list
    tes = np.asarray(tes)

    # use either the inputted number of indie echoes or the total number of echoes
    # to calculate the threshold for f tests
    f_thresh, _, _ = getfbounds(n_independent_echos or len(tes))

    # Get reference image from io_generator
    ref_img = io_generator.reference_img

    required_metrics = dependency_resolver(
        dependency_config["dependencies"],
        metrics,
        dependency_config["inputs"],
    )

    # Use copy to avoid changing the original variable outside of this function
    mixing = mixing.copy()

    # Generate the component table, which will be filled out, column by column,
    # throughout this function
    n_components = mixing.shape[1]
    component_table = pd.DataFrame(index=np.arange(n_components, dtype=int))
    component_table["Component"] = [
        io.add_decomp_prefix(comp, prefix=label, max_value=component_table.shape[0])
        for comp in component_table.index.values
    ]

    # Metric maps
    # Maps will be stored as arrays in an easily-indexable dictionary
    metric_maps = {}
    if "map weight" in required_metrics:
        LGR.info("Calculating weight maps")
        metric_maps["map weight"] = dependence.calculate_weights(
            data_optcom=data_optcom,
            mixing=mixing,
        )
        signs = determine_signs(metric_maps["map weight"], axis=0)
        component_table["optimal sign"] = signs
        metric_maps["map weight"], mixing = flip_components(
            metric_maps["map weight"], mixing, signs=signs
        )

    if "map optcom betas" in required_metrics:
        LGR.info("Calculating parameter estimate maps for optimally combined data")
        metric_maps["map optcom betas"] = dependence.calculate_betas(
            data=data_optcom,
            mixing=mixing,
        )
        if io_generator.verbose:
            metric_maps["map echo betas"] = dependence.calculate_betas(
                data=data_cat,
                mixing=mixing,
            )

    if "map percent signal change" in required_metrics:
        LGR.info("Calculating percent signal change maps")
        # used in kundu v3.2 tree
        metric_maps["map percent signal change"] = dependence.calculate_psc(
            data_optcom=data_optcom,
            optcom_betas=metric_maps["map optcom betas"],
        )

    if "map Z" in required_metrics:
        LGR.info("Calculating z-statistic maps")
        metric_maps["map Z"] = dependence.calculate_z_maps(weights=metric_maps["map weight"])

        if io_generator.verbose:
            io_generator.save_file(
                utils.unmask(metric_maps["map Z"] ** 2, mask),
                f"{label} component weights img",
            )

    if ("map FT2" in required_metrics) or ("map FS0" in required_metrics):
        LGR.info("Calculating F-statistic maps")
        m_t2, m_s0, p_m_t2, p_m_s0 = dependence.calculate_f_maps(
            data_cat=data_cat,
            z_maps=metric_maps["map Z"],
            mixing=mixing,
            adaptive_mask=adaptive_mask,
            tes=tes,
            n_independent_echos=n_independent_echos,
        )
        metric_maps["map FT2"] = m_t2
        metric_maps["map FS0"] = m_s0
        metric_maps["map predicted T2"] = p_m_t2
        metric_maps["map predicted S0"] = p_m_s0

        if io_generator.verbose:
            io_generator.save_file(
                utils.unmask(metric_maps["map FT2"], mask),
                f"{label} component F-T2 img",
            )
            io_generator.save_file(
                utils.unmask(metric_maps["map FS0"], mask),
                f"{label} component F-S0 img",
            )

    if "map Z clusterized" in required_metrics:
        LGR.info("Thresholding z-statistic maps")
        z_thresh = 1.95
        metric_maps["map Z clusterized"] = dependence.threshold_map(
            maps=metric_maps["map Z"],
            mask=mask,
            ref_img=ref_img,
            threshold=z_thresh,
        )

    if "map FT2 clusterized" in required_metrics:
        LGR.info("Calculating T2* F-statistic maps")

        metric_maps["map FT2 clusterized"] = dependence.threshold_map(
            maps=metric_maps["map FT2"],
            mask=mask,
            ref_img=ref_img,
            threshold=f_thresh,
        )

    if "map FS0 clusterized" in required_metrics:
        LGR.info("Calculating S0 F-statistic maps")

        metric_maps["map FS0 clusterized"] = dependence.threshold_map(
            maps=metric_maps["map FS0"],
            mask=mask,
            ref_img=ref_img,
            threshold=f_thresh,
        )

    # Intermediate metrics
    if "countsigFT2" in required_metrics:
        LGR.info("Counting significant voxels in T2* F-statistic maps")
        component_table["countsigFT2"] = dependence.compute_countsignal(
            stat_cl_maps=metric_maps["map FT2 clusterized"],
        )

    if "countsigFS0" in required_metrics:
        LGR.info("Counting significant voxels in S0 F-statistic maps")
        component_table["countsigFS0"] = dependence.compute_countsignal(
            stat_cl_maps=metric_maps["map FS0 clusterized"],
        )

    # Back to maps
    if "map beta T2 clusterized" in required_metrics:
        LGR.info("Thresholding optimal combination beta maps to match T2* F-statistic maps")
        metric_maps["map beta T2 clusterized"] = dependence.threshold_to_match(
            maps=metric_maps["map optcom betas"],
            n_sig_voxels=component_table["countsigFT2"],
            mask=mask,
            ref_img=ref_img,
        )

    if "map beta S0 clusterized" in required_metrics:
        LGR.info("Thresholding optimal combination beta maps to match S0 F-statistic maps")
        metric_maps["map beta S0 clusterized"] = dependence.threshold_to_match(
            maps=metric_maps["map optcom betas"],
            n_sig_voxels=component_table["countsigFS0"],
            mask=mask,
            ref_img=ref_img,
        )

    # Dependence metrics
    if ("kappa" in required_metrics) or ("rho" in required_metrics):
        LGR.info("Calculating kappa and rho")
        component_table["kappa"], component_table["rho"] = dependence.calculate_dependence_metrics(
            f_t2_maps=metric_maps["map FT2"],
            f_s0_maps=metric_maps["map FS0"],
            z_maps=metric_maps["map Z"],
        )

    # Generic metrics
    if "variance explained" in required_metrics:
        LGR.info("Calculating variance explained")
        component_table["variance explained"] = dependence.calculate_varex(
            optcom_betas=metric_maps["map optcom betas"],
        )

    if "normalized variance explained" in required_metrics:
        LGR.info("Calculating normalized variance explained")
        component_table["normalized variance explained"] = dependence.calculate_varex_norm(
            weights=metric_maps["map weight"],
        )

    # Spatial metrics
    if "dice_FT2" in required_metrics:
        LGR.info(
            "Calculating DSI between thresholded T2* F-statistic and optimal combination beta maps"
        )
        component_table["dice_FT2"] = dependence.compute_dice(
            clmaps1=metric_maps["map beta T2 clusterized"],
            clmaps2=metric_maps["map FT2 clusterized"],
            axis=0,
        )

    if "dice_FS0" in required_metrics:
        LGR.info(
            "Calculating DSI between thresholded S0 F-statistic and "
            "optimal combination beta maps"
        )
        component_table["dice_FS0"] = dependence.compute_dice(
            clmaps1=metric_maps["map beta S0 clusterized"],
            clmaps2=metric_maps["map FS0 clusterized"],
            axis=0,
        )

    if "signal-noise_t" in required_metrics:
        LGR.info("Calculating signal-noise t-statistics")
        RepLGR.info(
            "A t-test was performed between the distributions of T2*-model F-statistics "
            "associated with clusters (i.e., signal) and non-cluster voxels (i.e., noise) to "
            "generate a t-statistic (metric signal-noise_z) and p-value (metric signal-noise_p) "
            "measuring relative association of the component to signal over noise."
        )
        (
            component_table["signal-noise_t"],
            component_table["signal-noise_p"],
        ) = dependence.compute_signal_minus_noise_t(
            z_maps=metric_maps["map Z"],
            z_clmaps=metric_maps["map Z clusterized"],
            f_t2_maps=metric_maps["map FT2"],
        )

    if "signal-noise_z" in required_metrics:
        LGR.info("Calculating signal-noise z-statistics")
        RepLGR.info(
            "A t-test was performed between the distributions of T2*-model F-statistics "
            "associated with clusters (i.e., signal) and non-cluster voxels (i.e., noise) to "
            "generate a z-statistic (metric signal-noise_z) and p-value (metric signal-noise_p) "
            "measuring relative association of the component to signal over noise."
        )
        (
            component_table["signal-noise_z"],
            component_table["signal-noise_p"],
        ) = dependence.compute_signal_minus_noise_z(
            z_maps=metric_maps["map Z"],
            z_clmaps=metric_maps["map Z clusterized"],
            f_t2_maps=metric_maps["map FT2"],
        )

    if "countnoise" in required_metrics:
        LGR.info("Counting significant noise voxels from z-statistic maps")
        RepLGR.info(
            "The number of significant voxels not from clusters was "
            "calculated for each component."
        )
        component_table["countnoise"] = dependence.compute_countnoise(
            stat_maps=metric_maps["map Z"],
            stat_cl_maps=metric_maps["map Z clusterized"],
        )

    # Composite metrics
    if "d_table_score" in required_metrics:
        LGR.info("Calculating decision table score")
        component_table["d_table_score"] = dependence.generate_decision_table_score(
            kappa=component_table["kappa"],
            dice_ft2=component_table["dice_FT2"],
            signal_minus_noise_t=component_table["signal-noise_t"],
            countnoise=component_table["countnoise"],
            countsig_ft2=component_table["countsigFT2"],
        )

    # External regressor-based metrics
    if external_regressors is not None and external_regressor_config is not None:
        # external_regressor_names = external_regressors.columns.tolist()
        for config_idx in range(len(external_regressor_config)):
            LGR.info(
                "Calculating external regressor fits. "
                f"{external_regressor_config[config_idx]['info']}"
            )
            RepLGR.info({external_regressor_config[config_idx]["report"]})

        component_table = external.fit_regressors(
            component_table, external_regressors, external_regressor_config, mixing
        )
    # Write verbose metrics if needed
    if io_generator.verbose:
        write_betas = "map echo betas" in metric_maps
        write_t2s0 = "map predicted T2" in metric_maps
        if write_betas:
            betas = metric_maps["map echo betas"]

        if write_t2s0:
            pred_t2_maps = metric_maps["map predicted T2"]
            pred_s0_maps = metric_maps["map predicted S0"]

        for i_echo in range(len(tes)):
            if write_betas:
                echo_betas = betas[:, i_echo, :]
                io_generator.save_file(
                    utils.unmask(echo_betas, mask),
                    f"echo weight {label} map split img",
                    echo=(i_echo + 1),
                )

            if write_t2s0:
                echo_pred_t2_maps = pred_t2_maps[:, i_echo, :]
                io_generator.save_file(
                    utils.unmask(echo_pred_t2_maps, mask),
                    f"echo T2 {label} split img",
                    echo=(i_echo + 1),
                )

                echo_pred_s0_maps = pred_s0_maps[:, i_echo, :]
                io_generator.save_file(
                    utils.unmask(echo_pred_s0_maps, mask),
                    f"echo S0 {label} split img",
                    echo=(i_echo + 1),
                )

    # Reorder component table columns based on previous tedana versions
    # NOTE: Some new columns will be calculated and columns may be reordered during
    # component selection
    preferred_order = (
        "Component",
        "kappa",
        "rho",
        "variance explained",
        "normalized variance explained",
        "estimated normalized variance explained",
        "countsigFT2",
        "countsigFS0",
        "dice_FT2",
        "dice_FS0",
        "countnoise",
        "signal-noise_t",
        "signal-noise_p",
        "d_table_score",
        "kappa ratio",
        "d_table_score_scrub",
        "external fit",
        "classification",
        "rationale",
    )
    first_columns = [col for col in preferred_order if col in component_table.columns]
    other_columns = [col for col in component_table.columns if col not in preferred_order]
    component_table = component_table[first_columns + other_columns]

    return component_table, mixing


def get_metadata(component_table: pd.DataFrame) -> Dict:
    """Fill in metric metadata for a given component_table.

    Parameters
    ----------
    component_table : pandas.DataFrame
        The component table for this workflow

    Returns
    -------
    metric_metadata : dict
        The metadata for each column in the component_table for which we have a metadata
        description, plus the "Component" metadata description (always).
    """
    metric_metadata = {}
    if "kappa" in component_table:
        metric_metadata["kappa"] = {
            "LongName": "Kappa",
            "Description": (
                "A pseudo-F-statistic indicating TE-dependence of the "
                "component. This metric is calculated by computing fit to "
                "the TE-dependence model at each voxel, and then "
                "performing a weighted average based on the voxel-wise "
                "weights of the component."
            ),
            "Units": "arbitrary",
        }
    if "rho" in component_table:
        metric_metadata["rho"] = {
            "LongName": "Rho",
            "Description": (
                "A pseudo-F-statistic indicating TE-independence of the "
                "component. This metric is calculated by computing fit to "
                "the TE-independence model at each voxel, and then "
                "performing a weighted average based on the voxel-wise "
                "weights of the component."
            ),
            "Units": "arbitrary",
        }
    if "variance explained" in component_table:
        metric_metadata["variance explained"] = {
            "LongName": "Variance explained",
            "Description": (
                "Variance explained in the optimally combined data of "
                "each component. On a scale from 0 to 100."
            ),
            "Units": "arbitrary",
        }
    if "normalized variance explained" in component_table:
        metric_metadata["normalized variance explained"] = {
            "LongName": "Normalized variance explained",
            "Description": (
                "Normalized variance explained in the optimally combined "
                "data of each component."
                "On a scale from 0 to 1."
            ),
            "Units": "arbitrary",
        }
    if "countsigFT2" in component_table:
        metric_metadata["countsigFT2"] = {
            "LongName": "T2 model F-statistic map significant voxel count",
            "Description": (
                "Number of significant voxels from the cluster-extent "
                "thresholded T2 model F-statistic map for each component."
            ),
            "Units": "voxel",
        }
    if "countsigFS0" in component_table:
        metric_metadata["countsigFS0"] = {
            "LongName": "S0 model F-statistic map significant voxel count",
            "Description": (
                "Number of significant voxels from the cluster-extent "
                "thresholded S0 model F-statistic map for each component."
            ),
            "Units": "voxel",
        }
    if "dice_FT2" in component_table:
        metric_metadata["dice_FT2"] = {
            "LongName": "T2 model beta map-F-statistic map Dice similarity index",
            "Description": (
                "Dice value of cluster-extent thresholded maps of "
                "T2-model betas and F-statistics."
            ),
            "Units": "arbitrary",
        }
    if "dice_FS0" in component_table:
        metric_metadata["dice_FS0"] = {
            "LongName": ("S0 model beta map-F-statistic map Dice similarity index"),
            "Description": (
                "Dice value of cluster-extent thresholded maps of "
                "S0-model betas and F-statistics."
            ),
            "Units": "arbitrary",
        }
    if "countnoise" in component_table:
        metric_metadata["countnoise"] = {
            "LongName": "Noise voxel count",
            "Description": (
                "Number of 'noise' voxels (voxels highly weighted for "
                "component, but not from clusters) from each component."
            ),
            "Units": "voxel",
        }
    if "signal-noise_t" in component_table:
        metric_metadata["signal-noise_t"] = {
            "LongName": "Signal > noise t-statistic",
            "Description": (
                "T-statistic for two-sample t-test of F-statistics from "
                "'signal' voxels (voxels in clusters) against 'noise' "
                "voxels (voxels not in clusters) for T2 model."
            ),
            "Units": "arbitrary",
        }
    if "signal-noise_p" in component_table:
        metric_metadata["signal-noise_p"] = {
            "LongName": "Signal > noise p-value",
            "Description": (
                "P-value for two-sample t-test of F-statistics from "
                "'signal' voxels (voxels in clusters) against 'noise' "
                "voxels (voxels not in clusters) for T2 model."
            ),
            "Units": "arbitrary",
        }
    if "d_table_score" in component_table:
        metric_metadata["d_table_score"] = {
            "LongName": "Decision table score",
            "Description": (
                "Summary score compiled from five metrics, with smaller "
                "values (i.e., higher ranks) indicating more BOLD "
                "dependence and less noise."
            ),
            "Units": "arbitrary",
        }
    if "original_classification" in component_table:
        metric_metadata["original_classification"] = {
            "LongName": "Original classification",
            "Description": ("Classification from the original decision tree."),
            "Levels": {
                "accepted": ("A BOLD-like component included in denoised and high-Kappa data."),
                "rejected": ("A non-BOLD component excluded from denoised and high-Kappa data."),
                "ignored": (
                    "A low-variance component included in denoised, "
                    "but excluded from high-Kappa data."
                ),
            },
        }

    if "classification" in component_table:
        metric_metadata["classification"] = {
            "LongName": "Component classification",
            "Description": ("Classification from the manual classification procedure."),
            "Levels": {
                "accepted": ("A BOLD-like component included in denoised and high-Kappa data."),
                "rejected": ("A non-BOLD component excluded from denoised and high-Kappa data."),
                "ignored": (
                    "A low-variance component included in denoised, "
                    "but excluded from high-Kappa data."
                ),
            },
        }
    if "classification_tags" in component_table:
        metric_metadata["classification_tags"] = {
            "LongName": "Component classification tags",
            "Description": (
                "A single tag or a comma separated list of tags to describe why a component"
                " received its classification"
            ),
        }
    if "rationale" in component_table:
        metric_metadata["rationale"] = {
            "LongName": "Rationale for component classification",
            "Description": (
                "The reason for the original classification. "
                "This column label was replaced with classification_tags in late 2022"
            ),
        }
    if "kappa ratio" in component_table:
        metric_metadata["kappa ratio"] = {
            "LongName": "Kappa ratio",
            "Description": (
                "Ratio score calculated by dividing range of kappa "
                "values by range of variance explained values."
            ),
            "Units": "arbitrary",
        }
    if "d_table_score_scrub" in component_table:
        metric_metadata["d_table_score_scrub"] = {
            "LongName": "Updated decision table score",
            "Description": (
                "Summary score compiled from five metrics and computed "
                "from a subset of components, with smaller values "
                "(i.e., higher ranks) indicating more BOLD dependence "
                "and less noise."
            ),
            "Units": "arbitrary",
        }
    if "optimal sign" in component_table:
        metric_metadata["optimal sign"] = {
            "LongName": "Optimal component sign",
            "Description": (
                "Optimal sign determined based on skew direction of component parameter estimates "
                "across the brain. In cases where components were left-skewed (-1), the component "
                "time series and map weights are flipped prior to metric calculation. "
                "This sign applies to the original mixing matrix and map weights. "
                "The outputs produced by tedana are already flipped."
            ),
            "Levels": {
                1: "Component is not flipped prior to metric calculation.",
                -1: "Component is flipped prior to metric calculation.",
            },
        }
    if "Var Exp of rejected to accepted" in component_table:
        metric_metadata["Var Exp of rejected to accepted"] = {
            "LongName": "100*R2 of fit of rejected to accepted mixing matrix time series",
            "Description": (
                "The time series for each component are not independent. "
                "Time series from rejected components are regressed from the data. "
                "For each accepted component, this is a calculation of how much "
                "the variance of the accept component times series is explained "
                "by the rejected component time series. "
                "This is not used in decision trees but might be a useful quality metric."
            ),
            "Units": "percent",
        }

    # There are always components in the component_table, definitionally
    metric_metadata["Component"] = {
        "LongName": "Component identifier",
        "Description": (
            "The unique identifier of each component. "
            "This identifier matches column names in the mixing matrix "
            "TSV file."
        ),
    }

    return metric_metadata
