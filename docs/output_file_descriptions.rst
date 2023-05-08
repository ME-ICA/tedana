#############################
Output file name descriptions
#############################

tedana allows for multiple file naming conventions. The key labels and naming options for
each convention that can be set using the ``--convention`` option are in `outputs.json`_.
The output of ``tedana`` also includes a file called ``registry.json`` or
``desc-tedana_registry.json`` that includes the keys and the matching file names for the
output.
The table below lists both these keys and the default "BIDS Derivatives" file names.

.. _outputs.json: https://github.com/ME-ICA/tedana/blob/main/tedana/resources/config/outputs.json

===========================================================================  =====================================================
Key: Filename                                                                Content
===========================================================================  =====================================================
"registry json": desc-tedana_registry.json                                   Mapping of file name keys to filename locations
"data description json": dataset_description.json                            Top-level metadata for the workflow.
tedana_report.html                                                           The interactive HTML report.
"combined img": desc-optcom_bold.nii.gz                                      Optimally combined time series.
"denoised ts img": desc-optcomDenoised_bold.nii.gz                           Denoised optimally combined time series. Recommended
                                                                             dataset for analysis.
"adaptive mask img": desc-adaptiveGoodSignal_mask.nii.gz                     Integer-valued mask used in the workflow, where
                                                                             each voxel's value corresponds to the number of good
                                                                             echoes to be used for T2\*/S0 estimation. Will be
                                                                             calculated whether original mask estimated within
                                                                             tedana or user-provided. All voxels with 1 good
                                                                             echo will be included in outputted time series
                                                                             but only voxels with at least 3 good echoes will be
                                                                             used in ICA and metric calculations
"t2star img": T2starmap.nii.gz                                               Full estimated T2* 3D map.
                                                                             Values are in seconds. If a voxel has at least 1 good
                                                                             echo then the first two echoes will be used to estimate
                                                                             a value (an impresise weighting for optimal combination
                                                                             is better than fully excluding a voxel)
"s0 img": S0map.nii.gz                                                       Full S0 3D map. If a voxel has at least 1 good
                                                                             echo then the first two echoes will be used to estimate
                                                                             a value
"PCA mixing tsv": desc-PCA_mixing.tsv                                        Mixing matrix (component time series) from PCA
                                                                             decomposition in a tab-delimited file. Each column is
                                                                             a different component, and the column name is the
                                                                             component number.
"PCA decomposition json": desc-PCA_decomposition.json                        Metadata for the PCA decomposition.
"z-scored PCA components img": desc-PCA_stat-z_components.nii.gz             Component weight maps from PCA decomposition.
                                                                             Each map corresponds to the same component index in
                                                                             the mixing matrix and component table.
                                                                             Maps are in z-statistics.
"PCA metrics tsv": desc-PCA_metrics.tsv                                      TEDPCA component table. A BIDS Derivatives-compatible
                                                                             TSV file with summary metrics and inclusion/exclusion
                                                                             information for each component from the PCA
                                                                             decomposition.
"PCA metrics json": desc-PCA_metrics.json                                    Metadata about the metrics in ``desc-PCA_metrics.tsv``.
"PCA cross component metrics json": desc-PCACrossComponent_metrics.json      Measures calculated across PCA compononents including
                                                                             values for the full cost function curves for all
                                                                             AIC, KIC, and MDL cost functions and the number of
                                                                             components and variance explained for multiple options
                                                                             Figures for the cost functions and variance explained
                                                                             are also in
                                                                             ``./figures//pca_[criteria|variance_explained.png]``
"ICA mixing tsv": desc-ICA_mixing.tsv                                        Mixing matrix (component time series) from ICA
                                                                             decomposition in a tab-delimited file. Each column is
                                                                             a different component, and the column name is the
                                                                             component number.
"ICA components img": desc-ICA_components.nii.gz                             Full ICA coefficient feature set.
"z-scored ICA components img": desc-ICA_stat-z_components.nii.gz             Z-statistic component weight maps from ICA
                                                                             decomposition.
                                                                             Values are z-transformed standardized regression
                                                                             coefficients. Each map corresponds to the same
                                                                             component index in the mixing matrix and component table.
"ICA decomposition json": desc-ICA_decomposition.json                        Metadata for the ICA decomposition.
"ICA metrics tsv": desc-tedana_metrics.tsv                                   TEDICA component table. A BIDS Derivatives-compatible
                                                                             TSV file with summary metrics and inclusion/exclusion
                                                                             information for each component from the ICA
                                                                             decomposition.
"ICA metrics json": desc-tedana_metrics.json                                 Metadata about the metrics in
                                                                             ``desc-tedana_metrics.tsv``.
"ICA cross component metrics json": desc-ICACrossComponent_metrics.json      Metric names and values that are each a single number
                                                                             calculated across components. For example, kappa and
                                                                             rho elbows.
"ICA decision tree json": desc-ICA_decision_tree                             A copy of the inputted decision tree specification with
                                                                             an added "output" field for each node. The output field
                                                                             contains information about what happened during
                                                                             execution.
"ICA status table tsv": desc-ICA_status_table.tsv                            A table where each column lists the classification
                                                                             status of each component after each node was run.
                                                                             Columns are only added for runs where component
                                                                             statuses can change.
"ICA accepted components img": desc-ICAAccepted_components.nii.gz            High-kappa ICA coefficient feature set
"z-scored ICA accepted components img": desc-ICAAcceptedZ_components.nii.gz  Z-normalized spatial component maps
report.txt                                                                   A summary report for the workflow with relevant
                                                                             citations.
"low kappa ts img": desc-optcomRejected_bold.nii.gz                          Combined time series from rejected components.
"high kappa ts img": desc-optcomAccepted_bold.nii.gz                         High-kappa time series. This dataset does not
                                                                             include thermal noise or low variance components.
                                                                             Not the recommended dataset for analysis.
references.bib                                                               The BibTeX entries for references cited in
                                                                             report.txt.

===========================================================================  =====================================================

If ``verbose`` is set to True:

=============================================================================================  =====================================================
Key: Filename                                                                                  Content
=============================================================================================  =====================================================
"limited t2star img": desc-limited_T2starmap.nii.gz                                            Limited T2* map/time series.
                                                                                               Values are in seconds.
                                                                                               Unlike the full T2* maps, if only one 1 echo contains
                                                                                               good data the limited map will have NaN
"limited s0 img": desc-limited_S0map.nii.gz                                                    Limited S0 map/time series.
                                                                                               Unlike the full S0 maps, if only one 1 echo contains
                                                                                               good data the limited map will have NaN
"whitened img": desc-optcom_whitened_bold                                                      The optimally combined data after whitening
"echo weight [PCA|ICA] maps split img": echo-[echo]_desc-[PCA|ICA]_components.nii.gz           Echo-wise PCA/ICA component weight maps.
"echo T2 [PCA|ICA] split img": echo-[echo]_desc-[PCA|ICA]T2ModelPredictions_components.nii.gz  Component- and voxel-wise R2-model predictions,
                                                                                               separated by echo.
"echo S0 [PCA|ICA] split img": echo-[echo]_desc-[PCA|ICA]S0ModelPredictions_components.nii.gz  Component- and voxel-wise S0-model predictions,
                                                                                               separated by echo.
"[PCA|ICA] component weights img": desc-[PCA|ICA]AveragingWeights_components.nii.gz            Component-wise averaging weights for metric
                                                                                               calculation.
"[PCA|ICA] component F-S0 img": desc-[PCA|ICA]S0_stat-F_statmap.nii.gz                         F-statistic map for each component, for the S0 model.
"[PCA|ICA] component F-T2 img": desc-[PCA|ICA]T2_stat-F_statmap.nii.gz                         F-statistic map for each component, for the T2 model.
"PCA reduced img": desc-optcomPCAReduced_bold.nii.gz                                           Optimally combined data after dimensionality
                                                                                               reduction with PCA. This is the input to the ICA.
"high kappa ts split img": echo-[echo]_desc-Accepted_bold.nii.gz                               High-Kappa time series for echo number ``echo``
"low kappa ts split img": echo-[echo]_desc-Rejected_bold.nii.gz                                Low-Kappa time series for echo number ``echo``
"denoised ts split img": echo-[echo]_desc-Denoised_bold.nii.gz                                 Denoised time series for echo number ``echo``
=============================================================================================  =====================================================

If ``tedort`` is True

========================================================  =====================================================
Key: Filename                                             Content
========================================================  =====================================================
"ICA orthogonalized mixing tsv": desc-ICAOrth_mixing.tsv  Mixing matrix with rejected components orthogonalized
                                                          from accepted components
========================================================  =====================================================

If ``gscontrol`` includes 'gsr':

=================================================================  =====================================================
Key: Filename                                                      Content
=================================================================  =====================================================
"gs img": desc-globalSignal_map.nii.gz                             Spatial global signal
"global signal time series tsv": desc-globalSignal_timeseries.tsv  Time series of global signal from optimally combined
                                                                   data.
"has gs combined img": desc-optcomWithGlobalSignal_bold.nii.gz     Optimally combined time series with global signal
                                                                   retained.
"removed gs combined img": desc-optcomNoGlobalSignal_bold.nii.gz   Optimally combined time series with global signal
                                                                   removed.
=================================================================  =====================================================

If ``gscontrol`` includes 'mir' (Minimal intensity regression, which may help remove some T1 noise and
was an option in the MEICA v2.5 code, but never fully explained or evaluted in a publication):

=======================================================================================  =====================================================
Key: Filename                                                                            Content
=======================================================================================  =====================================================
"t1 like img": desc-T1likeEffect_min.nii.gz                                              T1-like effect
"mir denoised img": desc-optcomMIRDenoised_bold.nii.gz                                   Denoised time series after MIR
"ICA MIR mixing tsv": desc-ICAMIRDenoised_mixing.tsv                                     ICA mixing matrix after MIR
"ICA accepted mir component weights img": desc-ICAAcceptedMIRDenoised_components.nii.gz  high-kappa components after MIR
"ICA accepted mir denoised img": desc-optcomAcceptedMIRDenoised_bold.nii.gz              high-kappa time series after MIR
=======================================================================================  =====================================================
