#############################
Output file name descriptions
#############################

tedana allows for multiple file naming conventions. The key labels and naming options for
each convention that can be set using the `--convention` option are in `outputs.json`_.
The output of `tedana` also includes a file called `registery.json` or 
`desc-tedana_registry.json` that includes the keys and the matching file names for the
output. The table below lists both these keys and the default "BIDS Derivatives"
file names.

.. _outputs.json: https://github.com/ME-ICA/tedana/blob/main/tedana/resources/config/outputs.json

===========================================================================  =====================================================
Key: Filename                                                                Content
===========================================================================  =====================================================
"data description json": dataset_description.json                            Top-level metadata for the workflow.
"t2star img": T2starmap.nii.gz                                               Full estimated T2* 3D map.
                                                                             Values are in seconds.
                                                                             The difference between the limited and full maps
                                                                             is that, for voxels affected by dropout where
                                                                             only one echo contains good data, the full map uses
                                                                             the T2* estimate from the first two echoes, while the
                                                                             limited map has a NaN.
"s0 img": S0map.nii.gz                                                       Full S0 3D map.
                                                                             The difference between the limited and full maps
                                                                             is that, for voxels affected by dropout where
                                                                             only one echo contains good data, the full map uses
                                                                             the S0 estimate from the first two echoes, while the
                                                                             limited map has a NaN.
"combined img": desc-optcom_bold.nii.gz                                      Optimally combined time series.
"denoised ts img": desc-optcomDenoised_bold.nii.gz                           Denoised optimally combined time series. Recommended
                                                                             dataset for analysis.
"low kappa ts img": desc-optcomRejected_bold.nii.gz                          Combined time series from rejected components.
"high kappa ts img": desc-optcomAccepted_bold.nii.gz                         High-kappa time series. This dataset does not
                                                                             include thermal noise or low variance components.
                                                                             Not the recommended dataset for analysis.
"adaptive mask img": desc-adaptiveGoodSignal_mask.nii.gz                     Integer-valued mask used in the workflow, where
                                                                             each voxel's value corresponds to the number of good
                                                                             echoes to be used for T2\*/S0 estimation.
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
"ICA accepted components img": desc-ICAAccepted_components.nii.gz            High-kappa ICA coefficient feature set
"z-scored ICA accepted components img": desc-ICAAcceptedZ_components.nii.gz  Z-normalized spatial component maps
report.txt                                                                   A summary report for the workflow with relevant
                                                                             citations.
references.bib                                                               The BibTeX entries for references cited in
                                                                             report.txt.
tedana_report.html                                                           The interactive HTML report.
===========================================================================  =====================================================

If ``verbose`` is set to True:

==============================================================  =====================================================
Filename                                                        Content
==============================================================  =====================================================
desc-limited_T2starmap.nii.gz                                   Limited T2* map/time series.
                                                                Values are in seconds.
                                                                The difference between the limited and full maps
                                                                is that, for voxels affected by dropout where
                                                                only one echo contains good data, the full map uses
                                                                the S0 estimate from the first two echoes, while the
                                                                limited map has a NaN.
desc-limited_S0map.nii.gz                                       Limited S0 map/time series.
                                                                The difference between the limited and full maps
                                                                is that, for voxels affected by dropout where
                                                                only one echo contains good data, the full map uses
                                                                the S0 estimate from the first two echoes, while the
                                                                limited map has a NaN.
echo-[echo]_desc-[PCA|ICA]_components.nii.gz                    Echo-wise PCA/ICA component weight maps.
echo-[echo]_desc-[PCA|ICA]R2ModelPredictions_components.nii.gz  Component- and voxel-wise R2-model predictions,
                                                                separated by echo.
echo-[echo]_desc-[PCA|ICA]S0ModelPredictions_components.nii.gz  Component- and voxel-wise S0-model predictions,
                                                                separated by echo.
desc-[PCA|ICA]AveragingWeights_components.nii.gz                Component-wise averaging weights for metric
                                                                calculation.
desc-[PCA|ICA]S0_stat-F_statmap.nii.gz                          F-statistic map for each component, for the S0 model.
desc-[PCA|ICA]T2_stat-F_statmap.nii.gz                          F-statistic map for each component, for the T2 model.
desc-optcomPCAReduced_bold.nii.gz                               Optimally combined data after dimensionality
                                                                reduction with PCA. This is the input to the ICA.
echo-[echo]_desc-Accepted_bold.nii.gz                           High-Kappa time series for echo number ``echo``
echo-[echo]_desc-Rejected_bold.nii.gz                           Low-Kappa time series for echo number ``echo``
echo-[echo]_desc-Denoised_bold.nii.gz                           Denoised time series for echo number ``echo``
==============================================================  =====================================================

If ``gscontrol`` includes 'gsr':

================================================    =====================================================
Filename                                            Content
================================================    =====================================================
desc-globalSignal_map.nii.gz                        Spatial global signal
desc-globalSignal_timeseries.tsv                    Time series of global signal from optimally combined
                                                    data.
desc-optcomWithGlobalSignal_bold.nii.gz             Optimally combined time series with global signal
                                                    retained.
desc-optcomNoGlobalSignal_bold.nii.gz               Optimally combined time series with global signal
                                                    removed.
================================================    =====================================================

If ``gscontrol`` includes 't1c':

================================================    =====================================================
Filename                                            Content
================================================    =====================================================
desc-T1likeEffect_min.nii.gz                        T1-like effect
desc-optcomAcceptedT1cDenoised_bold.nii.gz          T1-corrected high-kappa time series by regression
desc-optcomT1cDenoised_bold.nii.gz                  T1-corrected denoised time series
desc-TEDICAAcceptedT1cDenoised_components.nii.gz    T1-GS corrected high-kappa components
desc-TEDICAT1cDenoised_mixing.tsv                   T1-GS corrected mixing matrix
================================================    =====================================================