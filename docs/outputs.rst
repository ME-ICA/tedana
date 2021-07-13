.. _outputs:

Outputs of tedana
=================

tedana derivatives
------------------

================================================    =====================================================
Filename                                            Content
================================================    =====================================================
dataset_description.json                            Top-level metadata for the workflow.
T2starmap.nii.gz                                    Full estimated T2* 3D map.
                                                    Values are in seconds.
                                                    The difference between the limited and full maps
                                                    is that, for voxels affected by dropout where
                                                    only one echo contains good data, the full map
                                                    uses the single echo's value while the limited
                                                    map has a NaN.
S0map.nii.gz                                        Full S0 3D map.
                                                    The difference between the limited and full maps
                                                    is that, for voxels affected by dropout where
                                                    only one echo contains good data, the full map
                                                    uses the single echo's value while the limited
                                                    map has a NaN.
desc-optcom_bold.nii.gz                             Optimally combined time series.
desc-optcomDenoised_bold.nii.gz                     Denoised optimally combined time series. Recommended
                                                    dataset for analysis.
desc-optcomRejected_bold.nii.gz                     Combined time series from rejected components.
desc-optcomAccepted_bold.nii.gz                     High-kappa time series. This dataset does not
                                                    include thermal noise or low variance components.
                                                    Not the recommended dataset for analysis.
desc-adaptiveGoodSignal_mask.nii.gz                 Integer-valued mask used in the workflow, where
                                                    each voxel's value corresponds to the number of good
                                                    echoes to be used for T2\*/S0 estimation.
desc-PCA_mixing.tsv                                 Mixing matrix (component time series) from PCA
                                                    decomposition in a tab-delimited file. Each column is
                                                    a different component, and the column name is the
                                                    component number.
desc-PCA_decomposition.json                         Metadata for the PCA decomposition.
desc-PCA_stat-z_components.nii.gz                   Component weight maps from PCA decomposition.
                                                    Each map corresponds to the same component index in
                                                    the mixing matrix and component table.
                                                    Maps are in z-statistics.
desc-PCA_metrics.tsv                                TEDPCA component table. A BIDS Derivatives-compatible
                                                    TSV file with summary metrics and inclusion/exclusion
                                                    information for each component from the PCA
                                                    decomposition.
desc-PCA_metrics.json                               Metadata about the metrics in ``desc-PCA_metrics.tsv``.
desc-ICA_mixing.tsv                                 Mixing matrix (component time series) from ICA
                                                    decomposition in a tab-delimited file. Each column is
                                                    a different component, and the column name is the
                                                    component number.
desc-ICA_components.nii.gz                          Full ICA coefficient feature set.
desc-ICA_stat-z_components.nii.gz                   Z-statistic component weight maps from ICA
                                                    decomposition.
                                                    Values are z-transformed standardized regression
                                                    coefficients. Each map corresponds to the same
                                                    component index in the mixing matrix and component table.
desc-ICA_decomposition.json                         Metadata for the ICA decomposition.
desc-tedana_metrics.tsv                             TEDICA component table. A BIDS Derivatives-compatible
                                                    TSV file with summary metrics and inclusion/exclusion
                                                    information for each component from the ICA
                                                    decomposition.
desc-tedana_metrics.json                            Metadata about the metrics in
                                                    ``desc-tedana_metrics.tsv``.
desc-ICAAccepted_components.nii.gz                  High-kappa ICA coefficient feature set
desc-ICAAcceptedZ_components.nii.gz                 Z-normalized spatial component maps
report.txt                                          A summary report for the workflow with relevant
                                                    citations.
tedana_report.html                                  The interactive HTML report.
================================================    =====================================================

If ``verbose`` is set to True:

==============================================================  =====================================================
Filename                                                        Content
==============================================================  =====================================================
desc-limited_T2starmap.nii.gz                                   Limited T2* map/time series.
                                                                Values are in seconds.
                                                                The difference between the limited and full maps is
                                                                that, for voxels affected by dropout where only one
                                                                echo contains good data, the full map uses the
                                                                single echo's value while the limited map has a NaN.
desc-limited_S0map.nii.gz                                       Limited S0 map/time series.
echo-[echo]_desc-[PCA|ICA]_components.nii.gz                    Echo-wise PCA/ICA component weight maps.
echo-[echo]_desc-[PCA|ICA]R2ModelPredictions_components.nii.gz  Component- and voxel-wise R2-model predictions,
                                                                separated by echo.
echo-[echo]_desc-[PCA|ICA]S0ModelPredictions_components.nii.gz  Component- and voxel-wise S0-model predictions,
                                                                separated by echo.
desc-[PCA|ICA]AveragingWeights_components.nii.gz                Component-wise averaging weights for metric
                                                                calculation.
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

Component tables
----------------
TEDPCA and TEDICA use component tables to track relevant metrics, component
classifications, and rationales behind classifications.
The component tables are stored as json files for BIDS-compatibility.
This format is not very conducive to manual review, which is why we have
:py:func:`tedana.io.load_comptable` to load the json file into a pandas
DataFrame.

In order to make sense of the rationale codes in the component tables,
consult the tables below.
TEDPCA rationale codes start with a "P", while TEDICA codes start with an "I".

===============    =============================================================
Classification     Description
===============    =============================================================
accepted           BOLD-like components included in denoised and high-Kappa data
rejected           Non-BOLD components excluded from denoised and high-Kappa data
ignored            Low-variance components included in denoised, but excluded
                   from high-Kappa data
===============    =============================================================

TEDPCA codes
````````````
=====  ===============  ========================================================
Code   Classification   Description
=====  ===============  ========================================================
P001   rejected         Low Rho, Kappa, and variance explained
P002   rejected         Low variance explained
P003   rejected         Kappa equals fmax
P004   rejected         Rho equals fmax
P005   rejected         Cumulative variance explained above 95% (only in
                        stabilized PCA decision tree)
P006   rejected         Kappa below fmin (only in stabilized PCA decision tree)
P007   rejected         Rho below fmin (only in stabilized PCA decision tree)
=====  ===============  ========================================================

TEDICA codes
````````````
=====  =================  ========================================================
Code   Classification     Description
=====  =================  ========================================================
I001   rejected|accepted  Manual classification
I002   rejected           Rho greater than Kappa
I003   rejected           More significant voxels in S0 model than R2 model
I004   rejected           S0 Dice is higher than R2 Dice and high variance
                          explained
I005   rejected           Noise F-value is higher than signal F-value and high
                          variance explained
I006   ignored            No good components found
I007   rejected           Mid-Kappa component
I008   ignored            Low variance explained
I009   rejected           Mid-Kappa artifact type A
I010   rejected           Mid-Kappa artifact type B
I011   ignored            ign_add0
I012   ignored            ign_add1
=====  =================  ========================================================

Citable workflow summaries
--------------------------

``tedana`` generates a report for the workflow, customized based on the parameters used and including relevant citations.
The report is saved in a plain-text file, report.txt, in the output directory.

An example report

  TE-dependence analysis was performed on input data. An initial mask was generated from the first echo using nilearn's compute_epi_mask function. An adaptive mask was then generated, in which each voxel's value reflects the number of echoes with 'good' data. A monoexponential model was fit to the data at each voxel using nonlinear model fitting in order to estimate T2* and S0 maps, using T2*/S0 estimates from a log-linear fit as initial values. For each voxel, the value from the adaptive mask was used to determine which echoes would be used to estimate T2* and S0. In cases of model fit failure, T2*/S0 estimates from the log-linear fit were retained instead. Multi-echo data were then optimally combined using the T2* combination method (Posse et al., 1999). Principal component analysis in which the number of components was determined based on a variance explained threshold was applied to the optimally combined data for dimensionality reduction. A series of TE-dependence metrics were calculated for each component, including Kappa, Rho, and variance explained. Independent component analysis was then used to decompose the dimensionally reduced dataset. A series of TE-dependence metrics were calculated for each component, including Kappa, Rho, and variance explained. Next, component selection was performed to identify BOLD (TE-dependent), non-BOLD (TE-independent), and uncertain (low-variance) components using the Kundu decision tree (v2.5; Kundu et al., 2013). Rejected components' time series were then orthogonalized with respect to accepted components' time series.

  This workflow used numpy (Van Der Walt, Colbert, & Varoquaux, 2011), scipy (Jones et al., 2001), pandas (McKinney, 2010), scikit-learn (Pedregosa et al., 2011), nilearn, and nibabel (Brett et al., 2019).

  This workflow also used the Dice similarity index (Dice, 1945; Sørensen, 1948).

  References

  Brett, M., Markiewicz, C. J., Hanke, M., Côté, M.-A., Cipollini, B., McCarthy, P., … freec84. (2019, May 28). nipy/nibabel. Zenodo. http://doi.org/10.5281/zenodo.3233118

  Dice, L. R. (1945). Measures of the amount of ecologic association between species. Ecology, 26(3), 297-302.

  Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, http://www.scipy.org/

  Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., Vértes, P. E., Inati, S. J., ... & Bullmore, E. T. (2013). Integrated strategy for improving functional connectivity mapping using multiecho fMRI. Proceedings of the National Academy of Sciences, 110(40), 16187-16192.

  McKinney, W. (2010, June). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51-56).

  Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

  Posse, S., Wiese, S., Gembris, D., Mathiak, K., Kessler, C., Grosse‐Ruyken, M. L., ... & Kiselev, V. G. (1999). Enhancement of BOLD‐contrast sensitivity by single‐shot multi‐echo functional MR imaging. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 42(1), 87-97.

  Sørensen, T. J. (1948). A method of establishing groups of equal amplitude in plant sociology based on similarity of species content and its application to analyses of the vegetation on Danish commons. I kommission hos E. Munksgaard.

  Van Der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). The NumPy array: a structure for efficient numerical computation. Computing in Science & Engineering, 13(2), 22.
