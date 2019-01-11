Outputs of tedana
===========================

tedana derivatives
------------------

========================================   =====================================================
Filename                                   Content
========================================   =====================================================
*_desc-limited_t2s.nii.gz                  Limited estimated T2* 3D map.
                                           The difference between the limited and full maps
                                           is that, for voxels affected by dropout where
                                           only one echo contains good data, the full map
                                           uses the single echo's value while the limited
                                           map has a NaN.
*_desc-limited_s0.nii.gz                   Limited S0 3D map.
                                           The difference between the limited and full maps
                                           is that, for voxels affected by dropout where
                                           only one echo contains good data, the full map
                                           uses the single echo's value while the limited
                                           map has a NaN.
*_desc-optcom_bold.nii.gz                  Optimally combined time series.
*_desc-optcomDenoised_bold.nii.gz          Denoised optimally combined time series.
                                           **Recommended dataset for analysis.**
*_desc-optcomRejected_bold.nii.gz          Combined time series from rejected components.
*_desc-optcomRejectedMidK_bold.nii.gz      Combined time series from "mid-k" rejected components.
*_desc-optcomAccepted_bold.nii.gz          High-kappa time series. This dataset does not
                                           include thermal noise or low variance components.
                                           Not the recommended dataset for analysis.
*_desc-TEDPCA_comptable.tsv                TEDPCA component table. A tab-delimited file with
                                           summary metrics and inclusion/exclusion information
                                           for each component from the PCA decomposition.
*_desc-TEDPCA_mixing.tsv                   Mixing matrix (component time series) from PCA
                                           decomposition.
*_desc-TEDICA_mixing.tsv                   Mixing matrix (component time series) from ICA
                                           decomposition. The only differences between this
                                           mixing matrix and the one above are that
                                           components may be sorted differently and signs of
                                           time series may be flipped.
*_desc-TEDICA_components.nii.gz            Full ICA coefficient feature set.
*_desc-TEDICAAccepted_components.nii.gz    High-kappa ICA spatial component maps
*_desc-TEDICAAcceptedZ_components.nii.gz   Z-normalized high-kappa ICA spatial component maps
*_desc-TEDICA_comptable.tsv                TEDICA component table. A tab-delimited file with
                                           summary metrics and inclusion/exclusion information
                                           for each component from the ICA decomposition.
========================================   =====================================================

If ``verbose`` is set to True:

=================================================   =====================================================
Filename                                            Content
=================================================   =====================================================
*_desc-adaptiveGoodSignal_mask.nii.gz               Adaptive mask. Each voxel has value corresponding to
                                                    number of echoes with good signal.
*_desc-ascendingEstimates_t2s.nii.gz                Voxel-wise T2* estimates using ascending numbers
                                                    of echoes, starting with 2.
*_desc-ascendingEstimates_s0.nii.gz                 Voxel-wise S0 estimates using ascending numbers
                                                    of echoes, starting with 2.
*_desc-full_t2s.nii.gz                              Full T2* map/time series. The difference between
                                                    the limited and full maps is that, for voxels
                                                    affected by dropout where only one echo contains
                                                    good data, the full map uses the single echo's
                                                    value while the limited map has a NaN. Only used
                                                    for optimal combination.
*_desc-full_s0.nii.gz                               Full S0 map/time series. Only used for optimal
                                                    combination.
*_desc-initialTEDICA_mixing.tsv                     Mixing matrix (component time series) from ICA
                                                    decomposition.
*_echo-[echo]_desc-optcomAccepted_bold.nii.gz       High-Kappa time series for echo number ``echo``
*_echo-[echo]_desc-optcomRejectedMidK_bold.nii.gz   Mid-Kappa time series for echo number ``echo``
*_echo-[echo]_desc-optcomRejected_bold.nii.gz       Low-Kappa time series for echo number ``echo``
*_echo-[echo]_desc-optcomDenoised_bold.nii.gz       Denoised time series for echo number ``echo``
=================================================   =====================================================

If ``gscontrol`` includes 'gsr':

=========================================   =====================================================
Filename                                    Content
=========================================   =====================================================
*_desc-T1gs_bold.nii.gz                     Spatial global signal
*_globalSignal_regressors.tsv               Time series of global signal from optimally combined
                                            data.
*_desc-optcomWithGlobalSignal_bold.nii.gz   Optimally combined time series with global signal
                                            retained.
*_desc-optcomNoGlobalSignal_bold.nii.gz     Optimally combined time series with global signal
                                            removed.
=========================================   =====================================================

If ``gscontrol`` includes 't1c':

==================================================   =====================================================
Filename                                             Content
==================================================   =====================================================
*_desc-optcomAccepted_min.nii.gz                     T1-like effect
*_desc-optcomAcceptedT1cDenoised_bold.nii.gz         T1 corrected high-kappa time series by regression
*_desc-optcomT1cDenoised_bold.nii.gz                 T1 corrected denoised time series
*_desc-TEDICAAcceptedT1cDenoised_components.nii.gz   T1-GS corrected high-kappa components
*_desc-TEDICAT1cDenoised_mixing.tsv                  T1-GS corrected mixing matrix
==================================================   =====================================================

Component tables
----------------
TEDPCA and TEDICA use tab-delimited tables to track relevant metrics, component
classifications, and rationales behind classifications.
TEDPCA rationale codes start with a "P", while TEDICA codes start with an "I".

===============    =============================================================
Classification     Description
===============    =============================================================
accepted           BOLD-like components retained in denoised and high-Kappa data
rejected           Non-BOLD components removed from denoised and high-Kappa data
ignored            Low-variance components ignored in denoised, but not
                   high-Kappa, data
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
=====  ===============  ========================================================
Code   Classification   Description
=====  ===============  ========================================================
I001   rejected         Manual exclusion
I002   rejected         Rho greater than Kappa or more significant voxels
                        in S0 model than R2 model
I003   rejected         S0 Dice is higher than R2 Dice and high variance
                        explained
I004   rejected         Noise F-value is higher than signal F-value and high
                        variance explained
I005   ignored          No good components found
I006   rejected         Mid-Kappa component
I007   ignored          Low variance explained
I008   rejected         Artifact candidate type A
I009   rejected         Artifact candidate type B
I010   ignored          ign_add0
I011   ignored          ign_add1
=====  ===============  ========================================================

Visual reports
--------------
We're working on it.
