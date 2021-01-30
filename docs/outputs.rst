.. _outputs:

Outputs of tedana
=================

tedana derivatives
------------------

======================    =====================================================
Filename                  Content
======================    =====================================================
t2sv.nii.gz               Limited estimated T2* 3D map.
                          Values are in seconds.
                          The difference between the limited and full maps
                          is that, for voxels affected by dropout where
                          only one echo contains good data, the full map
                          uses the single echo's value while the limited
                          map has a NaN.
s0v.nii.gz                Limited S0 3D map.
                          The difference between the limited and full maps
                          is that, for voxels affected by dropout where
                          only one echo contains good data, the full map
                          uses the single echo's value while the limited
                          map has a NaN.
ts_OC.nii.gz              Optimally combined time series.
dn_ts_OC.nii.gz           Denoised optimally combined time series. Recommended
                          dataset for analysis.
lowk_ts_OC.nii.gz         Combined time series from rejected components.
midk_ts_OC.nii.gz         Combined time series from "mid-k" rejected components.
hik_ts_OC.nii.gz          High-kappa time series. This dataset does not
                          include thermal noise or low variance components.
                          Not the recommended dataset for analysis.
adaptive_mask.nii.gz      Integer-valued mask used in the workflow, where
                          each voxel's value corresponds to the number of good
                          echoes to be used for T2*/S0 estimation.
pca_decomposition.json    TEDPCA component table. A BIDS Derivatives-compatible
                          json file with summary metrics and inclusion/exclusion
                          information for each component from the PCA
                          decomposition. To view, you may want to use
                          ``io.load_comptable``, which returns a pandas
                          DataFrame from the json file.
pca_mixing.tsv            Mixing matrix (component time series) from PCA
                          decomposition in a tab-delimited file. Each column is
                          a different component, and the column name is the
                          component number.
pca_components.nii.gz     Component weight maps from PCA decomposition.
                          Each map corresponds to the same component index in
                          the mixing matrix and component table.
ica_decomposition.json    TEDICA component table. A BIDS Derivatives-compatible
                          json file with summary metrics and inclusion/exclusion
                          information for each component from the ICA
                          decomposition. To view, you may want to use
                          ``io.load_comptable``, which returns a pandas
                          DataFrame from the json file.
ica_mixing.tsv            Mixing matrix (component time series) from ICA
                          decomposition in a tab-delimited file. Each column is
                          a different component, and the column name is the
                          component number.
ica_components.nii.gz     Component weight maps from ICA decomposition.
                          Values are z-transformed standardized regression
                          coefficients. Each map corresponds to the same
                          component index in the mixing matrix and component table.
                          Should be the same as "feats_OC2.nii.gz".
betas_OC.nii.gz           Full ICA coefficient feature set.
betas_hik_OC.nii.gz       High-kappa ICA coefficient feature set
feats_OC2.nii.gz          Z-normalized spatial component maps
report.txt                A summary report for the workflow with relevant
                          citations.
======================    =====================================================

If ``verbose`` is set to True:

======================    =====================================================
Filename                  Content
======================    =====================================================
t2svG.nii.gz              Full T2* map/time series.
                          Values are in seconds.
                          The difference between the limited and full maps is
                          that, for voxels affected by dropout where only one
                          echo contains good data, the full map uses the
                          single echo's value while the limited map has a NaN.
                          Only used for optimal combination.
s0vG.nii.gz               Full S0 map/time series. Only used for optimal
                          combination.
hik_ts_e[echo].nii.gz     High-Kappa time series for echo number ``echo``
midk_ts_e[echo].nii.gz    Mid-Kappa time series for echo number ``echo``
lowk_ts_e[echo].nii.gz    Low-Kappa time series for echo number ``echo``
dn_ts_e[echo].nii.gz      Denoised time series for echo number ``echo``
======================    =====================================================

If ``gscontrol`` includes 'gsr':

======================    =====================================================
Filename                  Content
======================    =====================================================
T1gs.nii.gz               Spatial global signal
glsig.1D                  Time series of global signal from optimally combined
                          data.
tsoc_orig.nii.gz          Optimally combined time series with global signal
                          retained.
tsoc_nogs.nii.gz          Optimally combined time series with global signal
                          removed.
======================    =====================================================

If ``gscontrol`` includes 't1c':

=======================    =====================================================
Filename                   Content
=======================    =====================================================
sphis_hik.nii.gz           T1-like effect
hik_ts_OC_T1c.nii.gz       T1 corrected high-kappa time series by regression
dn_ts_OC_T1c.nii.gz        T1 corrected denoised time series
betas_hik_OC_T1c.nii.gz    T1-GS corrected high-kappa components
meica_mix_T1c.1D           T1-GS corrected mixing matrix
=======================    =====================================================

Component tables
----------------
TEDPCA and TEDICA use tab-delimited tables to track relevant metrics, component
classifications, and rationales behind classifications.
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

  TE-dependence analysis was performed on input data. An initial mask was generated from the first echo using nilearn's compute_epi_mask function. An adaptive mask was then generated, in which each voxel's value reflects the number of echoes with 'good' data. A monoexponential model was fit to the data at each voxel using log-linear regression in order to estimate T2* and S0 maps. For each voxel, the value from the adaptive mask was used to determine which echoes would be used to estimate T2* and S0. Multi-echo data were then optimally combined using the 't2s' (Posse et al., 1999) combination method. Global signal regression was applied to the multi-echo and optimally combined datasets. Principal component analysis followed by the Kundu component selection decision tree (Kundu et al., 2013) was applied to the optimally combined data for dimensionality reduction. Independent component analysis was then used to decompose the dimensionally reduced dataset. A series of TE-dependence metrics were calculated for each ICA component, including Kappa, Rho, and variance explained. Next, component selection was performed to identify BOLD (TE-dependent), non-BOLD (TE-independent), and uncertain (low-variance) components using the Kundu decision tree (v2.5; Kundu et al., 2013). T1c global signal regression was then applied to the data in order to remove spatially diffuse noise.

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
