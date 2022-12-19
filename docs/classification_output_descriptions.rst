##################################
Classification output descriptions
##################################

In addition to the denoised time series, tedana outputs multiple files that
can be used to subsequent analyses and to better understand one's denoising
results. `In addition to the descriptions of file names`_ this page explains
the contents of several of those files in more detail.

.. _In addition to the descriptions of file names: output_file_descriptions.html

===============    =============================================================
Classification     Description
===============    =============================================================
accepted           BOLD-like components included in denoised and high-Kappa data
rejected           Non-BOLD components excluded from denoised and high-Kappa data
ignored            Low-variance components included in denoised, but excluded
                   from high-Kappa data
===============    =============================================================


TEDPCA codes
============

In ``tedana`` PCA is used to reduce the number of dimensions (components) in the
dataset. Without this steps, the number of components would be one less than
the number of volumes, many of those components would effectively be
Gaussian noise and ICA would not reliably converge. Standard methods for data
reduction use cost functions, like AIC, MDL, and KIC to estimate the variance
that is just noise and remove the lowest variance components under that threshold.

``Tedana`` includes an addition `kundu` approach that identifies and removes
compnents that don't contain T2* or S0 signal and are more likely to be noise.
If the `--tedpca kundu` option is used, the PCA_metrics tsv file will include
an accepted vs rejected classification column and also a column of codes
documenting why a PCA component removed. These are brief explanations of those
codes.

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
============

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
