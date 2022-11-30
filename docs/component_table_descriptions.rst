#############################
Component table descriptions
#############################


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
============

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
