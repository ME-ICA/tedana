Introduction
============

``tedana`` works by decomposing multi-echo BOLD data via PCA and ICA.
These components are then analyzed to determine whether they are TE-dependent
or -independent. TE-dependent components are classified as BOLD, while
TE-independent components are classified as non-BOLD, and are discarded as part
of data cleaning.

Derivatives
-----------

* ``medn``
    'Denoised' BOLD time series after: basic preprocessing,
    T2* weighted averaging of echoes (i.e. 'optimal combination'),
    ICA denoising.
    Use this dataset for task analysis and resting state time series correlation
    analysis.
* ``tsoc``
    'Raw' BOLD time series dataset after: basic preprocessing
    and T2* weighted averaging of echoes (i.e. 'optimal combination').
    'Standard' denoising or task analyses can be assessed on this dataset
    (e.g. motion regression, physio correction, scrubbing, etc.)
    for comparison to ME-ICA denoising.
* ``*mefc``
    Component maps (in units of \delta S) of accepted BOLD ICA components.
    Use this dataset for ME-ICR seed-based connectivity analysis.
* ``mefl``
    Component maps (in units of \delta S) of ALL ICA components.
* ``ctab``
    Table of component Kappa, Rho, and variance explained values, plus listing
    of component classifications.
