Introduction
============

``tedana`` works in the following steps:

#. Computes PCA and ICA in conjunction with TE-dependence analysis

Derivatives
-----------

* ``medn``
    'Denoised' BOLD time series after: basic preprocessing,
    T2* weighted averaging of echoes (i.e. 'optimal combination'),
    ICA denoising.
    Use this dataset for task analysis and resting state time series correlation analysis.
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
    Table of component Kappa, Rho, and variance explained values, plus listing of component classifications.
