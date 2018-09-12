Processing pipeline details
===========================

``tedana`` works by decomposing multi-echo BOLD data via PCA and ICA.
These components are then analyzed to determine whether they are TE-dependent
or -independent. TE-dependent components are classified as BOLD, while
TE-independent components are classified as non-BOLD, and are discarded as part
of data cleaning.

In ``tedana``, we take the time series from all the collected TEs, combine them,
and decompose the resulting data into components that can be classified as BOLD
or non-BOLD. This is performed in a series of steps including:

* Principal components analysis
* Independent components analysis
* Component classification

.. image:: /_static/tedana-workflow.png

Fit monoexponential decay model
```````````````````````````````
The first step is to fit a monoexponential decay model to the data in order to
estimate voxel-wise T2* and S0.

Optimal combination
```````````````````
Using the T2* estimates, ``tedana`` combines signal across echoes using a
weighted average.

TEDPCA
``````
The next step is to identify and temporarily remove Gaussian (thermal) noise
with TE-dependent principal components analysis (PCA). TEDPCA applies PCA to
the optimally combined data in order to decompose it into component maps and
timeseries. These components are subjected to component selection, in which
components that are not significantly TE-dependent (low Kappa) or
TE-independent (low Rho), or which do not explain much variance, are removed
from the data, producing a dimensionally reduced dataset.

TEDICA
``````
Next, ``tedana`` applies TE-dependent independent components analysis (ICA) in
order to identify and remove TE-independent (i.e., non-BOLD noise) components.
The dimensionally reduced optimally combined data are first subjected to ICA in
order to fit a mixing matrix to the data *without* thermal noise.

The mixing matrix is then applied to the full optimally combined data in order
to compute component metrics (e.g., Kappa, Rho, variance explained). This
mixing matrix corresponds to components which do not include thermal noise, but
is fitted to data which does. This way, the thermal noise is retained in the
data, but is ignored by the TEDICA process. The component metrics are next
subjected to component selection in order to identify which components are
BOLD-related, which are not, and which cannot be classified as one or the other.

The BOLD components are retained in the ``ME-HK`` (high Kappa) output dataset, while the
BOLD components, ignored components, and model residuals (i.e., thermal noise)
are combined to form the ``ME-DN`` (denoised) dataset.

Removal of spatially diffuse noise
``````````````````````````````````
According to `Power et al. (2018)`_, ``tedana`` is able to remove spatially
localized motion-related artifacts from fMRI data, but cannot eliminate
widespread, BOLD-based motion noise linked to respiration. To that end,
``tedana`` includes optional methods for removing spatially diffuse noise,
including Go Decomposition (GODEC) and T1-global signal correction (T1c-GSR).

.. _Power et al. (2018): http://www.pnas.org/content/early/2018/02/07/1720985115.short
