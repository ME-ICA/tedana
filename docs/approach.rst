Processing pipeline details
===========================

``tedana`` works by decomposing multi-echo BOLD data via PCA and ICA.
These components are then analyzed to determine whether they are TE-dependent
or -independent. TE-dependent components are classified as BOLD, while
TE-independent components are classified as non-BOLD, and are discarded as part
of data cleaning.

In ``tedana``, we take the time series from all the collected TEs, combine them,
and decompose the resulting data into components that can be classified as BOLD
or non-BOLD. This is performed in a series of steps, including:

* Principal components analysis
* Independent components analysis
* Component classification

.. image:: /_static/tedana-workflow.png
  :align: center

Multi-echo data
```````````````

Here are the echo-specific time series for a single voxel in an example
resting-state scan with 5 echoes.

.. image:: /_static/01_echo_timeseries.png
  :align: center

The values across volumes for this voxel scale with echo time in a predictable
manner.

.. image:: /_static/02_echo_value_distributions.png
  :width: 400 px
  :align: center

Adaptive mask generation
````````````````````````
Longer echo times are more susceptible to signal dropout, which means that
certain brain regions (e.g., orbitofrontal cortex, temporal poles) will only
have good signal for some echoes. In order to avoid using bad signal from
affected echoes in calculating :math:`T_{2}^*` and :math:`S_{0}` for a given voxel,
``tedana`` generates an adaptive mask, where the value for each voxel is the
number of echoes with "good" signal. When :math:`T_{2}^*` and :math:`S_{0}` are
calculated below, each voxel's values are only calculated from the first :math:`n`
echoes, where :math:`n` is the value for that voxel in the adaptive mask.

.. note::
    ``tedana`` allows users to provide their own mask.
    The adaptive mask will be computed on this explicit mask, and may reduce
    it further based on the data.
    If a mask is not provided, ``tedana`` runs `nilearn.masking.compute_epi_mask`_
    on the first echo's data to derive a mask prior to adaptive masking.
    The workflow does this because the adaptive mask generation function
    sometimes identifies almost the entire bounding box as "brain", and
    ``compute_epi_mask`` restricts analysis to a more reasonable area.

.. image:: /_static/03_adaptive_mask.png
  :width: 600 px
  :align: center

Monoexponential decay model fit
```````````````````````````````
The next step is to fit a monoexponential decay model to the data in order to
estimate voxel-wise :math:`T_{2}^*` and :math:`S_0`.

In order to make it easier to fit the decay model to the data, ``tedana``
transforms the data. The BOLD data are transformed as :math:`log(|S|+1)`, where
:math:`S` is the BOLD signal. The echo times are also multiplied by -1.

.. image:: /_static/04_echo_log_value_distributions.png
  :width: 400 px
  :align: center

A simple line can then be fit to the transformed data with linear regression.
For the sake of this introduction, we can assume that the example voxel has
good signal in all five echoes (i.e., the adaptive mask has a value of 5 at
this voxel), so the line is fit to all available data.

.. note::
    ``tedana`` actually performs and uses two sets of :math:`T_{2}^*`/:math:`S_0` model fits.
    In one case, ``tedana`` estimates :math:`T_{2}^*` and :math:`S_0` for voxels with good signal in at
    least two echoes. The resulting "limited" :math:`T_{2}^*` and :math:`S_0` maps are used throughout
    most of the pipeline. In the other case, ``tedana`` estimates :math:`T_{2}^*` and :math:`S_0` for voxels
    with good data in only one echo as well, but uses the first two echoes for
    those voxels. The resulting "full" :math:`T_{2}^*` and :math:`S_0` maps are used to generate the
    optimally combined data.

.. image:: /_static/05_loglinear_regression.png
  :width: 400 px
  :align: center

The values of interest for the decay model, :math:`S_0` and :math:`T_{2}^*`,
are then simple transformations of the line's intercept (:math:`B_{0}`) and
slope (:math:`B_{1}`), respectively:

.. math:: S_{0} = e^{B_{0}}

.. math:: T_{2}^{*} = \frac{1}{B_{1}}

The resulting values can be used to show the fitted monoexponential decay model
on the original data.

.. image:: /_static/06_monoexponential_decay_model.png
  :width: 400 px
  :align: center

We can also see where :math:`T_{2}^*` lands on this curve.

.. image:: /_static/07_monoexponential_decay_model_with_t2.png
  :width: 400 px
  :align: center

.. _optimal combination:

Optimal combination
```````````````````
Using the :math:`T_{2}^*` estimates, ``tedana`` combines signal across echoes using a
weighted average. The echoes are weighted according to the formula

.. math:: w_{TE} = TE * e^{\frac{-TE}{T_{2}^*}}

The weights are then normalized across echoes. For the example voxel, the
resulting weights are:

.. image:: /_static/08_optimal_combination_echo_weights.png
  :width: 400 px
  :align: center

The distribution of values for the optimally combined data lands somewhere
between the distributions for other echoes.

.. image:: /_static/09_optimal_combination_value_distributions.png
  :width: 400 px
  :align: center

The time series for the optimally combined data also looks like a combination
of the other echoes (which it is).

.. image:: /_static/10_optimal_combination_timeseries.png
  :align: center

.. note::
    An alternative method for optimal combination that
    does not use :math:`T_{2}^*`, is the parallel-acquired inhomogeneity
    desensitized (PAID) ME-fMRI combination method (`Poser et al., 2006`_).
    This method specifically assumes that noise in the acquired echoes is "isotopic and 
    homogeneous throughout the image," meaning it should be used on smoothed data. 
    As we do not recommend performing tedana denoising  on smoothed data, 
    we discourage using PAID within the tedana workflow.
    We do, however, make it accessible as an alternative combination method 
    in the t2smap workflow.

TEDPCA
``````
The next step is to identify and temporarily remove Gaussian (thermal) noise
with TE-dependent principal components analysis (PCA). TEDPCA applies PCA to
the optimally combined data in order to decompose it into component maps and
time series. Here we can see time series for some example components (we don't
really care about the maps):

.. image:: /_static/11_pca_component_timeseries.png

These components are subjected to component selection, the
specifics of which vary according to algorithm.

In the simplest approach, ``tedana`` uses Minka’s MLE to estimate the
dimensionality of the data, which disregards low-variance components.

A more complicated approach involves applying a decision tree to identify and
discard PCA components which, in addition to not explaining much variance,
are also not significantly TE-dependent (i.e., have low Kappa) or
TE-independent (i.e., have low Rho).

After component selection is performed, the retained components and their
associated betas are used to reconstruct the optimally combined data, resulting
in a dimensionally reduced (i.e., whitened) version of the dataset.

.. image:: /_static/12_pca_whitened_data.png

TEDICA
``````
Next, ``tedana`` applies TE-dependent independent components analysis (ICA) in
order to identify and remove TE-independent (i.e., non-BOLD noise) components.
The dimensionally reduced optimally combined data are first subjected to ICA in
order to fit a mixing matrix to the whitened data.

.. image:: /_static/13_ica_component_timeseries.png

Linear regression is used to fit the component time series to each voxel in each
echo from the original, echo-specific data. This way, the thermal noise is
retained in the data, but is ignored by the TEDICA process. This results in
echo- and voxel-specific betas for each of the components.

TE-dependence (:math:`R_2`) and TE-independence (:math:`S_0`) models can then
be fit to these betas. These models allow calculation of F-statistics for the
:math:`R_2` and :math:`S_0` models (referred to as :math:`\kappa` and
:math:`\rho`, respectively).

.. image:: /_static/14_te_dependence_models_component_0.png
  :width: 400 px
  :align: center

.. image:: /_static/14_te_dependence_models_component_1.png
  :width: 400 px
  :align: center

.. image:: /_static/14_te_dependence_models_component_2.png
  :width: 400 px
  :align: center

A decision tree is applied to :math:`\kappa`, :math:`\rho`, and other metrics in order to
classify ICA components as TE-dependent (BOLD signal), TE-independent
(non-BOLD noise), or neither (to be ignored). The actual decision tree is
dependent on the component selection algorithm employed. ``tedana`` includes
two options: `kundu_v2_5` (which uses hardcoded thresholds applied to each of
the metrics) and `kundu_v3_2` (which trains a classifier to select components).

.. image:: /_static/15_denoised_data_timeseries.png

Removal of spatially diffuse noise (optional)
`````````````````````````````````````````````
Due to the constraints of ICA, MEICA is able to identify and remove spatially
localized noise components, but it cannot identify components that are spread
out throughout the whole brain. See `Power et al. (2018)`_ for more information
about this issue.
One of several post-processing strategies may be applied to the ME-DN or ME-HK
datasets in order to remove spatially diffuse (ostensibly respiration-related)
noise. Methods which have been employed in the past include global signal
regression (GSR), T1c-GSR, anatomical CompCor, Go Decomposition (GODEC), and
robust PCA.

.. image:: /_static/16_t1c_denoised_data_timeseries.png

.. _nilearn.masking.compute_epi_mask: https://nilearn.github.io/modules/generated/nilearn.masking.compute_epi_mask.html
.. _Power et al. (2018): http://www.pnas.org/content/early/2018/02/07/1720985115.short
.. _Poser et al., 2006: https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.20900
