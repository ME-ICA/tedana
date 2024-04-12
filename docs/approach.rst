###########################
tedana's denoising approach
###########################

``tedana`` works by decomposing multi-echo BOLD data via principal component analysis (PCA)
and independent component analysis (ICA).
The resulting components are then analyzed to determine whether they are
TE-dependent or -independent.
TE-dependent components are classified as BOLD, while TE-independent components
are classified as non-BOLD, and are discarded as part of data cleaning.

In ``tedana``, we take the time series from all the collected TEs, combine them,
and decompose the resulting data into components that can be classified as BOLD
or non-BOLD.
This is performed in a series of steps, including:

.. contents:: :local:

.. image:: /_static/tedana-workflow.png
  :align: center

We provide more detail on each step below.
The figures shown in this walkthrough are generated in the `provided notebooks <https://github.com/ME-ICA/tedana/tree/joss/docs/notebooks>`_.

***************
Multi-echo data
***************

Here are the echo-specific time series for a single voxel in an example
resting-state scan with 8 echoes.

.. image:: /_static/a01_echo_timeseries.png

The values across volumes for this voxel scale with echo time in a predictable
manner.

.. image:: /_static/a02_echo_value_distributions.png

.. note::
    In this example, the non-steady state volumes at the beginning of the run are
    excluded. Some pulse sequences save these initial volumes and some do not. If
    they are saved, then the first few volume in a run will have much larger relative
    magnitudes. These initial volumes should be removed before running ``tedana``

************************
Adaptive mask generation
************************

:func:`tedana.utils.make_adaptive_mask`

Longer echo times are more susceptible to signal dropout, which means that
certain brain regions (e.g., orbitofrontal cortex, temporal poles) will only
have good signal for some echoes.
In order to avoid using bad signal from affected echoes in calculating
:math:`T_{2}^*` and :math:`S_{0}` for a given voxel, ``tedana`` generates an
adaptive mask, where the value for each voxel is the number of echoes with
"good" signal. The voxel in the shortest echo with the 33rd percentile mean signal
across time is identified. The threshold for each echo is the signal in the same voxel
divided by 3. This is an arbitrary, but conservative threshold in that it only
excludes voxels where the signal is much lower than other measured signals in
each echo. When :math:`T_{2}^*` and :math:`S_{0}` are calculated below, each
voxel's values are only calculated from the first :math:`n` echoes, where
:math:`n` is the value for that voxel in the adaptive mask. By default, the
optimally combined and denoised time series will include voxels where there
is at least one good echo, but ICA and the fit maps require at least three
good echoes.

.. note::
    ``tedana`` allows users to provide their own mask.
    The adaptive mask will be computed on this explicit mask, and may reduce
    it further based on the data.
    If a mask is not provided, ``tedana`` runs :func:`nilearn.masking.compute_epi_mask`
    on the first echo's data to derive a mask prior to adaptive masking.
    Some brain masking is required because the percentile-based thresholding
    in the adaptive mask will be flawed if it includes all out-of-brain voxels.

.. image:: /_static/a03_adaptive_mask.png
  :width: 600 px
  :align: center


*******************************
Monoexponential decay model fit
*******************************

:func:`tedana.decay.fit_decay`

The next step is to fit a monoexponential decay model to the data in order to
estimate voxel-wise :math:`T_{2}^*` and :math:`S_0`.
:math:`S_0` corresponds to the total signal in each voxel before decay and can reflect coil sensivity.
:math:`T_{2}^*` corresponds to the rate at which a voxel decays over time, which
is related to signal dropout and BOLD sensitivity.
Estimates of the parameters are saved as **T2starmap.nii.gz** and **S0map.nii.gz**.

While :math:`T_{2}^*` and :math:`S_0` in fact fluctuate over time, estimating
them on a volume-by-volume basis with only a small number of echoes is not
feasible (i.e., the estimates would be extremely noisy).
As such, we estimate average :math:`T_{2}^*` and :math:`S_0` maps and use those
throughout the workflow.

In order to make it easier to fit the decay model to the data, ``tedana``
transforms the data by default.
The BOLD data are transformed as :math:`log(|S|+1)`, where :math:`S` is the BOLD signal.
The echo times are also multiplied by -1.

.. tip::
    It is now possible to do a nonlinear monoexponential fit to the original, untransformed
    data values by specifiying ``--fittype curvefit``.
    This method is slightly more computationally demanding but may obtain more
    accurate fits.

.. image:: /_static/a04_echo_log_value_distributions.png

A simple line can then be fit to the transformed data with linear regression.
For the sake of this introduction, we can assume that the example voxel has
good signal in all eight echoes (i.e., the adaptive mask has a value of 8 at
this voxel), so the line is fit to all available data.

.. note::
    ``tedana`` actually performs and uses two sets of :math:`T_{2}^*`/:math:`S_0` model fits.
    In one case, ``tedana`` estimates :math:`T_{2}^*` and :math:`S_0` for voxels with good signal in at
    least two echoes.
    In the other case, ``tedana`` estimates :math:`T_{2}^*` and :math:`S_0` for voxels
    with good data in only one echo as well, but uses the first two echoes for those voxels.
    The resulting "full" :math:`T_{2}^*` and :math:`S_0` maps are used throughout the rest of the pipeline.

.. image:: /_static/a05_loglinear_regression.png

The values of interest for the decay model, :math:`S_0` and :math:`T_{2}^*`,
are then simple transformations of the line's intercept (:math:`B_{0}`) and
slope (:math:`B_{1}`), respectively:

.. math:: S_{0} = e^{B_{0}}

.. math:: T_{2}^{*} = \frac{1}{B_{1}}

The resulting values can be used to show the fitted monoexponential decay model
on the original data.

.. image:: /_static/a06_monoexponential_decay_model.png

We can also see where :math:`T_{2}^*` lands on this curve.

.. image:: /_static/a07_monoexponential_decay_model_with_t2.png


.. _optimal combination:

*******************
Optimal combination
*******************

:func:`tedana.combine.make_optcom`

Using the :math:`T_{2}^*` estimates, ``tedana`` combines signal across echoes using a
weighted average.
The echoes are weighted according to the formula

.. math:: w_{TE} = TE * e^{\frac{-TE}{T_{2}^*}}

The weights are then normalized across echoes.
For the example voxel, the resulting weights are:

.. image:: /_static/a08_optimal_combination_echo_weights.png
  :width: 400 px
  :align: center

These normalized weights are then used to compute a weighted average that takes advantage
of the higher signal in earlier echoes and the higher sensitivity at later echoes.
The distribution of values for the optimally combined data lands somewhere
between the distributions for other echoes.

.. image:: /_static/a09_optimal_combination_value_distributions.png

The time series for the optimally combined data also looks like a combination
of the other echoes (which it is).
This optimally combined data is written out as **desc-optcom_bold.nii.gz**

.. image:: /_static/a10_optimal_combination_timeseries.png

.. note::
    An alternative method for optimal combination that
    does not use :math:`T_{2}^*` is the parallel-acquired inhomogeneity
    desensitized (PAID) ME-fMRI combination method (`Poser et al., 2006`_).
    This method specifically assumes that noise in the acquired echoes is "isotopic and
    homogeneous throughout the image," meaning it should be used on smoothed data.
    As we do not recommend performing tedana denoising on smoothed data,
    we discourage using PAID within the tedana workflow.
    We do, however, make it accessible as an alternative combination method
    in :func:`tedana.workflows.t2smap_workflow`.


*********
Denoising
*********

The next step is an attempt to remove noise from the data.
This process can be broadly separated into three steps: **decomposition**,
**metric calculation** and **component selection**.
Decomposition reduces the dimensionality of the optimally combined data using
`principal component analysis (PCA)`_ and then an `independent component analysis (ICA)`_.
Metrics that evaluate TE-dependence or independence are derived from these components.
Component selection uses these metrics in order to identify components that
should be kept in the data or discarded.
Unwanted components are then removed from the optimally combined data
to produce the denoised data output.

.. _principal component analysis (PCA): https://en.wikipedia.org/wiki/Principal_component_analysis
.. _independent component Analysis (ICA): https://en.wikipedia.org/wiki/Independent_component_analysis


******
TEDPCA
******

:func:`tedana.decomposition.tedpca`

The next step is to dimensionally reduce the data with TE-dependent principal
component analysis (PCA).
The goal of this step is to make it easier for the later ICA decomposition to converge.
Dimensionality reduction is a common step prior to ICA.
TEDPCA applies PCA to the optimally combined data in order to decompose it into component maps and
time series (saved as **desc-PCA_mixing.tsv**).
Here we can see time series for some example components (we don't really care about the maps):

.. image:: /_static/a11_pca_component_timeseries.png

These components are subjected to component selection, the specifics of which
vary according to algorithm.
Specifically, ``tedana`` offers three different approaches that perform this step.

The recommended approach (the default ``aic`` option, along with the ``kic`` and ``mdl`` options, for
``--tedpca``) is based on a moving average (stationary Gaussian) process
proposed by `Li et al (2007)`_ and used primarily in the Group ICA of fMRI Toolbox (GIFT).
A moving average process is the output of a linear system (which, in this case, is
a smoothing filter) that has an independent and identically distributed
Gaussian process as the input.
Simply put, this process more optimally selects the number of components for
fMRI data following a subsampling scheme described in `Li et al (2007)`_.

The number of selected principal components depends on the selection criteria.
For this PCA method in particular, ``--tedpca`` provides three different options
to select the PCA components based on three widely-used model selection criteria:

* ``mdl``: the Minimum Description Length (`MDL`_), which is the most aggressive option;
  i.e. returns the least number of components.
* ``kic``: the Kullback-Leibler Information Criterion (`KIC`_), which stands in the
  middle in terms of aggressiveness. You can see how KIC is related to AIC `here`_.
* ``aic``: the Akaike Information Criterion (`AIC`_), which is the least aggressive option;
  i.e., returns the largest number of components. We have chosen AIC as the default PCA
  criterion because it tends to result in fewer components than the Kundu methods, which increases
  the likelihood that the ICA step will successfully converge, but also, in our experience, retains
  enough components for meaningful interpretation later on.

.. note::
    Please, bear in mind that this is a data-driven dimensionality reduction approach. The default
    option ``aic`` might not yield perfect results on your data. Consider ``kic``
    and ``mdl`` options if running ``tedana`` with ``aic`` returns more components than expected.
    There is no definitively right number of components, but, for typical fMRI datasets, if the PCA
    explains more than 98% of the variance or if the number of components is more than half the number
    of time points, then it may be worth considering more aggressive thresholds.

The simplest approach uses a user-supplied threshold applied to the cumulative variance explained
by the PCA.
In this approach, the user provides a value to ``--tedpca`` between 0 and 1.
That value corresponds to the percent of variance that must be explained by the components.
For example, if a value of 0.9 is provided, then PCA components
(ordered by decreasing variance explained)
cumulatively explaining up to 90% of the variance will be retained.
Components explaining more than that threshold
(except for the component that crosses the threshold)
will be excluded.

In addition to the moving average process-based options and the variance explained threshold
described above,
we also support a decision tree-based selection method
(similar to the one in the :ref:`TEDICA` section below).
This method involves applying a decision tree to identify and discard PCA components which,
in addition to not explaining much variance, are also not significantly TE-dependent (i.e.,
have low Kappa) or TE-independent (i.e., have low Rho).
These approaches can be accessed using either the ``kundu`` or ``kundu_stabilize``
options for the ``--tedpca`` flag.

.. tip::
  For more information on how TE-dependence and TE-independence models are
  estimated in ``tedana``, see :ref:`dependence models`.
  For a more thorough explanation of this approach, consider the supplemental information
  in `Kundu et al (2013)`_.

After component selection is performed, the retained components and their
associated betas are used to reconstruct the optimally combined data, resulting
in a dimensionally reduced version of the dataset which is then used in the
:ref:`TEDICA` step.

.. image:: /_static/a12_pca_reduced_data.png
.. _AIC: https://en.wikipedia.org/wiki/Akaike_information_criterion
.. _KIC: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
.. _here: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Relationship_between_models_and_reality
.. _MDL: https://en.wikipedia.org/wiki/Minimum_description_length


.. _TEDICA:

******
TEDICA
******

:func:`tedana.decomposition.tedica`

Next, ``tedana`` applies TE-dependent independent component analysis (ICA) in
order to identify and remove TE-independent (i.e., non-BOLD noise) components.
The dimensionally reduced optimally combined data are first subjected to ICA in
order to fit a mixing matrix to the whitened data.
This generates a number of independent timeseries (saved as **desc-ICA_mixing.tsv**),
as well as parameter estimate maps which show the spatial loading of these components on the
brain (**desc-ICA_components.nii.gz**).

.. image:: /_static/a13_ica_component_timeseries.png

Linear regression is used to fit the component time series to each voxel in each
of the original, echo-specific data.
This results in echo- and voxel-specific betas for each of the components.
The beta values from the linear regression can be used to determine how the
fluctuations (in each component timeseries) change across the echo times.

TE-dependence (:math:`R_2` or :math:`1/T_{2}^*`) and TE-independence (:math:`S_0`) models can then
be fit to these betas.
These models allow calculation of F-statistics for the :math:`R_2` and :math:`S_0`
models (referred to as :math:`\kappa` and :math:`\rho`, respectively).

.. tip::
  For more information on how TE-dependence and TE-independence models are
  estimated, see :ref:`dependence models`.

The grey lines below shows how beta values (a.k.a. parameter estimates) change
with echo time, for one voxel and one component.
The blue and red lines show the predicted values for the :math:`S_0` and
:math:`T_2^*` models, respectively, for the same voxel and component.

.. image:: /_static/a14_te_dependence_models_component_0.png

.. image:: /_static/a14_te_dependence_models_component_1.png

.. image:: /_static/a14_te_dependence_models_component_2.png

A decision tree is applied to :math:`\kappa`, :math:`\rho`, and other metrics in order to
classify ICA components as TE-dependent (BOLD signal), TE-independent
(non-BOLD noise), or neither (to be ignored).
These classifications are saved in **desc-tedana_metrics.tsv**.
The actual decision tree is dependent on the component selection algorithm employed.
``tedana`` includes three options `tedana_orig`, `meica` and `minimal` (which uses hardcoded
thresholds applied to each of the metrics). `These decision trees are detailed here`_.

Components that are classified as noise are projected out of the optimally combined data,
yielding a denoised timeseries, which is saved as **desc-optcomDenoised_bold.nii.gz**.

.. image:: /_static/a15_denoised_data_timeseries.png

.. _These decision trees are detailed here: included_decision_trees.html

*******************************
Manual classification with RICA
*******************************

``RICA`` is a tool for manual ICA classification. Once the .tsv file containing the result of
manual component classification is obtained, it is necessary to `re-run the tedana workflow`_
passing the manual_classification.tsv file with the --ctab option. To save the output correctly,
make sure that the output directory does not coincide with the input directory. See `this example`_
presented at MRITogether 2022 for a hands-on tutorial.

.. _re-run the tedana workflow: https://tedana.readthedocs.io/en/stable/usage.html#Arguments%20for%20Rerunning%20the%20Workflow
.. _this example: https://www.youtube.com/live/P4cV-sGeltk?feature=share&t=1347


*********************************************
Removal of spatially diffuse noise (optional)
*********************************************

:func:`tedana.gscontrol.gscontrol_raw`, :func:`tedana.gscontrol.gscontrol_mmix`

Due to the constraints of ICA, TEDICA is able to identify and remove spatially
localized noise components, but it cannot identify components that are spread
out throughout the whole brain. See `Power et al. (2018)`_ for more information
about this issue.
One of several post-processing strategies may be applied to the ME-DN or ME-HK
datasets in order to remove spatially diffuse (ostensibly respiration-related)
noise.
Methods which have been employed in the past include global signal
regression (GSR), minimum image regression (MIR), anatomical CompCor, Go Decomposition (GODEC), and
robust PCA.
Currently, ``tedana`` implements GSR and MIR.

.. image:: /_static/a16_t1c_denoised_data_timeseries.png

.. _Power et al. (2018): http://www.pnas.org/content/early/2018/02/07/1720985115.short
.. _Poser et al., 2006: https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.20900

.. _physics section: https://tedana.readthedocs.io/en/latest/multi_echo.html
.. _Kundu et al (2013): https://www.ncbi.nlm.nih.gov/pubmed/24038744
.. _Li et al (2007): https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.20359
