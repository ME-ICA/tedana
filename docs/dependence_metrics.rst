.. _dependence models:

########################
TE (In)Dependence Models
########################

Functional MRI signal can be described in terms of fluctuations in :math:`S_0`
and :math:`R_2^*`.
In the below equation, :math:`S(t, TE_k)` is signal at a given time :math:`t`
and for a given echo time :math:`TE_k`.
:math:`\bar{S}(TE_k)` is the mean signal across time for the echo time
:math:`TE_k`.
:math:`{\Delta}{S_0}(t)` is the difference in :math:`S_0` at time :math:`t` from the average :math:`S_0`.
:math:`{\Delta}{R_2^*}(t)` is the difference in :math:`R_2^*` at time :math:`t` from the average :math:`R_2^*`.

.. math::
  S(t, TE_k) = \bar{S}(TE_k) * (1 + \frac{{\Delta}{S_0}(t)}{\bar{S}_0} - {\Delta}{R_2^*}(t)*TE_k)

If we ignore time, this can be simplified to

.. math::
  \frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)} = \frac{{\Delta}S_0}{S_0}-{\Delta}{R_2^*}*TE_k

In order to evaluate whether signal change is being driven by fluctuations in
**either** :math:`S_0` **or** :math:`R_2^*`, one can break this overall model
into submodels by zeroing out certain terms.

.. important::
   Remember- :math:`R_2^*` is just :math:`\frac{1}{T_2^*}`

For a **TE-independence model**, if there were no fluctuations in :math:`R_2^*`:

.. math::
  \frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)} = \frac{{\Delta}S_0}{S_0}

  {\Delta}S(TE_k) = {\bar{S}(TE_k)} * \frac{{\Delta}S_0}{S_0}

Note that TE is not a parameter in this model.
Hence, this model is TE-independent.

Since :math:`\frac{{\Delta}S_0}{S_0}` is a scalar (i.e., doesn't change with
TE), it can be absorbed into a scalar coefficient, leaving :math:`{\bar{S}(TE_k)}`
(mean echo-wise signal) as the regressor.
The model is fit by regressing :math:`{\Delta}S(TE_k)` against
:math:`{\bar{S}(TE_k)}`:

.. math::
  {\Delta}S(TE_k) \approx c * {\bar{S}(TE_k)}

For a **TE-dependence model**, if there were no fluctuations in :math:`S_0`:

.. math::
  \frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)} = -{\Delta}{R_2^*}*TE_k

  {\Delta}S(TE_k) = {\bar{S}(TE_k)} * (-{\Delta}{R_2^*}) * TE_k

Since :math:`-{\Delta}R_2^*` is a scalar (does not vary with TE), it can be
absorbed into a scalar coefficient, leaving :math:`{\bar{S}(TE_k)} * TE_k` as the regressor.
The model is fit by regressing :math:`{\Delta}S(TE_k)` against
:math:`{\bar{S}(TE_k)} * TE_k`:

.. math::
  {\Delta}S(TE_k) \approx c * {\bar{S}(TE_k)} * TE_k

Note that TE is a parameter in this model. Hence, it is TE-dependent.

Now that we have each model, we can estimate :math:`c` for each voxel and evaluate
how well each model describes the data.
For a given voxel, the scalar coefficient :math:`\hat{c}` is estimated by ordinary
least squares:

.. math::
   \hat{c} = \frac{\sum_k {\Delta}S(TE_k) \cdot X(TE_k)}{\sum_k X(TE_k)^2}

where :math:`X(TE_k)` is :math:`\bar{S}(TE_k)` for the TE-independence model and
:math:`\bar{S}(TE_k) \cdot TE_k` for the TE-dependence model.

Using this estimate, the predicted signal change at each echo is:

.. math::
   \widehat{{\Delta}S}(TE_k) = \hat{c} \cdot X(TE_k)

The fit quality is then evaluated by comparing how much of the total signal the
model accounts for.
The total sum of squares and residual sum of squares are:

.. math::
   \alpha = \sum_k {\Delta}S(TE_k)^2

   SSE = \sum_k \left({\Delta}S(TE_k) - \widehat{{\Delta}S}(TE_k)\right)^2

and the pseudo-F-statistic is:

.. math::
   F = \frac{(\alpha - SSE) \cdot (E - 1)}{SSE}

where :math:`E` is the number of echoes.
In a standard F-statistic, :math:`\alpha` would be the sum of squared deviations
from the mean of :math:`{\Delta}S(TE_k)`, measuring how much better the model is
than simply predicting the mean.
Here, :math:`\alpha = \sum_k {\Delta}S(TE_k)^2` instead measures how much better
the model is than predicting no signal change at all.
These statistics are therefore called "pseudo" F-statistics.

.. topic:: Why not fit these models to the data directly?

  While it is possible to fit these models to the data directly,
  we do not necessarily want to know whether a voxel's time series is TE-dependent or TE-independent,
  since each voxel will contain a mixture of TE-dependent and TE-independent signals.
  Therefore, tedana relies on blind source separation using PCA or ICA to identify underlying signals (components) that may be TE-dependent or TE-independent,
  and evaluates those components instead.

The per-voxel F-statistic from the **TE-dependence model** is the per-voxel :math:`\kappa`,
and from the **TE-independence model** it is the per-voxel :math:`\rho`.
These per-voxel values are combined into a single component-level value by taking a
weighted average across voxels, where each voxel is weighted by the square of its
standardized parameter estimate for that component.
This gives more influence to voxels where the component has a stronger signal.
The resulting weighted averages are the component-level :math:`\kappa` and :math:`\rho`.


******************************************
Applying our models to signal decay curves
******************************************

As an example, let us simulate some data.
We will simulate signal decay for two time points, as well as the signal decay
for the hypothetical overall average over time.
In one time point, only :math:`S_0` will fluctuate.
In the other, only :math:`R_2^*` will fluctuate.

.. caution::
  To make things easier, we're simulating these data with echo times of 0 to
  200 milliseconds, at 1 ms intervals.
  In real life, you'll generally only have 3-5 echoes to work with.
  Real signal from each echo will also be contaminated with random noise and
  will have influences from both :math:`S_0` and :math:`R_2^*`.

.. image:: /_static/b01_simulated_fluctuations.png

We can see that :math:`{\Delta}S(TE_k)` has very different curves for the two
simulated datasets.
Moreover, as expected, :math:`\frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)}` is flat
across echoes for the :math:`S_0`-fluctuating data and scales roughly linearly with TE
for the :math:`R_2^*`-fluctuating data.

We then fit our TE-dependence and TE-independence models to the
:math:`{\Delta}S(TE_k)` data, which gives us predicted data for each model for
each dataset.

.. image:: /_static/b02_model_fits.png

As expected, the :math:`S_0` model fits perfectly to the :math:`S_0`-fluctuating dataset, while
the :math:`R_2^*` model fits quite well to the :math:`R_2^*`-fluctuating dataset.

The model fit quality at each voxel is quantified using the pseudo-F-statistics defined above.


****************************************************
Applying our models to spatiotemporal decompositions
****************************************************

Now let us see how this extends to time series, components, and component
parameter estimates.

We have the means to simulate :math:`R_2^*`- and :math:`S_0`-based fluctuations, so here we have
generated two time series- one :math:`R_2^*`-based and one :math:`S_0`-based.
Both time series share the same level of percent signal change (a standard
deviation equivalent to 5\% of the mean), although the mean :math:`S_0` (16000) is very
different from the mean :math:`R_2^*` (33.3 s⁻¹).

We can then average those two time series with different weights to create
components that are :math:`R_2^*`- or :math:`S_0`-based to various degrees.
In this case, both :math:`R_2^*` and :math:`S_0` contribute equally to the simulated time series.
This simulated time series will act as our ICA component for this example.

.. image:: /_static/b03_component_timeseries.png

We also simulate multi-echo data for a single voxel with the same levels of
:math:`R_2^*` and :math:`S_0` fluctuations as in the pure :math:`R_2^*` and :math:`S_0` time series above.
Here we show time series for a subset of echo times.

.. image:: /_static/b04_echo_timeseries.png

And here we can see how those signals decay with echo time (again for only a
subset of echo times):

.. image:: /_static/b05_echo_value_distributions.png

We then run a regression for each echo's data against the component time series,
producing one parameter estimate (PE) for each echo time.
The PEs correspond to :math:`{\Delta}S(TE_k)`, so the same TE-dependence and
-independence models apply directly with the PEs as the dependent variable.

For the **TE-independence model**, the component PEs are regressed against
:math:`{\bar{S}(TE_k)}`:

.. math::
  PE(TE_k) \approx c * {\bar{S}(TE_k)}

For the **TE-dependence model**, the component PEs are regressed against
:math:`{\bar{S}(TE_k)} * TE_k`:

.. math::
  PE(TE_k) \approx c * {\bar{S}(TE_k)} * TE_k

These are the per-voxel :math:`\kappa` and :math:`\rho` values defined in the previous section.
Note that the metric values are extremely high, due to the inflated
degrees of freedom resulting from using so many echoes in the simulations.

.. attention::
   You may also notice that, despite the fact that :math:`R_2^*` and :math:`S_0` fluctuate the same
   amount and that both contributed equally to the component, :math:`\rho` is
   much higher than :math:`\kappa`.

.. image:: /_static/b06_component_model_fits.png


***********************
Other Available Metrics
***********************

In addition to the core (in)dependence model metrics, TEDANA can also calculate the following metrics:


countnoise
==========
:func:`tedana.metrics.dependence.compute_countnoise`

The number of significant voxels in each component's standardized parameter estimate map
that are not in clusters.
In theory, these non-cluster voxels should be noise,
and if a component exhibits more non-cluster voxels than cluster voxels,
it is more likely to be noise.


countsigFT2
===========
:func:`tedana.metrics.dependence.compute_countsignal`

The number of significant voxels in each component's R2*-model F-statistic map
that are in clusters.
Having these "signal" voxels in the R2*-model F-statistic map is a good
indicator that the component is signal,
and having more cluster voxels in the S0-model F-statistic map than
in the R2*-model F-statistic map is a good indicator that the component is noise.


countsigFS0
===========
:func:`tedana.metrics.dependence.compute_countsignal`

The number of significant voxels in each component's S0-model F-statistic map
that are in clusters.
Having more cluster voxels in the S0-model F-statistic map than
in the R2*-model F-statistic map is a good indicator that the component is noise.


dice_FT2
========
:func:`tedana.metrics.dependence.compute_dice`

The Dice similarity index between each component's cluster-extent thresholded
R2*-model F-statistic map (using a p<0.05 threshold) and
its cluster-extent thresholded standardized parameter estimate map (using a 5% threshold).
This is a measure of the similarity between the R2*-model F-statistic map and the weight map.
If the standardized parameter estimate map has a higher DSI with the R2*-model F-statistic map than the S0-model F-statistic map,
it is more likely to be signal.


dice_FS0
========
:func:`tedana.metrics.dependence.compute_dice`

The Dice similarity index between each component's cluster-extent thresholded
S0-model F-statistic map (using a p<0.05 threshold) and
its cluster-extent thresholded standardized parameter estimate map (using a 5% threshold).
This is a measure of the similarity between the S0-model F-statistic map and the standardized parameter estimate map.
If the standardized parameter estimate map has a higher DSI with the S0-model F-statistic map than the R2*-model F-statistic map,
it is more likely to be noise.


signal-noise_t
==============
:func:`tedana.metrics.dependence.compute_signal_minus_noise_t`

A t-test is performed between the distributions of unique R2*-model F-statistics
associated with clusters (i.e., signal) and non-cluster voxels (i.e., noise) to
generate a t-statistic (metric ``signal-noise_t``) and p-value (metric ``signal-noise_p``)
measuring relative association of the component to signal over noise.


signal-noise_z
==============
:func:`tedana.metrics.dependence.compute_signal_minus_noise_z`

A t-test is performed between the distributions of R2*-model F-statistics
associated with clusters (i.e., signal) and non-cluster voxels (i.e., noise) to
generate a z-statistic (metric ``signal-noise_z``) and p-value (metric ``signal-noise_p``)
measuring relative association of the component to signal over noise.

This metric has not been used in the literature- the tedana developers created it
to address statistical issues with ``signal-noise_t``
(e.g., the fact that ``signal-noise_t`` limits inputs to unique F-statistics
and it later applies thresholds appropriate for z-statistics to the t-statistics).


variance explained
==================
:func:`tedana.metrics.dependence.calculate_varex`

The "variance explained" by each component is calculated as the square of the
parameter estimates from the regression of the mean-centered, but not z-scored,
optimally combined data against the component time series,
divided by the sum of the squares of the parameter estimates.

.. important::
  Please note that:

  - This is NOT variance explained (R^2).
  - Values sum to 100% by construction.
  - Shared variance among correlated components is implicitly distributed
    across coefficients in a model-dependent manner.
  - This metric reflects relative participation in the fitted model,
    not unique or marginal explanatory power.

  This corresponds to the quantity historically referred to as "variance
  explained" in tedana, but is more accurately described as relative
  coefficient energy.


normalized variance explained
=============================
:func:`tedana.metrics.dependence.calculate_varex`

The "normalized variance explained" by each component is calculated as the
square of the standardized parameter estimates from the regression of the z-scored
optimally combined data against the z-scored component time series,
divided by the sum of the squares of the standardized parameter estimates.

This is not actually a measure of normalized variance explained.

In the tedpca metrics, "normalized variance explained" actually comes from
the fitted PCA object's ``explained_variance_ratio_`` attribute,
and the TEDANA-calculated value is retained as "estimated normalized variance explained".


marginal R-squared
==================
:func:`tedana.metrics.dependence.calculate_marginal_r2`

The "marginal R-squared" by each component is calculated as 100 times the
squared correlation between the component time series and the data,
averaged over voxels.

This represents the variance in the data explained by each component
without controlling for other components.

partial R-squared
=================
:func:`tedana.metrics.dependence.calculate_partial_r2`

The "partial R-squared" by each component is calculated as the proportion (expressed
as a percentage) of variance uniquely explained by that component relative to the sum
of this uniquely explained variance and the variance that is not explained by the full model.

This is equivalent to the variance explained by each component after regressing the other
components out of the data *and* the component itself. It is a conditional effect size.


semi-partial R-squared
======================
:func:`tedana.metrics.dependence.calculate_semipartial_r2`

The "semi-partial R-squared" by each component is computed by first orthogonalizing that
component with respect to all other components (i.e., regressing it onto the remaining
components and taking the residuals). The squared Pearson correlation between this
orthogonalized regressor and the data is then computed for each voxel, and the
semi-partial R-squared for that component is the mean of these squared correlations
across voxels.

This corresponds to the variance in the data that is uniquely explained by each component,
after removing variance that is shared with the other components. It indicates the
incremental increase in R-squared when adding the target component to the model.


kappa_rho_difference
====================
:func:`tedana.metrics.dependence.compute_kappa_rho_difference`

The difference between the kappa and rho metrics is calculated as the absolute
value of the difference between the kappa and rho metrics divided by the sum of the kappa and rho metrics.

Higher values indicate that the component is more dominated by either kappa or rho,
which indicates "specificity" of the component to either TE-dependent or TE-independent signals.


d_table_score
=============
:func:`tedana.metrics.dependence.generate_decision_table_score`

A five-metric decision table is generated by ranking a number of metrics in either
descending or ascending order if they measure TE-dependence or TE-independence, respectively,
and then averaging the ranks.
The metrics are:
- kappa
- dice_FT2
- signal-noise_t
- countnoise
- countsigFT2

The decision table score is then calculated as the average of the ranks of the metrics.
