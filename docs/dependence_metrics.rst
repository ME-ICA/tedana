.. _dependence models:

########################
TE (In)Dependence Models
########################

Functional MRI signal can be described in terms of fluctuations in :math:`S_0`
and :math:`T_2^*`.
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
**either** :math:`S_0` **or** :math:`T_2^*`, one can break this overall model
into submodels by zeroing out certain terms.

.. important::
   Remember- :math:`R_2^*` is just :math:`\frac{1}{T_2^*}`

For a **TE-independence model**, if there were no fluctuations in :math:`T_2^*`:

.. math::
  \frac{{\Delta}S(TE_k)}{\bar{S(TE_k)}} = \frac{{\Delta}S_0}{S_0}

  {\Delta}S(TE_k) = {\bar{S}(TE_k)} * \frac{{\Delta}S_0}{S_0}

Note that TE is not a parameter in this model.
Hence, this model is TE-independent.

Also, :math:`\frac{{\Delta}S_0}{S_0}` is a scalar (i.e., doesn't change with
TE), so we can just ignore that, which means we only use :math:`{\bar{S}(TE_k)}`
(mean echo-wise signal).

Thus, the model becomes :math:`{\Delta}S(TE_k) = {\bar{S}(TE_k)} * X`, where we
fit X to the data using regression and evaluate model fit.

For TEDPCA/TEDICA, we use regression to get parameter estimates (raw PEs; not
standardized beta values) for component time-series against echo-specific data,
and substitute those PEs for :math:`{\bar{S}(TE_k)}`.
Thus, to assess the TE-independence of a component, we use the model
:math:`PE(TE_k) = {\bar{S}(TE_k)} * X`, fit X to the data, and evaluate model
fit.

For a **TE-dependence model**, if there were no fluctuations in :math:`S_0`:

.. math::
  \frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)} = -{\Delta}{R_2^*}*TE_k

  {\Delta}S(TE_k) = {\bar{S}(TE_k)} * -{\Delta}{R_2^*}*TE_k

  {\Delta}S(TE_k) = {\bar{S}(TE_k)} * TE_k * X

  PE(TE_k) = {\bar{S}(TE_k)} * TE_k * X

Note that TE is a parameter in this model. Hence, it is TE-dependent.


******************************************
Applying our models to signal decay curves
******************************************

As an example, let us simulate some data.
We will simulate signal decay for two time points, as well as the signal decay
for the hypothetical overall average over time.
In one time point, only :math:`S_0` will fluctuate.
In the other, only :math:`T_2^*` will fluctuate.

.. caution::
  To make things easier, we're simulating these data with echo times of 0 to
  200 milliseconds, at 1ms intervals.
  In real life, you'll generally only have 3-5 echoes to work with.
  Real signal from each echo will also be contaminated with random noise and
  will have influences from both :math:`S_0` and :math:`T_2^*`.

.. image:: /_static/b01_simulated_fluctuations.png

We can see that :math:`{\Delta}S(TE_k)` has very different curves for the two
simulated datasets.
Moreover, as expected, :math:`\frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)}` is flat
across echoes for the :math:`S_0`-fluctuating data and scales roughly linearly with TE
for the :math:`T_2^*`-fluctuating data.

We then fit our TE-dependence and TE-independence models to the
:math:`{\Delta}S(TE_k)` data, which gives us predicted data for each model for
each dataset.

.. image:: /_static/b02_model_fits.png

As expected, the :math:`S_0` model fits perfectly to the :math:`S_0`-fluctuating dataset, while
the :math:`T_2^*` model fits quite well to the :math:`T_2^*`-fluctuating dataset.

The actual model fits can be calculated as F-statistics.
Then, the F-statistics per voxel are averaged across voxels into the Kappa and
Rho pseudo-F-statistics.


****************************************************
Applying our models to spatiotemporal decompositions
****************************************************

Now let us see how this extends to time series, components, and component
parameter estimates.

We have the means to simulate :math:`T_2^*`- and :math:`S_0`-based fluctuations, so here we have
generated two time series- one :math:`T_2^*`-based and one :math:`S_0`-based.
Both time series share the same level of percent signal change (a standard
deviation equivalent to 5\% of the mean), although the mean :math:`S_0` (16000) is very
different from the mean :math:`T_2^*` (30).

We can then average those two time series with different weights to create
components that are :math:`T_2^*`- or :math:`S_0`-based to various degrees.
In this case, both :math:`T_2^*` and :math:`S_0` contribute equally to the simulated time series.
This simulated time series will act as our ICA component for this example.

.. image:: /_static/b03_component_timeseries.png

We also simulate multi-echo data for a single voxel with the same levels of
:math:`T_2^*` and :math:`S_0` fluctuations as in the pure :math:`T_2^*` and :math:`S_0` time series above.
Here we show time series for a subset of echo times.

.. image:: /_static/b04_echo_timeseries.png

And here we can see how those signals decay with echo time (again for only a
subset of echo times):

.. image:: /_static/b05_echo_value_distributions.png

We then run a regression for each echo's data against the component time series,
producing one parameter estimate for each echo time.
The parameter estimates match the signal decay curve for :math:`{\Delta}S(TE_k)`,
as seen above.
We can thus apply the same TE-dependence and -independence models as above,
in order to calculate single-voxel :math:`\rho` and :math:`\kappa` values.
Note that the metric values are extremely high, due to the inflated
degrees of freedom resulting from using so many echoes in the simulations.

.. attention::
   You may also notice that, despite the fact that :math:`T_2^*` and :math:`S_0` fluctuate the same
   amount and that both contributed equally to the component, :math:`\rho` is
   much higher than :math:`\kappa`.

.. image:: /_static/b06_component_model_fits.png


***********************
Other Available Metrics
***********************

In addition to the core (in)dependence model metrics, TEDANA can also calculate the following metrics:


HFC
===
:func:`tedana.metrics.frequency.calculate_hfc`

High-frequency content (HFC) is calculated from each component time series.
The component's one-sided power spectrum is calculated, frequencies at or below
the high-pass cutoff are removed, and HFC is the normalized frequency where the
cumulative retained power is closest to 50%.
Values are bounded between 0 and 1.
Values near 0 indicate that the retained power is concentrated near the high-pass
cutoff, while values near 1 indicate that the retained power is concentrated closer
to the Nyquist frequency.

The commonly used threshold of ``HFC > 0.35`` is inherited from single-echo
ICA-AROMA :footcite:p:`pruim2015ica`.
It has not yet been validated as a tedana ME-fMRI component classification threshold,
though there is no reason to believe that ME-fMRI components would have different
frequency characteristics than single-echo components.


max_rp_corr
===========
:func:`tedana.metrics.external.calculate_max_rp_corr`

``max_rp_corr`` is an external-regressor metric calculated from a user-selected
regressor set, typically motion parameters.
For each component, tedana computes the mean across random 90% timepoint subsamples
of the maximum absolute Pearson correlation between the component time series and
an expanded regressor model.
The expanded model contains the original N regressors, their derivatives, and both
sets shifted forward and backward by one TR, for 6*N model columns.
Correlations are calculated for the raw time series and for element-wise squared
time series, giving 12*N total comparisons per split.

Values are bounded between 0 and 1, where higher values indicate stronger
association between the component and at least one expanded regressor.

This metric is taken from ICA-AROMA :footcite:p:`pruim2015ica`,
but the corresponding classification step cannot be reproduced within tedana,
as ICA-AROMA combines this metric with its "edge fraction" metric,
which has not been implemented in tedana.
As such, any decision-tree thresholds based on this metric must be considered experimental.


spike
=====
:func:`tedana.metrics.temporal.compute_spike`

The Fisher (excess) kurtosis of each component's time series, after linear detrending.
A transient artifact- a single-volume motion jump or a spike in a few TRs-
concentrates a large fraction of a component's variance into a handful of timepoints,
producing a heavy-tailed (leptokurtic) distribution that kurtosis is built to detect.
Smooth or oscillatory signal (task blocks, drift, respiration) spreads its variance
across many timepoints and stays near-Gaussian, so it scores low.
Linear detrending first ensures a strong drift is not mistaken for a transient.
The metric is sign- and scale-invariant, and requires only the mixing matrix.

This metric reports *that* a component is spiky, not *when*.
Because kurtosis is maximized by a single dominant outlier,
a component with many spread-out spikes can score *lower* than a single-spike one,
and genuinely sparse signal (e.g., rare event-related responses) also produces heavy tails.
No threshold is provided, as kurtosis has no natural cut-off.


Multiband (slice-leakage) artifacts
===================================

The next four metrics (``slice_banding``, ``slice_leakage``,
``slice_leakage_aliasing_z``, and ``slice_leakage_periodicity_z``) all detect the
same artifact: simultaneous multi-slice leakage.

Modern fMRI often uses **simultaneous multi-slice (SMS)**, or **multiband**,
acquisition to scan faster: instead of exciting one slice at a time, the scanner
excites several evenly-spaced slices at once and separates them afterward.
When that separation is imperfect, signal from one slice "leaks" into the other
slices excited alongside it.
Those co-excited slices sit a fixed distance apart (``n_slices / mb_factor``),
so the leakage shows up as a **regular banding pattern** through the volume,
with slices at that spacing looking suspiciously alike.

In an ICA component's spatial map, this banding is a tell-tale sign that the
component reflects an acquisition artifact rather than neural signal,
so components with strong banding are good noise candidates.
The metrics below quantify that banding in different ways: ``slice_banding`` is a
quick check that needs no acquisition metadata, while the ``slice_leakage`` family
uses the true slice direction and multiband factor to target the exact leakage
spacing and calibrate its scores against chance.

The leakage leaves two distinct fingerprints, and ``slice_leakage`` requires both:
the co-excited slices become partial *copies of one another* (**aliasing**),
and each slice's *overall brightness* settles into a repeating rhythm
(**periodicity**).


slice_banding
=============
:func:`tedana.metrics.spatial.compute_slice_banding`

**In plain terms:** ``slice_banding`` flags components whose spatial map has
stripes that alternate from one slice to the next -- the fingerprint of multiband
leakage.
Higher values mean stronger, more regular alternating bands.
It checks all three image axes and reports the strongest banding it finds,
so it needs no information about how the data were acquired.
Because it only looks for *alternating* (high-frequency) bands, smooth patterns
from anatomy, coil sensitivity, or drift don't trigger it.

A spatial metric flagging components whose weight map has a slice-to-slice banding
structure characteristic of simultaneous multi-slice (SMS / "multiband") acceleration leakage.
Operating on each component's squared weight (standardized parameter estimate) map,
for each of the three array axes two quantities are computed:
``bandR2``, the fraction of in-mask weight variance explained by the per-slice mean profile
(the *magnitude* of banding),
and ``bandMB``, the fraction of that profile's power spectrum in its upper half after linear detrending
(the alternating-slice *character* of the banding).
The axis score is ``bandR2 * bandMB``, and the metric is the maximum over the three axes.

Multiplying the two terms means a component is only flagged when the banding is both strong
and high-frequency, which separates true acceleration leakage from benign smooth gradients
(coil sensitivity, anatomy, drift), whose slice profiles are low-frequency and removed by
the detrend step.
Squaring the weights sharpens the contrast, since multiband banding often alternates in sign
and a signed per-slice mean would partly cancel.
The metric is axis-agnostic within the array frame and scale-invariant.

This metric assumes the slice direction is aligned with an array axis,
so it should be run in (or close to) native acquisition space-
after resampling to a rotated grid the banding becomes oblique and is largely missed.
It targets *alternating* banding specifically and is not a general slice-dropout detector.


slice_leakage
=============
:func:`tedana.metrics.spatial.compute_slice_leakage`

**In plain terms:** ``slice_leakage`` is a more targeted version of
``slice_banding``.
Using the acquisition metadata, it looks specifically at the slices that were
excited together and asks two questions: do they contain the same in-plane picture
(``aliasing``), and does the slice-by-slice brightness repeat at the multiband
spacing (``periodicity``)?
Both must hold for a high score.
Crucially, ``slice_leakage`` adds the in-plane *aliasing* check that
``slice_banding`` lacks: ``slice_banding`` collapses each slice to a single number
and never compares whether co-excited slices are copies of one another.
Scores are z-scores calibrated against chance, so a value like ``z > 3`` means
"far more banding than you'd expect by coincidence."

A metadata-aware, null-calibrated detector for the same SMS/multiband leakage as ``slice_banding``.
Operating on each component's squared weight map along the true slice axis
(from the ``SliceEncodingDirection``/``SliceTiming`` metadata, or inferred from the affine)
and using the multiband factor
(from ``MultibandAccelerationFactor``, or inferred from ``SliceTiming``),
it computes two z-scored statistics against a slice-index permutation null-
``slice_leakage_aliasing_z`` and ``slice_leakage_periodicity_z``
(each documented separately below).
The reported ``slice_leakage`` is the **minimum** of the two z-scores,
so both signatures must be present for a component to score highly.
When the multiband factor is unknown,
candidate spacings are scanned and the null applies the same scan,
keeping the z-scores calibrated.

Relative to ``slice_banding``, this uses the true slice axis instead of a max-over-axes
heuristic, targets the specific aliasing frequency rather than a broad upper-half-spectrum
rule, preserves in-plane information, and yields a portable z-score (e.g., ``z > 3``)
instead of a fixed raw cutoff.
It returns 0 when there is no simultaneous multi-slice acquisition (``mb_factor < 2``)
or when no candidate slice spacing divides the slice count.
Its main cost is the permutation null (configurable ``n_permutations``, default 256),
and it shares the axis-alignment assumption of ``slice_banding``.


slice_leakage_aliasing_z
========================
:func:`tedana.metrics.spatial.compute_slice_leakage`

**In plain terms:** this score looks at the *2D image inside each slice*.
Multiband leakage copies part of one slice's picture into the slices excited
alongside it, so those co-excited slices end up containing the same spatial
features.
The score is high when slices acquired at the same time are unusually similar
*as images* -- the same blobs showing up in each.
It says nothing about a slice's overall brightness;
it only asks whether co-excited slices contain the **same picture**
(contrast with ``slice_leakage_periodicity_z``, which tracks per-slice brightness
instead).

The aliasing z-score component of ``slice_leakage``, available on its own.
Slices from the same slice group, separated by ``n_slices / mb_factor``, are made abnormally
similar by leakage; this metric is the mean within-group Pearson correlation between
their in-plane images (computed on the squared weight map), z-scored against a
slice-index permutation null.
Because of the null calibration it measures *excess* correlation beyond the overall
inter-slice trend, so a large positive value indicates aliasing-specific similarity
rather than generic spatial smoothness.


slice_leakage_periodicity_z
===========================
:func:`tedana.metrics.spatial.compute_slice_leakage`

**In plain terms:** this score ignores what each slice looks like and tracks only
its *overall brightness* (mean energy), one number per slice.
It is high when that slice-to-slice brightness rises and falls in a steady rhythm
repeating at the multiband spacing -- every ``g``-th slice unusually bright or dark.
Where ``slice_leakage_aliasing_z`` asks "do the co-excited slices contain the same
picture?", this asks "does the amount of signal per slice beat at the leakage
spacing?"

The periodicity z-score component of ``slice_leakage``, available on its own.
This is the fraction of the per-slice energy profile's variance explained by
period-``g`` group means (a one-way ANOVA R^2, with ``g = n_slices / mb_factor``),
z-scored against a slice-order permutation null.
A large positive value indicates a periodic slice-to-slice energy pattern at the
aliasing spacing, the spatial signature of multiband leakage.


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

The number of significant voxels in each component's T2*-model F-statistic map
that are in clusters.
Having these "signal" voxels in the T2*-model F-statistic map is a good
indicator that the component is signal,
and having more cluster voxels in the S0-model F-statistic map than
in the T2*-model F-statistic map is a good indicator that the component is noise.


countsigFS0
===========
:func:`tedana.metrics.dependence.compute_countsignal`

The number of significant voxels in each component's S0-model F-statistic map
that are in clusters.
Having more cluster voxels in the S0-model F-statistic map than
in the T2*-model F-statistic map is a good indicator that the component is noise.


dice_FT2
========
:func:`tedana.metrics.dependence.compute_dice`

The Dice similarity index between each component's cluster-extent thresholded
T2*-model F-statistic map (using a p<0.05 threshold) and
its cluster-extent thresholded standardized parameter estimate map (using a 5% threshold).
This is a measure of the similarity between the T2*-model F-statistic map and the weight map.
If the standardized parameter estimate map has a higher DSI with the T2*-model F-statistic map than the S0-model F-statistic map,
it is more likely to be signal.


dice_FS0
========
:func:`tedana.metrics.dependence.compute_dice`

The Dice similarity index between each component's cluster-extent thresholded
S0-model F-statistic map (using a p<0.05 threshold) and
its cluster-extent thresholded standardized parameter estimate map (using a 5% threshold).
This is a measure of the similarity between the S0-model F-statistic map and the standardized parameter estimate map.
If the standardized parameter estimate map has a higher DSI with the S0-model F-statistic map than the T2*-model F-statistic map,
it is more likely to be noise.


signal-noise_t
==============
:func:`tedana.metrics.dependence.compute_signal_minus_noise_t`

A t-test is performed between the distributions of unique T2*-model F-statistics
associated with clusters (i.e., signal) and non-cluster voxels (i.e., noise) to
generate a t-statistic (metric ``signal-noise_t``) and p-value (metric ``signal-noise_p``)
measuring relative association of the component to signal over noise.


signal-noise_z
==============
:func:`tedana.metrics.dependence.compute_signal_minus_noise_z`

A t-test is performed between the distributions of T2*-model F-statistics
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


References
==========
.. footbibliography::
