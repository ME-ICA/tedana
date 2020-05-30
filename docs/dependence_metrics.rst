.. _dependence models:

TE (In)Dependence Models
````````````````````````

Functional MRI signal can be described in terms of fluctuations in :math:`S_0`
and :math:`T_2^*`.
In the below equation, :math:`S(t, TE_k)` is signal at a given time :math:`t`
and for a given echo time :math:`TE_k`.
:math:`\bar{S}(TE_k)` is the mean signal across time for the echo time
:math:`TE_k`.
:math:`{\Delta}{S_0}(t)` is the difference in :math:`S_0` at time :math:`t` from the average :math:`S_0` (:math:`\bar{S}_0`).
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
  will have influences from both S0 and T2*.

.. image:: /_static/b01_simulated_fluctuations.png

We can see that :math:`{\Delta}S(TE_k)` has very different curves for the two
simulated datasets.
Moreover, as expected, :math:`\frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)}` is flat
across echoes for the S0-fluctuating data and scales roughly linearly with TE
for the T2\*-fluctuating data.

We then fit our TE-dependence and TE-independence models to the
:math:`{\Delta}S(TE_k)` data, which gives us predicted data for each model for
each dataset.

.. image:: /_static/b02_model_fits.png

As expected, the S0 model fits perfectly to the S0-fluctuating dataset, while
the T2\* model fits quite well to the T2\*-fluctuating dataset.

The actual model fits can be calculated as F-statistics.
Then, the F-statistics per voxel are averaged across voxels into the Kappa and
Rho pseudo-F-statistics.

Applying our models to spatiotemporal decompositions
****************************************************

Now let us see how this extends to time series, components, and component
parameter estimates.

We have the means to simulate T2\*- and S0-based fluctuations, so here we have
generated two time series- one T2\*-based and one S0-based.
Both time series share the same level of percent signal change (a standard
deviation equivalent to 5\% of the mean), although the mean S0 (16000) is very
different from the mean T2* (30).

We can then average those two time series with different weights to create
components that are T2\*- or S0-based to various degrees.
In this case, both T2\* and S0 contribute equally to the simulated time series.
This simulated time series will act as our ICA component for this example.

.. image:: /_static/b03_component_timeseries.png

We also simulate multi-echo data for a single voxel with the same levels of
T2\* and S0 fluctuations as in the pure T2\* and S0 time series above.
Here we show time series for a subset of echo times.

.. image:: /_static/b04_echo_timeseries.png

And here we can see how those signals decay with echo time (again for only a
subset of echo times):

.. image:: /_static/b05_echo_value_distributions.png

We then run a regression for each echo's data against the component time series,
producing one parameter estimate for each echo time.
We can compare predicted T2* and S0 model values against the parameter estimates
in order to calculate single-voxel :math:`\rho` and :math:`\kappa` values.
Note that the metric values are extremely high, due to the inflated
degrees of freedom resulting from using so many echoes in the simulations.

.. attention::
   You may also notice that, despite the fact that T2* and S0 fluctuate the same
   amount and that both contributed equally to the component, :math:`\rho` is
   much higher than :math:`\kappa`.

.. image:: /_static/b06_component_model_fits.png
