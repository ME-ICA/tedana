.. _dependence models:

TE (In)Dependence Models
````````````````````````

Functional MRI signal can be described in terms of fluctuations in :math:`S_0`
and :math:`T_2^*`.
In the below equation, :math:`S(t, TE_k)` is signal at a given time :math:`t`
and for a given echo time :math:`TE_k`.
:math:`\bar{S}(TE_k)` is the mean signal across time for the echo time
:math:`TE_k`.
:math:`{\Delta}{S_0}(t)` is the difference in :math:`S_0` at time :math:`t`
from the average :math:`S_0` (:math:`\bar{S}_0`).
:math:`{\Delta}{R_2^*}(t)` is the difference in :math:`R_2^*` at time :math:`t`
from the average :math:`R_2^*`.

.. math::
  S(t, TE_k) = \bar{S}(TE_k) * (1 + \frac{{\Delta}{S_0}(t)}{\bar{S}_0} - {\Delta}{R_2^*}(t)*TE_k)

If we ignore time, this can be simplified to

.. math::
  \frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)} = \frac{{\Delta}S_0}{S_0}-{\Delta}{R_2^*}*TE_k

In order to evaluate whether signal change is being driven by fluctuations in
**either** :math:`S_0` **or** :math:`T_2^*`, one can break this overall model
into submodels by zeroing out certain terms.

For a **TE-independence model**, if there were no fluctuations in :math:`R_2^*`:

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

For TEDPCA/TEDICA, we use regression to get parameter estimates (PEs; not beta
values) for component time-series against echo-specific data, and substitute
those PEs for :math:`{\bar{S}(TE_k)}`.
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

As an example, let us simulate some data.
We will simulate signal decay for two time points, as well as the signal decay
for the hypothetical overall average over time.
In one time point, only S0 will fluctuate.
In the other, only R2* will fluctuate.

.. note::
  To make things easier, we're simulating these data with echo times of 0 to
  200 milliseconds, at 1ms intervals.
  In real life, you'll generally only have 3-5 echoes to work with.
  Real signal from each echo will be contaminated with random noise and will
  have influences from both S0 and R2*.

.. image:: /_static/simulated_fluctuations.png

We can see that :math:`{\Delta}S(TE_k)` has very different curves for the two
simulated datasets.
Moreover, as expected, :math:`\frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)}` is flat
across echoes for the S0-fluctuating data and scales roughly linearly with TE
for the R2*-fluctuating data.

We then fit our TE-dependence and TE-independence models to the
:math:`{\Delta}S(TE_k)` data, which gives us predicted data for each model for
each dataset.

.. image:: /_static/model_fits.png

As expected, the S0 model fits perfectly to the S0-fluctuating dataset, while
the R2* model fits quite well to the R2*-fluctuating dataset.

The actual model fits can be calculated as F-statistics.
Then, the F-statistics per voxel are averaged across voxels into the Kappa and
Rho pseudo-F-statistics.
