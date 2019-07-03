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

In order to evaluate whether signal is being driven by fluctuations in **either**
:math:`S_0` **or** :math:`T_2^*`, one can break this overall model into submodels
by zeroing out certain terms.

For a **TE-independence model**, if there were no fluctuations in :math:`R_2^*`:

.. math::
  \frac{{\Delta}S(TE_k)}{\bar{S(TE_k)}} = \frac{{\Delta}S_0}{S_0}

  {\Delta}S(TE_k) = {\bar{S}(TE_k)} * \frac{{\Delta}S_0}{S_0}

Note that TE is not a parameter in this model. Hence, it is TE-independent.

For a **TE-dependence model**, if there were no fluctuations in :math:`S_0`:

.. math::
  \frac{{\Delta}S(TE_k)}{\bar{S}(TE_k)} = -{\Delta}{R_2^*}*TE_k

  {\Delta}S(TE_k) = {\bar{S}(TE_k)} * -{\Delta}{R_2^*}*TE_k

Note that TE is a parameter is this model. Hence, it is TE-dependent.

Now we can predict how signal should change with fluctuations in :math:`S_0` or
:math:`R_2^*` across echoes.

Since :math:`\frac{{\Delta}S_0}{S_0}` and :math:`-{\Delta}{R_2^*}` are unknown,
we can estimate each from the observed timepoint's data
(:math:`{\Delta}S(TE_k)`), the mean data (:math:`{\bar{S}(TE_k)}`), and
echo times, using linear regression.
The goal of the regression isn't to estimate these two values, per se, but
rather to evaluate the fit of the predicted model

.. math::
  {\Delta}S(TE_k) = {\bar{S}(TE_k)} * -{\Delta}{R_2^*}*TE_k
