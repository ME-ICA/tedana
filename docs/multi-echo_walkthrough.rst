########################
What is multi-echo fMRI?
########################

.. admonition:: TL;DR

    Most echo-planar image (EPI) sequences collect a single brain image following
    a radio frequency (RF) pulse, at a rate known as the repetition time (TR).
    This typical approach is known as single-echo fMRI.

    In contrast, multi-echo (ME) fMRI refers to collecting data at multiple echo times,
    resulting in multiple volumes with varying levels of contrast acquired per RF pulse.
    Multi-echo fMRI can be used to identify and remove certain types of noise
    that can't be removed from single-echo data.

To understand what multi-echo fMRI is and why it's useful,
we will walk through a number of things.


*******************
The physics of fMRI
*******************

In a typical fMRI sequence,
the protons in a brain slice are aligned perpendicularly to the main magnetic
field of the scanner (:math:`B_0`) with an "excitation pulse".
Those protons start releasing energy from the excitation pulse as they fall back in line with :math:`B_0`,
and that energy is actively measured at an "echo time" (also known as `TE`_).

The time it takes for 37% of the protons fall back in line with :math:`B_0`
is known as the transverse relaxation time, or "T2".
The exact value of T2 varies across voxels in the brain, based on tissue type,
which is why protocols which are T2-weighted or which quantitatively measure T2 are useful for structural analyses.

fMRI works based on the fact that differences in blood oxygenation
(indicative of gross differences in neural metabolism ostensibly driven by neural activity)
also impact observed T2, also known as T2*.
Thus, changes in the measured signal from an fMRI sequence reflect (at least in part) changes in blood oxygenation.


*****************************
BOLD signal and BOLD contrast
*****************************

As mentioned above, energy is released after the excitation pulse, and this is our observed fMRI signal.
The echo time is the point at which that signal is recorded.

Let's take a look at how fMRI signal varies as a function of echo time.

.. image:: https://mfr.osf.io/render?url=https://osf.io/m7aw3/?direct%26mode=render%26action=download%26mode=render
   :alt: physics_signal_decay.png

As you can see, signal decays as echo time increases.
However, one very important feature of this signal decay is that the signal decays differently
depending on the level of blood oxygenation in the voxel.

When a voxel contains more deoxygenated blood,
its signal decays more slowly than when the blood within it is more oxygenated.

.. image:: https://mfr.osf.io/render?url=https://osf.io/ve7cf/?direct%26mode=render%26action=download%26mode=render
   :alt: physics_signal_decay_activity.png

This is the "BOLD contrast" functional neuroimagers care about.
Namely, you can compare the signal from a voxel during a cognitive task
(when you expect that voxel's brain region to be more "active")
against that voxel's signal from a different cognitive task in order to determine if there is a
meaningful different in activation between the two tasks.

There are two relevant features to the BOLD contrast.
First, as repeatedly noted above, overall signal decays as echo time increases.
Eventually, with a long enough echo time, there is not enough signal to detect over noise
(i.e., signal-to-noise ratio is too low)
and meaningful signal cannot be observed.
Second, BOLD contrast increases as echo time increases.
That is, the relative difference in signal between active and inactive states increases with echo time,
even as the overall signal decreases.

While the signal decay curve can be described using many models,
one of the most useful approximations (and the one we used to simulate the signals in this walkthrough)
is a monoexponential decay curve.
In this model, the signal is driven by two factors:
the "intercept" (signal at echo time = 0s), also known as S0; and the "slope" (decay rate), also known as T2*.
For more information about different models of fMRI signal decay, please see XXXXX.

.. note:: T2*
    We said above that observed T2 is T2*, and now we're saying that the slope of the decay is T2*.
    That's because they're the same thing!

In the above figure, the difference between the "inactive" and "active" signals was driven by a change in T2* from 20ms to 40ms.
Thus, the point at which the difference in signal is maximized between the two states in this voxel is ~30ms.

And, in fact, this is the standard approach for the most common version of fMRI: single-echo fMRI.


*************************
What is single-echo fMRI?
*************************

Above, we saw that the difference in BOLD signal between active and inactive voxels is maximized at a specific echo time.

With single-echo fMRI, we have one echo time for each excitation pulse.
As such, we record one data point for each voxel, at each time point.
That data point is assumed to reflect blood oxygenation-level dependent signal,
and the single echo time is chosen to maximize BOLD contrast across the brain.

However, a major drawback to this approach is that it ignores several features of fMRI data:

1. T2* varies across voxels in the brain, depending on features like tissue type distribution, brain-air boundaries, etc.
   As such, the "optimal" echo time _also_ varies across the brain.
2. Fluctuations in voxel activation (i.e., T2*) mean that T2* (and thus optimal echo time)
   varies _within_ each voxel as well as between voxels.
   Basically, "activation" is not a binary state: a voxel can be active at many different levels, and each level reflects a change in T2*.
3. With only a single data point at each time point, there is no way to characterize the signal decay curve.
   Unfortunately, there are multiple factors which can influence how signal decays, and many of them are problematic.
   With single-echo fMRI, we must rely on more standard signal- and image-processing techniques to remove noise.

Taking these drawbacks into account, one might wonder why anyone would use single-echo fMRI
when there _must_ be something called multi-echo fMRI (what with that being the topic of this walkthrough).
There are a bunch of reasons, but perhaps the biggest is that fMRI is always balancing the amount of data against its utility.
You can always get more useful information from data that is closer to its original form (e.g., in k-space, before combining across coils),
but this involves increased work for the researcher, in that they must choose the appropriate tools and process all of those data,
as well as _massively_ increased storage requirements.

Single-echo fMRI performs quite well considering its limitations.
While T2* varies across the brain, for _most_ regions the typical echo time for a given magnetic strength
(e.g., 30ms for 3T) will result in sufficient signal-to-noise ratio (SNR).
BOLD contrast can reliably be detected over noise, and in most analyses the SNR is sufficient.
With a standard univariate analysis, more data will generally swamp issues like thermal noise,
especially when those issues are not correlated with measures of interest (like cognitive tasks).


==========================================
Sources of fMRI signal fluctuations
==========================================

There are, in fact, many factors that impact observed fMRI signal, but we will focus on a small number.

1. Neurally-driven BOLD signal.
   This is the signal we generally care about in fMRI.
2. Non-BOLD noise.
   This noise is often driven by things like instrument noise, subject motion, and thermal noise.
3. Non-neural BOLD signal.
   There are physiological sources of changes in blood oxygenation that are unrelated to neural activity,
   including heart rate and breathing changes.

Let's take a look at what single-echo data looks like over time.

.. image:: https://mfr.osf.io/render?url=https://osf.io/g9dqc/?direct%26mode=render%26action=download%26mode=render
   :alt: fluctuations_single-echo.gif

As you can see, the single data point fluctuates over time.
Let's assume that those fluctuations reflect meaningful BOLD signal.
Nothing to be concerned about, right?

Okay, let's check out the underlying signal decay curve we're sampling from.

.. image:: https://mfr.osf.io/render?url=https://osf.io/5yjwx/?direct%26mode=render%26action=download%26mode=render
   :alt: fluctuations_single-echo_with_curve.gif

Everything still looks fine, right?
We know there's an underlying signal decay curve, and we're sampling that curve at a single point, at our TE.

What if we describe the curve in terms of S0 and T2*?

.. image:: https://mfr.osf.io/render?url=https://osf.io/6a7nv/?direct%26mode=render%26action=download%26mode=render
   :alt: fluctuations_single-echo_with_curve_and_t2s_s0.gif

Now we see that the changes in the signal are driven by changes in _both_ S0 and T2*.
Why should we care about that?
Well, we know that T2* reflects BOLD signal, but we don't really care about S0.
In fact, S0 changes are driven by non-BOLD noise.
Things like motion, thermal noise, instrument noise, etc.

So if our observed signal is affected by both S0 and T2*,
and the S0 changes are introducing noise into our data,
is there anything we can do?

Well, first, let's see if there's a way to tell S0-based fluctuations and T2*-based fluctuations apart.
We'll plot two signal decay curves.
One will _only_ include S0 changes and the other will only include T2* changes.

To make sure we can _really_ see the curves, we'll also make the S0 and T2* changes roughly equivalent.
They have different scales, so we'll use the same time series of fluctuations,
scaled to have matching percent signal changes between the two values.

.. image:: https://mfr.osf.io/render?url=https://osf.io/g29ez/?direct%26mode=render%26action=download%26mode=render
   :alt: fluctuations_t2s_s0.gif

Hey, look at that!
The curves change differently!
If you look at the whole curve, you can differentiate S0 changes from T2* changes.

Now that we know that, what about single-echo fMRI?

.. image:: https://mfr.osf.io/render?url=https://osf.io/mx4ku/?direct%26mode=render%26action=download%26mode=render
   :alt: fluctuations_t2s_s0_single-echo.gif

Hm... with only one data point per time point, we really can't tell whether the changes are due to S0 or T2*.

What if... what if we had _multiple_ data points for each volume?

***************
Multi-echo fMRI
***************

Multi-echo fMRI involves defining and acquiring multiple echo times in your sequence.
Instead of sampling the decay curve at one point, you sample at multiple points.

Typical multi-echo protocols use somewhere between three and five echoes,
though more are possible if you make certain compromises with your parameters.

Here we have some simulated data with six echoes.

.. image:: https://mfr.osf.io/render?url=https://osf.io/mf3ae/?direct%26mode=render%26action=download%26mode=render
   :alt: fluctuations_t2s_s0_multi-echo.gif

Now we can tell the two curves apart again!

Okay, so what does this all mean?
Simply put, you need multiple echoes in order to differentiate S0 and T2* fluctuations in your fMRI data.


.. _multi-echo physics2:

******************************
The physics of multi-echo fMRI
******************************

Multi-echo fMRI data is obtained by acquiring multiple echo times (commonly called
`TEs`_) for each MRI volume during data collection.
While fMRI signal contains important neural information (termed the blood
oxygen-level dependent, or `BOLD signal`_,
it also contains "noise" (termed non-BOLD signal) caused by things like
participant motion and changes in breathing.
Because the BOLD signal is known to decay at a set rate, collecting multiple
echos allows us to assess non-BOLD.

The image below shows the basic relationship between echo times and the image acquired at
3T (top, A) and 7T (bottom, B). Note that the earliest echo time is the brightest, as the
signal has only had a limited amount of time to decay.
In addition, the latter echo times show areas in which is the signal has decayed completely ('drop out')
due to inhomogeneity in the magnetic field.
By using the information across multiple echos these images can be combined in
an optimal manner to take advantage of the signal
in the earlier echos (see :ref:`optimal combination`).

.. image:: https://mfr.osf.io/render?url=https://osf.io/m7aw3/?direct%26mode=render%26action=download%26mode=render
   :alt: physics_signal_decay.png

.. image:: https://mfr.osf.io/render?url=https://osf.io/m7aw3/?direct%26mode=render%26action=download%26mode=render
   :alt: physics_multiple_echos.png

In order to classify the relationship between the signal and the echo time we can consider a
single voxel at two timepoints (x and y) and the measured signal measured at three different echo times - :math:`S(TE_n)`.

For the left column, we are observing a change that we term :math:`{\Delta}{S_0}` - that is a change
in the intercept or raw signal intensity.
A common example of this is participant movement, in which the voxel (which is at a static
location within the scanner) now contains different tissue or even an area outside of the brain.

As we have collected three separate echos, we can compare the change in signal at each echo time, :math:`{\Delta}{S(TE_n)}`.
For  :math:`{\Delta}{S_0}` we see that this produces a decaying curve.
If we compare this to the original signal, as in :math:`\frac{{\Delta}{S(TE_n)}}{S(TE_n)}`
we see that there is no echo time dependence, as the final plot is a flat line.

In the right column, we consider changes that are related to brain activity.
For example, imagine that the two brain states here (x and y) are a baseline and task activated state respectively.
This effect is a change in in :math:`{\Delta}{R_2^*}` which is equivalent
to the inverse of :math:`{T_2^*}`.
We typically observe this change in signal amplitude occurring over volumes with
the hemodynamic response, while here we are examining the change in signal over echo times.
Again we can plot the difference in the signal between these two states as a function of echo time,
finding that the signal rises and falls.
If we compare this curve to the original signal we find
that the magnitude of the changes is dependent on the echo time.

For a more comprehensive review of these topics and others, see `Kundu et al. (2017)`_.

.. _TE: http://mriquestions.com/tr-and-te.html
.. _TEs: http://mriquestions.com/tr-and-te.html
.. _BOLD signal: http://www.fil.ion.ucl.ac.uk/spm/course/slides10-zurich/Kerstin_BOLD.pdf
.. _Kundu et al. (2017): https://www.sciencedirect.com/science/article/pii/S1053811917302410?via%3Dihub


*******************
Why use multi-echo?
*******************

There are many potential reasons an investigator would be interested in using multi-echo EPI (ME-EPI).
Among these are the different levels of analysis ME-EPI enables.
Specifically, by collecting multi-echo data, researchers are able to:

**Compare results across different echos**: currently, field standards are largely set using single-echo EPI.
Because multi-echo is composed of multiple single-echo time series, each of these can be analyzed separately
and compared to one another.

**Combine the results by weighted averaging**: Rather than analyzing single-echo time series separately,
we can combine them into an "optimally combined time series".
For more information on this combination, see :ref:`optimal combination`.
Optimally combined data exhibits higher SNR and improves statistical power of analyses in regions
traditionally affected by drop-out.

**Denoise the data based on information contained in the echos**: Collecting multi-echo data allows
access to unique denoising methods.
ICA-based denoising methods like ICA-AROMA (`Pruim et al. (2015)`_)
have been shown to significantly improve the quality of cleaned signal.
These methods, however, have comparably limited information, as they are designed to work with single-echo EPI.

``tedana`` is an ICA-based denoising pipeline built especially for
multi-echo data. Collecting multi-echo EPI allows us to leverage all of the information available for single-echo datasets,
as well as additional information only available when looking at signal decay across multiple TEs.
We can use this information to denoise the optimally combined time series.

.. _Pruim et al. (2015): https://www.sciencedirect.com/science/article/pii/S1053811915001822
