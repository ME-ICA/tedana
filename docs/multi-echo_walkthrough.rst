What is multi-echo fMRI?
========================

.. admonition:: TL;DR

    Most echo-planar image (EPI) sequences collect a single brain image following
    a radio frequency (RF) pulse, at a rate known as the repetition time (TR).
    This typical approach is known as single-echo fMRI.

    In contrast, multi-echo (ME) fMRI refers to collecting data at multiple echo times,
    resulting in multiple volumes with varying levels of contrast acquired per RF pulse.
    Multi-echo fMRI can be used to identify and remove certain types of noise that can't
    be removed from single-echo data.

To understand what multi-echo fMRI is and why it's useful,
we will walk through a number of things.

To start, let's look at the most common type of fMRI, single-echo EPI.

What is single-echo fMRI?
-------------------------

In a typical single-echo EPI sequence,
the protons in a brain slice are aligned perpendicularly to the main magnetic field of the scanner (:math:`B_0`) with an "excitation pulse".
Those protons start releasing energy from the excitation pulse as they fall back in line with :math:`B_0`,
and that energy is actively measured at an "echo time" (also known as `TE`_).

When you have a single echo time, you acquire one value for each voxel, at each time point.

.. image:: /_static/mr_schematic_single_echo.png

This works rather well, in that there is relatively high BOLD contrast as every voxel,
although the level of contrast (and thus the signal-to-noise ratio) will vary across the brain.
However, this

.. _multi-echo physics:

The physics of multi-echo fMRI
------------------------------
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

.. image:: /_static/physics_signal_decay.png

.. image:: /_static/physics_multiple_echos.png

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


Why use multi-echo?
-------------------
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
