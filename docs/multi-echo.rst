What is Multi-echo fMRI
=======================
Most echo-planar image (EPI) sequences collect a single brain image following 
a radio frequency (RF) pulse, at a rate known as the repetition time (TR). 
This typical approach is known as single-echo fMRI. In contrast, multi-echo (ME) 
fMRI refers to collecting data at multiple echo times, resulting in
multiple volumes with varying levels of contrast acquired per RF pulse.

The physics of multi-echo fMRI
------------------------------
Multi-echo fMRI data is obtained by acquiring multiple echo times (commonly called
`TEs`_) for each MRI volume during data collection.
While fMRI signal contains important neural information (termed the blood
oxygen-level dependent, or `BOLD signal`_,
it also contains "noise" (termed non-BOLD signal) caused by things like
participant motion and changes in breathing.
Because the BOLD signal is known to decay at a set rate, collecting multiple
echos allows us to assess whether components of the fMRI signal are BOLD- or
non-BOLD.

The image below shows the basic relationship between echo times and the image acquired at
3T (top, A) and 7T (bottom, B). Note that the earliest echo time is the brightest, as the 
signal has only had a limited amount of time to decay. 
In addition, the latter echo times show areas in which is the signal has decayed completely ('drop out') 
due to inhomgeneity in the magnetic field. By using the information across multiple 
echoes these images can be combined in an optimal manner to take advantage of the signal 
in the earlier echoes (see `processing pipeline details`_).

.. image:: /_static/physics_kundu_2017_multiple_echoes.jpg

In order to classify the relationship between the signal and the echo time we can consider a 
single voxel at two timepoints (x and y) and the measured signal measured at three different echo times - :math:`S(TE_N)`. 

For the left column, we are observing a change that we term :math:`{\Delta}{S_0}` - that is a change
in the intercept or raw signal intensity. A common example of this is participant movement, 
in which the voxel (which is at a static location within the scanner)
now contains different tissue or even an area outside of the brain.  

As we have collected three seperate echoes, we can compare the change in signal at each echo time, :math:`{\Delta}{S(TE_n)}`. For 
:math:`{\Delta}{S_0}` we see that this produces a decaying curve. If we compare this to the original signal, as in
:math:`\frac{{\Delta}{S(TE_n)}}{S(TE_n)}` we see that there is no echo time dependence. 

In the right column, we consider changes that are related to brain activity, that is the two brain states here 
(x and y) could be a baseline and task activated state. These we term as :math:`{\Delta}{R_2^*}`. Again we can plot the
change in the signal between these two states as a function of echo time, finding that the signal rises and falls. If we compare this 
curve to the original signal we find that magnitude of the changes is dependent on the echo time.

.. image:: /_static/physics_kundu_2017_TE_dependence.jpg

For a comprehensive review, see `Kundu et al. (2017)`_.

.. _TEs: http://mriquestions.com/tr-and-te.html
.. _BOLD signal: http://www.fil.ion.ucl.ac.uk/spm/course/slides10-zurich/Kerstin_BOLD.pdf
.. _Kundu et al. (2017): https://www.sciencedirect.com/science/article/pii/S1053811917302410?via%3Dihub

Why use multi-echo?
-------------------
There are many potential reasons an investigator would be interested in using multi-echo EPI (ME-EPI).
Among these are the different levels of analysis ME-EPI enables.
Specifically, by collecting multi-echo data, researchers are able to:

**Compare results across different echoes**: currently, field standards are largely set using single-echo EPI.
Because multi-echo is composed of multiple single-echo time series, each of these can be analyzed separately 
and compared to one another. 

**Combine the results by weighted averaging**: Rather than analyzing single-echo time series separately,
we can combine them into an "optimally combined time series".
For more information on this combination, see `processing pipeline details`_.
Optimally combined data exhibits higher SNR and improves statistical power of analyses in regions
traditionally affected by drop-out.

**Denoise the data based on information contained in the echoes**: Collecting multi-echo data allows 
access to unique denoising methods. ICA-based denoising methods like ICA-AROMA (`Pruim et al. (2015)`_)
have been shown to significantly improve the quality of cleaned signal. These methods, however, have comparably 
limited information, as they are designed to work with single-echo EPI.

``tedana`` is an ICA-based denoising pipeline built especially for 
multi-echo data. Collecting multi-echo EPI allows us to leverage all of the information available for single-echo datasets,
as well as additional information only available when looking at signal decay across multiple TEs.
We can use this information to denoise the optimally combined time series.

.. _processing pipeline details: https://tedana.readthedocs.io/en/latest/approach.html#optimal-combination
.. _Pruim et al. (2015): https://www.sciencedirect.com/science/article/pii/S1053811915001822

