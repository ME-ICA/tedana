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

Non-BOLD like components are changes in the static signal. A common example is
movement, in which the voxel (which is at a static location within the scanner)
now contains different tissue or even an area outside of the brain. These changes 
in signal, termed :math:`{\Delta}{S_0}` 

.. image:: /_static/physics_kundu_2017_TE_dependence.jpg


.. image:: /_static/physics_kundu_2017_multiple_echoes.jpg

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

