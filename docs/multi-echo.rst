Multi-echo fMRI
===============
In most echo-planar image (EPI) fMRI sequences, 
only one brain image is acquired at each repetition time, at the rate of radio frequency (RF).
In contrast, in multi-echo (ME) fMRI, data are acquired for multiple echo times, 
resulting in multiple volumes with varying levels of contrast acquired per RF.

The physics of multi-echo fMRI
------------------------------
Multi-echo fMRI data is obtained by acquiring multiple TEs (commonly called
`echo times`_) for each MRI volume
during data collection.
While fMRI signal contains important neural information (termed the blood
oxygen-level dependent, or `BOLD signal`_,
it also contains "noise" (termed non-BOLD signal) caused by things like
participant motion and changes in breathing.
Because the BOLD signal is known to decay at a set rate, collecting multiple
echos allows us to assess whether components of the fMRI signal are BOLD- or
non-BOLD.
For a comprehensive review, see `Kundu et al. (2017)`_.

.. _echo times: http://mriquestions.com/tr-and-te.html
.. _BOLD signal: http://www.fil.ion.ucl.ac.uk/spm/course/slides10-zurich/Kerstin_BOLD.pdf
.. _Kundu et al. (2017): https://paperpile.com/shared/eH3PPu

Why use multi-echo?
-------------------
There are many potential reasons an investigator would be interested in using multi-echo EPI (ME-EPI).
Among these are the different levels of analysis ME-EPI enables.
Specifically, by collecting multi-echo data, researchers are able to compare results for
(1) single-echo, (2) optimally combined, and (3) denoised data.
Each of these levels of analysis have their own advantages.

For single-echo: currently, field standards are largely set using single-echo EPI.
Because multi-echo is composed of multiple single-echo time series, each of these can be analyzed separately.
This allows researchers to benchmark their results.

For optimally combined: Rather than analyzing single-echo time series separately,
we can combine them into a "optimally combined time series".
For more information on this combination, see `processing pipeline details`_.
Optimally combined data exhibits higher SNR and improves statistical power of analyses in regions
traditionally affected by drop-out.

For denoised: Collecting multi-echo data allows access to unique denoising metrics.
``tedana`` is one ICA-based denoising pipeline built on this information.
Other ICA-based denoising methods like ICA-AROMA (`Pruim et al. (2015)`_)
have been shown to significantly improve the quality of cleaned signal.

These methods, however, have comparably limited information, as they are designed to work with single-echo EPI.
Collecting multi-echo EPI allows us to leverage all of the information available for single-echo datasets,
as well as additional information only available when looking at signal decay across multiple TEs.
We can use this information to denoise the optimally combined time series.

.. _processing pipeline details: https://tedana.readthedocs.io/en/latest/approach.html#optimal-combination
.. _Pruim et al. (2015): https://www.sciencedirect.com/science/article/pii/S1053811915001822

Acquisition Parameter Recommendations
-------------------------------------
There is no empirically tested best parameter set for multi-echo acquisition.
The guidelines for optimizing parameters are similar to single-echo fMRI.
For multi-echo fMRI, the same factors that may guide priorities for single echo
fMRI sequences are also relevant.
Choose sequence parameters that meet the priorities of a study with regards to spatial resolution,
spatial coverage, sample rate, signal-to-noise ratio, signal drop-out, distortion, and artifacts.

The one difference with multi-echo is a slight time cost.
For multi-echo fMRI, the shortest echo time (TE) is essentially free since it is collected in the
gap between the RF pulse and the single-echo acquisition.
The second echo tends to roughly match the single-echo TE.
Additional echoes require more time.
For example, on a 3T MRI, if the T2* weighted TE is 30ms for single echo fMRI,
a multi-echo sequence may have TEs of 15.4, 29.7, and 44.0ms.
In this example, the extra 14ms of acquisition time per RF pulse is the cost of multi-echo fMRI.

One way to think about this cost is in comparison to single-echo fMRI.
If a multi-echo sequence has identical spatial resolution and acceleration as a single-echo sequence,
then a rough rule of thumb is that the multi-echo sequence will have 10% fewer slices or 10% longer TR.
Instead of compromising on slice coverage or TR, one can increase acceleration.
If one increases acceleration, it is worth doing an empirical comparison to make sure there
isn't a non-trivial loss in SNR or an increase of artifacts.

A minimum of 3 echoes is recommended for running TE-dependent denoising.
While there are successful studies that don’t follow this rule,
it may be useful to have at least one echo that is earlier and one echo that is later than the
TE one would use for single-echo T2* weighted fMRI.

More than 3 echoes may be useful, because that would allow for more accurate
estimates of BOLD and non-BOLD weighted fluctuations, but more echoes have an
additional time cost, which would result in either less spatiotemporal coverage
or more acceleration.
Where the benefits of more echoes balance out the additional costs is an open research question.

We are not recommending specific parameter options at this time.
There are multiple ways to balance the slight time cost from the added echoes that have
resulted in research publications.
We suggest new multi-echo fMRI users examine the `spreadsheet`_ of journal articles that use
multi-echo fMRI to identify studies with similar acquisition priorities,
and use the parameters from those studies as a starting point.

.. _spreadsheet: https://docs.google.com/spreadsheets/d/1WERojJyxFoqcg_tndUm5Kj0H1UfUc9Ban0jFGGfPaBk/edit#gid=0

Resources
---------

Journal articles
****************
* A `review`_ on multi-echo fMRI and its applications
* A `spreadsheet`_ cataloguing papers using multi-echo fMRI, with information about acquisition parameters.

.. _review: https://www.ncbi.nlm.nih.gov/pubmed/28363836
.. _spreadsheet: https://docs.google.com/spreadsheets/d/1WERojJyxFoqcg_tndUm5Kj0H1UfUc9Ban0jFGGfPaBk/edit#gid=0

Videos
******
* An `educational session from OHBM 2017`_ by Dr. Prantik Kundu about multi-echo denoising
* A `series of lectures from the OHBM 2017 multi-echo session`_ on multiple facets of multi-echo data analysis

.. _educational session from OHBM 2017: https://www.pathlms.com/ohbm/courses/5158/sections/7788/video_presentations/75977
.. _series of lectures from the OHBM 2017 multi-echo session: https://www.pathlms.com/ohbm/courses/5158/sections/7822


Datasets
********
A number of multi-echo datasets have been made public so far.
This list is not necessarily up-to-date, so please check out OpenNeuro to potentially find more.

* `Multi-echo fMRI replication sample of autobiographical memory, prospection and theory of mind reasoning tasks`_
* `Multi-echo Cambridge`_
* `Multiband multi-echo imaging of simultaneous oxygenation and flow timeseries for resting state connectivity`_
* `Valence processing differs across stimulus modalities`_
* `Cambridge Centre for Ageing Neuroscience (Cam-CAN)`_

.. _Multi-echo fMRI replication sample of autobiographical memory, prospection and theory of mind reasoning tasks: https://openneuro.org/datasets/ds000210/
.. _Multi-echo Cambridge: https://openneuro.org/datasets/ds000258
.. _Multiband multi-echo imaging of simultaneous oxygenation and flow timeseries for resting state connectivity: https://openneuro.org/datasets/ds000254
.. _Valence processing differs across stimulus modalities: https://openneuro.org/datasets/ds001491
.. _Cambridge Centre for Ageing Neuroscience (Cam-CAN): https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/
