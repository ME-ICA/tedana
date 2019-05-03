Multi-echo fMRI
===============
In multi-echo (ME) fMRI, data are acquired for multiple echo times, resulting in
multiple time series for each voxel.

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
ME-EPI exhibits higher SNR and improves statistical power of analyses.

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

Sequences
*********
* Multi-echo sequences: who has them and how to get them.

Acquisition Parameter Recommendations
************************************
There is no empirically tested best parameter set for multi-echo acquisition. The guidelines for optimizing parameters are similar to single-echo fMRI. Parameter choices require balancing the needs of specific studies with regards to spatial resolution and coverage, sampling rate, signal-to-noise ratio, signal drop-out, distortion, and artifacts. Optimize these parameters based on whatever a study's priorities are. The compromise for transitioning to multi-echo is either losing a few slices, increasing the TR, or increasing within-slice or multi-slice acceleration. A rough rule of thumb is that collecting 3 echoes with a similar parameter set to single-echo will involve a loss of 10% of slices (36 instead of 40 slices), a 10% increase in TR (2.2sec instead of 2s) or an increase in acceleration. If one increases acceleration to avoid compromises in spatial coverage or time, it is worth doing an empirical comparison to make sure there isn't a non-trivial loss in SNR or an increase of artifacts.

A minimum of 3 echoes is recommended for running TE-dependent denoising. More echoes may be useful, because that would allow for more accurate estimates of BOLD and non-BOLD weighted fluctuations, but more echoes would require additional compromises in spatiotemporal coverage or more acceleration. Whether the benefits of more echoes balance out the additional acquisition compromises is an open research question. While there are successful studies that don’t follow this rule, it may be useful to have at least one echo that is earlier and one echo that is later than TE one would use for single-echo T2* weighted fMRI. 

The `spreadsheet`_ spreadsheet of journal articles that use multi-echo fMRI is a useful place to find acqusition parameters.

.. _spreadsheet: https://docs.google.com/spreadsheets/d/1WERojJyxFoqcg_tndUm5Kj0H1UfUc9Ban0jFGGfPaBk/edit#gid=0

Datasets
********
A small number of multi-echo datasets have been made public so far. This list is
not necessarily up-to-date, so please check out OpenNeuro to potentially
find more.

* `Multi-echo fMRI replication sample of autobiographical memory, prospection and theory of mind reasoning tasks`_
* `Multi-echo Cambridge`_
* `Multiband multi-echo imaging of simultaneous oxygenation and flow timeseries for resting state connectivity`_
* `Valence processing differs across stimulus modalities`_

.. _Multi-echo fMRI replication sample of autobiographical memory, prospection and theory of mind reasoning tasks: https://openneuro.org/datasets/ds000210/
.. _Multi-echo Cambridge: https://openneuro.org/datasets/ds000258
.. _Multiband multi-echo imaging of simultaneous oxygenation and flow timeseries for resting state connectivity: https://openneuro.org/datasets/ds000254
.. _Valence processing differs across stimulus modalities: https://openneuro.org/datasets/ds001491
