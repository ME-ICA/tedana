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

.. _review: https://www.ncbi.nlm.nih.gov/pubmed/28363836

Videos
******
* An `educational session from OHBM 2017`_ by Dr. Prantik Kundu about multi-echo denoising
* A `series of lectures from the OHBM 2017 multi-echo session`_ on multiple facets of multi-echo data analysis

.. _educational session from OHBM 2017: https://www.pathlms.com/ohbm/courses/5158/sections/7788/video_presentations/75977
.. _series of lectures from the OHBM 2017 multi-echo session: https://www.pathlms.com/ohbm/courses/5158/sections/7822

Sequences
*********
* Multi-echo sequences: who has them and how to get them.

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
