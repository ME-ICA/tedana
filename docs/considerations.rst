##########################
Considerations for ME-fMRI
##########################
Multi-echo fMRI acquisition sequences and analysis methods are rapidly maturing. Someone who has access
to a multi-echo fMRI sequence should seriously consider using it. 

The possible costs and benefits of multi-echo fMRI
==================================================
The following are a few points to consider when deciding whether or not to collect multi-echo data.

Possible increase in TR
-----------------------
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

Weighted Averaging may lead to an increase in SNR 
-------------------------------------------------
Multiple studies have shown that a
weighted average of the echoes to optimize T2* weighting, sometimes called "optimally combined,"
gives a reliable, modest boost in data quality. The optimal combination of echoes can currently be
calculated in several software packages including AFNI, fMRIPrep, and tedana. In tedana, the weighted
average can be calculated with `t2smap`_ If no other
acquisition compromises are necessary to acquire multi-echo data, this boost is worthwhile. 

Consider the life of the dataset
--------------------------------
If other
compromises are necessary, consider the life of the data set. If data is being acquired for a discrete
study that will be acquired, analyzed, and published in a year or two, it might not be worth making
compromises to acquire multi-echo data. If a data set is expected to be used for future analyses in later
years, it is likely that more powerful approaches to multi-echo denoising will sufficiently mature and add
even more value to a data set.

Other multi-echo denoising methods, such as MEICA, the predecessor to tedana, have shown the potential for
much greater data quality improvements, as well as the ability to more accurately separate visually similar
signal vs noise, such as scanner based drifts vs slow changes in BOLD signal. These more powerful methods are
still being improved, and the algorithms are still changing. Users need to have the time and knowledge to look
at the denoising output from every run to make sure denoising worked as intended. 

You may recover signal in areas affected by dropout
---------------------------------------------------
Typical single echo fMRI uses an echo time that is appropriate for signal across most of the brain. While this is effective
it also leads to drop out in regions with low :math:T_2^* values. This can lead to low or even no signal at all in some areas. 
If your research question could benefit from having either
improve signal characteristics in regions such as the orbitofrontal cortex, ventral temporal cortex or 
the ventral striatum them multi-echo fMRI may be beneficial. 

Consider the cost of added quality control
------------------------------------------
If someone wants a push-button
way to use multi-echo data to improve data quality, that doesn't require as deep an inspection of every output,
stick with using the weighted average. The developers of tedana look forward to when tedana and other methods
have sufficiently stable algorithms, which have been validated on a wide range of data sets, so that we can
recommend the wide use of tedana.



.. _t2smap: https://tedana.readthedocs.io/en/latest/usage.html#run-t2smap

Acquisition Parameter Recommendations
=====================================
There is no empirically tested best parameter set for multi-echo acquisition.
The guidelines for optimizing parameters are similar to single-echo fMRI.
For multi-echo fMRI, the same factors that may guide priorities for single echo
fMRI sequences are also relevant.
Choose sequence parameters that meet the priorities of a study with regards to spatial resolution,
spatial coverage, sample rate, signal-to-noise ratio, signal drop-out, distortion, and artifacts.

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
We suggest new multi-echo fMRI users examine the :ref:`spreadsheet of publications` that use
multi-echo fMRI to identify studies with similar acquisition priorities,
and use the parameters from those studies as a starting point. More complete recomendations
and guidelines are discussed in the `appendix`_ of Dipasquale et al, 2017.

.. _appendix: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173289

.. _found here: https://www.cmrr.umn.edu/multiband/
.. _this link: http://license.umn.edu/technologies/cmrr_center-for-magnetic-resonance-research-software-for-siemens-mri-scanners
.. _available here: https://www.nmr.mgh.harvard.edu/software/c2p/sms
.. _GE Collaboration Portal: https://collaborate.mr.gehealthcare.com
.. note:: 
    In order to increase the number of contrasts ("echoes") you may need to first increase the TR, shorten the 
    first TE and/or enable in-plane acceleration. For typically used parameters see the 
    `parameters and publications page`_
.. _parameters and publications page: https://tedana.readthedocs.io/en/latest/publications.html

Resources
=========

Journal articles
----------------
* | :ref:`spreadsheet of publications` catalogues papers using multi-echo fMRI,
    with information about acquisition parameters.
* | `Multi-echo acquisition`_
  | Posse, NeuroImage 2012
  | Includes an historical overview of multi-echo acquisition and research
* | `Multi-Echo fMRI A Review of Applications in fMRI Denoising and Analysis of BOLD Signals`_
  |  Kundu et al, NeuroImage 2017
  |  A review of multi-echo denoising with a focus on the MEICA algorithm
* | `Enhanced identification of BOLD-like componenents with MESMS and MEICA`_
  |  Olafsson et al, NeuroImage 2015
  |  The appendix includes a good explanation of the math underlying MEICA denoising
* | `Comparing resting state fMRI de-noising approaches using multi- and single-echo acqusitions`_
  |  Dipasquale et al, PLoS One 2017
  |  The appendix includes some recommendations for multi-echo acqusition

.. _Multi-echo acquisition: https://www.ncbi.nlm.nih.gov/pubmed/22056458
.. _Multi-Echo fMRI A Review of Applications in fMRI Denoising and Analysis of BOLD Signals: https://www.ncbi.nlm.nih.gov/pubmed/28363836
.. _Enhanced identification of BOLD-like componenents with MESMS and MEICA: https://www.ncbi.nlm.nih.gov/pubmed/25743045
.. _Comparing resting state fMRI de-noising approaches using multi- and single-echo acqusitions: https://www.ncbi.nlm.nih.gov/pubmed/28323821

Videos
------
* An `educational session from OHBM 2017`_ by Dr. Prantik Kundu about multi-echo denoising
* A `series of lectures from the OHBM 2017 multi-echo session`_ on multiple facets of multi-echo data analysis
* | Multi-echo fMRI lecture from the `2018 NIH FMRI Summer Course`_ by Javier Gonzalez-Castillo
  | `Slides from 2018 NIH FMRI Summer Course`_

.. _educational session from OHBM 2017: https://www.pathlms.com/ohbm/courses/5158/sections/7788/video_presentations/75977
.. _series of lectures from the OHBM 2017 multi-echo session: https://www.pathlms.com/ohbm/courses/5158/sections/7822
.. _2018 NIH FMRI Summer Course: https://fmrif.nimh.nih.gov/course/fmrif_course/2018/14_Javier_20180713
.. _Slides from 2018 NIH FMRI Summer Course: https://fmrif.nimh.nih.gov/COURSE/fmrif_course/2018/content/14_Javier_20180713.pdf

Available multi-echo fMRI sequences for multiple vendors
--------------------------------------------------------

**For Siemens** users, there are two options for Works In Progress (WIPs) Sequences. 
The Center for Magnetic Resonance Research at the University of Minnesota 
provides a custom MR sequence that allows users to collect multiple echoes 
(termed **Contrasts**). The sequence and documentation can be `found here`_. For details 
on obtaining a license follow `this link`_. By default the number of contrasts is 1, 
yielding a signal echo sequence. In order to collect multiple echoes, increase number of 
Contrasts on the **Sequence Tab, Part 1** on the MR console. 

In addition, the Martinos Center at Harvard also has a MR sequence available, with the 
details `available here`_. The number of echoes can be specified on the **Sequence, Special** tab 
in this sequence. 

**For GE users**, there are currently two sharable pulse sequences:

Multi-echo EPI (MEPI) – Software releases: DV24, MP24 and DV25 (with offline recon)
Hyperband Multi-echo EPI (HyperMEPI) - Software releases: DV26, MP26, DV27, RX27 
(here Hyperband can be deactivated to do simple Multi-echo EPI – online recon)

Please reach out to the GE Research Operation team or each pulse sequence’s 
author to begin the process of obtaining this software. More information can be 
found on the `GE Collaboration Portal`_ 

Once logged-in, go to Groups > GE Works-in-Progress you can find the description of the current ATSM (i.e. prototypes)

Multi-echo preprocessing software
---------------------------------

tedana requires data that has already been preprocessed for head motion, alignment, etc.

AFNI can process multi-echo data natively as well as apply tedana denoising through the use of 
**afni_proc.py**. To see various implementations, start with Example 12 in the `afni_proc.py help`_

.. _afni_proc.py help: https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html

`fmriprep` can also process multi-echo data, but is currently limited to using the optimally combined
timeseries. For more details, see the `fmriprep workflows page`_

.. _fmriprep workflows page: https://fmriprep.readthedocs.io/en/stable/workflows.html

Currently SPM and FSL do not natively support mutli-echo fmri data processing. 

Other software that uses multi-echo fMRI
----------------------------------------

Information and links to other approaches for denoising multi-echo fMRI data will be added here.

Datasets
--------
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
