Resources
=========

Journal articles describing multi-echo methods
----------------------------------------------
* | :ref:`spreadsheet of publications` catalogues papers using multi-echo fMRI,
  | with information about acquisition parameters.
* | `Multi-echo acquisition`_
  | Posse, NeuroImage 2012
  | Includes an historical overview of multi-echo acquisition and research
* | `Multi-Echo fMRI A Review of Applications in fMRI Denoising and Analysis of BOLD Signals`_
  | Kundu et al, NeuroImage 2017
  | A review of multi-echo denoising with a focus on the MEICA algorithm
* | `Enhanced identification of BOLD-like components with MESMS and MEICA`_
  | Olafsson et al, NeuroImage 2015
  | The appendix includes a good explanation of the math underlying MEICA denoising
* | `Comparing resting state fMRI de-noising approaches using multi- and single-echo acquisitions`_
  | Dipasquale et al, PLoS One 2017
  | The appendix includes some recommendations for multi-echo acquisition

.. _Multi-echo acquisition: https://www.ncbi.nlm.nih.gov/pubmed/22056458
.. _Multi-Echo fMRI A Review of Applications in fMRI Denoising and Analysis of BOLD Signals: https://www.ncbi.nlm.nih.gov/pubmed/28363836
.. _Enhanced identification of BOLD-like components with MESMS and MEICA: https://www.ncbi.nlm.nih.gov/pubmed/25743045
.. _Comparing resting state fMRI de-noising approaches using multi- and single-echo acquisitions: https://www.ncbi.nlm.nih.gov/pubmed/28323821

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

Multi-echo preprocessing software
---------------------------------
tedana requires data that has already been preprocessed for head motion, alignment, etc.

AFNI can process multi-echo data natively as well as apply tedana denoising through the use of
**afni_proc.py**. To see various implementations, start with Example 12 in the `afni_proc.py help`_

.. _afni_proc.py help: https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html

`fmriprep` can also process multi-echo data, but is currently limited to using the optimally combined
timeseries.
For more details, see the `fmriprep workflows page`_.

.. _fmriprep workflows page: https://fmriprep.readthedocs.io/en/stable/workflows.html

Currently SPM and FSL do not natively support multi-echo fmri data processing.

Other software that uses multi-echo fMRI
----------------------------------------
``tedana`` represents only one approach to processing multi-echo data.
Currently there are a number of methods that can take advantage of or use the
information contained in multi-echo data.
These include:

* | `3dMEPFM`_: A multi-echo implementation of 'paradigm free mapping', that is
  | detection of neural events in the absence of a prespecified model. By
  | leveraging the information present in multi-echo data, changes in relaxation
  | time can be directly estimated and more events can be detected.
  | For more information, see the `following paper`_.
* | `Bayesian approach to denoising`_: An alternative approach to separating out
  | BOLD and non-BOLD signals within a Bayesian framework is currently under
  | development.
* | `Multi-echo Group ICA`_: Current approaches to ICA just use a single run of
  | data in order to perform denoising. An alternative approach is to use
  | information from multiple subjects or multiple runs from a single subject
  | in order to improve the classification of BOLD and non-BOLD components.
* | `Dual Echo Denoising`_: If the first echo can be collected early enough,
  | there are currently methods that take advantage of the very limited BOLD
  | weighting at these early echo times.
* | `qMRLab`_: This is a MATLAB software package for quantitative magnetic
  | resonance imaging. While it does not support ME-fMRI, it does include methods
  | for estimating T2*/S0 from high-resolution, complex-valued multi-echo GRE
  | data with correction for background field gradients.

.. _3dMEPFM: https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dMEPFM.html
.. _following paper: https://www.sciencedirect.com/science/article/pii/S105381191930669X
.. _Bayesian approach to denoising: https://ww5.aievolution.com/hbm1901/index.cfm?do=abs.viewAbs&abs=5026
.. _Multi-echo Group ICA: https://ww5.aievolution.com/hbm1901/index.cfm?do=abs.viewAbs&abs=1286
.. _Dual Echo Denoising: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3518782/
.. _qMRLab: https://github.com/qMRLab/qMRLab

Datasets
--------
A number of multi-echo datasets have been made public so far.
This list is not necessarily up to date, so please check out OpenNeuro to potentially find more.

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

.. _spreadsheet of publications:

Publications using multi-echo fMRI
----------------------------------
You can view and suggest additions to this spreadsheet `here`_
This is a volunteer-led effort so, if you know of a excluded publication, whether or not it is yours,
please add it.

.. raw:: html

    <iframe style="position: absolute; height: 60%; width: 60%; border: none" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vS0nEVp27NpwdzPunvMLflyKzcZbCo4k2qPk5zxEiaoJTD_IY1OGbWICizogAEZlTyL7d_7aDA92uwf/pubhtml?widget=true&amp;headers=false"></iframe>

.. _here: https://docs.google.com/spreadsheets/d/1WERojJyxFoqcg_tndUm5Kj0H1UfUc9Ban0jFGGfPaBk/edit#gid=0
