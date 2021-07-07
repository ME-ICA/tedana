##################
The tedana roadmap
##################


**************
Project vision
**************

``tedana`` was originally developed as a place for the multi-echo fMRI
denoising method that was originally defined in ME-ICA
(`ME-ICA source <https://github.com/ME-ICA/me-ica>`_).
tedana was designed to be more understandable, modular, and adaptable so
that it can serve as a testing ground for novel multi-echo fMRI denoising
methods.
We have expanded to welcome additional multi-echo fMRI processing
approaches, and to support communal resources for multi-echo fMRI, whether
or not they directly involve the tedana software.


***************
Scope of tedana
***************

tedana is a collection of tools, software and a community related to echo time
(TE) dependent analyses. The umbrella of tedana covers a number of overlapping,
but somewhat distinct, ideas related to multi-echo analysis. This scope includes
collecting multi-echo data (Acquisition), combining those echoes together
(Combination), with optional noise removal (Denoising), inspecting the outputs
(Visualization) and answering multi-echo related questions (Community). In
general, tedana accepts previously preprocessed data to produce outputs that
are ready for further analyses.


Acquisition
===========

While the development of multi-echo sequences is beyond the current scope
of tedana, the tedana community is committed to providing guidelines on current
multi-echo implementations. This will include both specific instructions for
how to collect multi-echo data for multiple vendors as well as details about
what types of data have been collected thus far. These details are subject to
change, and are intended to provide users with an idea of what is possible,
rather than definitive recommendations.

Our focus is on functional MRI, including both magnitude and phase data,
however we understand that quantitative mapping has the potential to aid in
data processing. Thus, we believe that some details on non-functional MRI
acquisitions, such as detailed T2* mapping, may fall within the scope of
tedana.
Acquisition related details can be found in the `tedana Documentation.`_

.. _tedana Documentation.: https://tedana.readthedocs.io/en/latest/acquisition.html


Combining echoes
================

An early step in processing data collected with multiple echoes is the
combination of the data into a single time series. We currently implement
multiple options to combine multi-echo data and will add more as they continue
to be developed. This is an area of active development and interest.


Denoising
=========

tedana was developed out of a package known as `multi-echo ICA, ME-ICA, or MEICA`_
developed by Dr. Prantik Kundu. Though the usage of ICA for classification of
signal vs noise components has continued in tedana, this is not a rule. The
tedana community is open and encouraging of new denoising methods, whether or not they
have a basis in ICA.

Specifically, we are interested in any method that seeks to use the information from multiple
echoes to identify signal (defined here as BOLD signals arising from neural
processing) and noise (defined here as changes unrelated to neural
processing, such as motion, cardiac, respiration).

tedana is primarily intended to work on volume data, that is, data that is
still in structured voxel space. This is because several of the currently used denoising metrics rely on spatial continuity, and they have not yet been updated to consider continuity over cortical vertices.
Therefore, surface-based denoising is not currently
within the scope of tedana, but code could be written so that it is a
possible option in the future.

Currently tedana works on a single subject, run by run basis; however, methods
that use information across multiple runs are welcome.

.. _`multi-echo ICA, ME-ICA, or MEICA`: https://github.com/ME-ICA/me-ica


Visualization
=============

As part of the processing stream, tedana provides figures and an
HTML-based report for inspecting results. These are intended to help
users understand the outputs from tedana and diagnose problems. Though a
comprehensive viewer (such as fsleyes) is outside of the scope of tedana, we
will continue to improve the reports and add new information as needed.


Community
=========

tedana is intended to be a community of multi-echo users. The primary resource
is the github repository and related documentation. In addition, the tedana
group will attempt to answer multi-echo related questions on NeuroStars
(`multi-echo tag <https://neurostars.org/tag/multi-echo>`_ or
`tedana tag <https://neurostars.org/tag/tedana>`_).


What tedana isn’t
=================

While the list of things that do not fall under the scope of tedana are
infinite, it is worth mentioning a few points:

- tedana will not offer a GUI for usage
- it is intended to be either a stand
  alone processing package or serve as a processing step as part of a larger
  package (i.e. fmriprep or afni_proc.py).
- tedana will not provide basic preprocessing steps, such as motion correction
  or slice timing correction. While these were previously part of the ME-ICA
  pipeline, the sheer variety of possible choices, guidelines and data types
  precludes including it within the tedana package.
- tedana will not provide statistical analyses in the form of general linear models,
  connectivity or decoding. Though multi-echo data is amenable to all methods
  of analysis, these methods will not be included in the tedana package.


***********************************************
Metrics of success and corresponding milestones
***********************************************

We will know that we have been successful in creating ``tedana`` when we have succeeded in providing
several concrete deliverables, which can be broadly categorized into:

1. :ref:`Documentation`,
2. :ref:`Transparent and Reproducible Processing`,
3. :ref:`Testing`,
4. :ref:`Workflow Integrations`,
5. :ref:`Extensions and Improvements to ME-EPI processing`, and
6. :ref:`Developing a healthy community`

Each deliverable has been synthesized into a milestone that gives the ``tedana`` community a link
between the issues and the high level vision for the project.


.. _Documentation:

Documentation
=============

**Summary**:
One long-standing concern with ME-EPI denoising has been the availability of
documentation for the method outside of published scientific papers.
To address this, we have created `a ReadTheDocs site`_;
however, there are still several sections either explicitly marked as "#TODO"
or otherwise missing crucial information.

We are committed to providing helpful documentation for all users of ``tedana``.
One metric of success, then, is to develop documentation that includes:

1. Motivations for conducting echo time dependent analysis,
2. A collection of key ME-EPI references and acqusition sequences
   from the published literature,
3. Tutorials on how to use ``tedana``,
4. The different processing steps that are conducted in each workflow,
5. An up-to-date description of the API,
6. A transparent explanation of the different decisions that are made
   through the ``tedana`` pipeline, and
7. Where to seek support

.. _a ReadTheDocs site: https://tedana.readthedocs.io

|milestone1|_

.. _milestone1: https://github.com/ME-ICA/tedana/milestone/6

.. |milestone1| replace:: **Associated Milestone**

This milestone will close when the online documentation contains the minimum necessary information
to orient a complete newcomer to ME-EPI, both on the theoretical basis of the method as well as
the practical steps used in ME-EPI denoising.


.. _Transparent and Reproducible Processing:

Transparent and reproducible processing
=======================================

**Summary**:
Alongside the lack of existing documentation,
there is a general unfamiliarity with how selection criteria are applied to individual data sets.
This lack of transparency, combined with the non-deterministic nature of the decomposition,
has generated significant uncertainty when interpreting results.

In order to build and maintain confidence in ME-EPI processing,
any analysis software---including ``tedana``---must provide enough information such that
the user is empowered to conduct transparent and reproducible analyses.
This will permit clear reporting of the ME-EPI results in published studies
and facilitate a broader conversation in the scientific community on the nature of ME-EPI processing.

We are therefore committed to making ``tedana`` analysis transparent and reproducible
such that we report back all processing steps applied to any individual data set,
including the specific selection criteria used in making denoising decisions.
This, combined with the reproducibility afforded by seeding all non-deterministic steps,
will enable both increased confidence and better reporting of ME-EPI results.

A metric of success for ``tedana`` then, should be enhancements to the code such that:

1. Non-deterministic steps are made reproducible by enabling access to a "seed value", and
2. The decision process for individual component data is made accessible to the end user.

|milestone2|_

.. _milestone2: https://github.com/ME-ICA/tedana/milestone/4

.. |milestone2| replace:: **Associated Milestone**

This milestone will close when when the internal decision making process for
component selection is made accessible to the end user,
and an analysis can be reproduced by an independent researcher who has access to the same data.


.. _Testing:

Testing
=======

**Summary**:
Historically, the lack of testing for ME-EPI analysis pipelines has prevented new
developers from engaging with the code for fear of silently breaking or otherwise degrading
the existing implementation.
Moving forward, we want to grow an active development community,
where developers feel empowered to explore new enhancements to the ``tedana`` code base.

One means to ensure that new code does not introduce bugs is through extensive testing.
We are therefore committed to implementing high test coverage at both
the unit test and integration test levels;
that is, both in testing individual functions and broader workflows, respectively.

A metric of success should thus be:

1. Achieving 90% test coverage for unit tests, as well as
2. Three distinguishable integration tests over a range of possible acquisition conditions.

|milestone3|_

.. _milestone3: https://github.com/ME-ICA/tedana/milestone/7

.. |milestone3| replace:: **Associated Milestone**

This milestone will close when we have 90% test coverage for unit tests and
three distinguishable integration tests,
varying number of echos and acquisition type (i.e., task vs. rest).


.. _Workflow Integrations:

Workflow integration: AFNI
==========================

**Summary**:
Currently, `afni_proc.py`_ distributes an older version of ``tedana``,
around which they have built a wrapper script, `tedana_wrapper.py`_, to ensure compatibility.
AFNI users at this point are therefore not accessing the latest version of ``tedana``.
We will grow our user base if ``tedana`` can be accessed through AFNI,
and we are therefore committed to supporting native integration of ``tedana`` in AFNI.

.. _afni_proc.py: https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html
.. _tedana_wrapper.py: https://github.com/afni/afni/blob/a3288abefb66bc7c76e98fdf13425ab48651bf36/src/python_scripts/afni_python/tedana_wrapper.py

One metric of success, therefore, will be if we can demonstrate sufficient stability and support
such that the ``afni_proc.py`` maintainers are willing to switch to ``tedana`` as the recommended
method of accessing ME-EPI denoising in AFNI.
We will aim to aid in this process by increasing compatibility between ``tedana``
and the ``afni_proc.py`` workflow, eliminating the need for an additional wrapper script.

|milestone4|_

.. _milestone4: https://github.com/ME-ICA/tedana/milestone/8

.. |milestone4| replace:: **Associated Milestone**

This milestone will close when ``tedana`` is stable enough such that the recommended default in
``afni_proc.py`` is to access ME-EPI denoising via ``pip install tedana``,
rather than maintaining the alternative version that is currently used.


Workflow integration: BIDS
==========================

**Summary**:
Currently, the BIDS ecosystem has limited support for ME-EPI processing.
We will grow our user base if ``tedana`` is integrated into existing BIDS Apps and
therefore accessible to members of the BIDS community.
One promising opportunity is if ``tedana`` can be used natively in `FMRIPrep`_.
Some of the work is not required at this repository, but other changes will need to happen here;
for example, making sure the outputs are BIDS compliant.

A metric of success, then, will be:

1. Fully integrating ``tedana`` into ``FMRIPrep``, and
2. Making ``tedana`` outputs compliant with the `BIDS derivatives specification`_.

.. _FMRIPrep: https://github.com/poldracklab/fmriprep
.. _BIDS derivatives specification: https://docs.google.com/document/d/1Wwc4A6Mow4ZPPszDIWfCUCRNstn7d_zzaWPcfcHmgI4/edit

|milestone5|_

.. _milestone5: https://github.com/ME-ICA/tedana/milestone/9

.. |milestone5| replace:: **Associated Milestone**

This milestone will close when the denoising steps of ``tedana`` are stable enough
to integrate into ``FMRIPrep`` and the ``FMRIPrep`` project is updated to process ME-EPI scans.


.. _Extensions and Improvements to ME-EPI processing:

Method extensions & improvements
================================

**Summary**:
Overall, each of the listed deliverables will support a broader goal:
to improve on ME-EPI processing itself.
This is an important research question and will advance the state-of-the-art in ME-EPI processing.

A metric of success here would be
* *EITHER* integrating a new decomposition method, beyond ICA
* *OR* validating new selection criteria.

To achieve either of these metrics, it is likely that we will need to incoporate a
quality-assurance module into ``tedana``, possibly as visual reports.

|milestone6|_

.. _milestone6: https://github.com/ME-ICA/tedana/milestone/10

.. |milestone6| replace:: **Associated Milestone**

This milestone will close when the codebase is stable enough to integrate novel methods
into ``tedana``, and that happens!


.. _Developing a healthy community:

Developing a healthy community
==============================

**Summary**:
In developing ``tedana``, we are committed to  fostering a healthy community.
A healthy community is one in which the maintainers are happy and not overworked,
and which empowers users to contribute back to the project.
By making ``tedana`` stable and well-documented, with enough modularity to integrate improvements,
we will enable new contributors to feel that their work is welcomed.

We therefore have one additional metric of success:

1. An outside contributor integrates an improvement to ME-EPI denoising.

|milestone7|_

.. _milestone7: https://github.com/ME-ICA/tedana/milestone/5

.. |milestone7| replace:: **Associated Milestone**

This milestone will probably never close,
but will serve to track issues related to building and supporting the ``tedana`` community.
