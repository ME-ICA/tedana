
###
FAQ
###


.. _collecting fMRIPrepped data:

***************************************************
[tedana] How do I use tedana with fMRIPrepped data?
***************************************************

fMRIPrep versions >= 21.0.0
===========================

Starting with version 21.0.0, `fMRIPrep`_ added the ``--me-output-echos`` argument,
which outputs individual echoes after slice timing, motion correction, and distortion correction have been performed.
These preprocessed echoes can be denoised with tedana,
after which warps written out by fMRIPrep can be applied to transform the denoised data to standard space.

As the fMRIPrep outputs become more formalized,
it is possible to write functions that can select the appropriate derivative files and run tedana on them.
Below is one example of such a function.

.. raw:: html

    <script src="https://gist.github.com/jbdenniso/73ec8281229d584721563a41aba410cf.js"></script>

fMRIPrep versions < 21.0.0
==========================

Prior to version 21.0.0, `fMRIPrep`_ outputted the preprocessed, optimally-combined fMRI data, rather than echo-wise data.
This means that you cannot use the standard fMRIPrep outputs with tedana for multi-echo denoising.

However, as long as you still have access to fMRIPrep's working directory,
you can extract the partially-preprocessed echo-wise data,
along with the necessary transform file to linearly transform that echo-wise data into the native structural scan space.
The transform from that structural scan space to standard space will already be available,
so you can easily chain these transforms to get your echo-wise data
(or, more importantly, the scanner-space multi-echo denoised data) into standard space.

Unfortunately, fMRIPrep's working directory structure is not stable across versions,
so writing code to grab the relevant files from the working directory is a bit of a moving target.
Nevertheless, we have some code (thanks to Julio Peraza) that works for version 20.2.1.

.. raw:: html

    <script src="https://gist.github.com/tsalo/83828e0c1e9009f3cbd82caed888afba.js"></script>

Warping scanner-space fMRIPrep outputs to standard space
========================================================

Here is a basic approach to normalizing scanner-space tedana-denoised data created from fMRIPrep outputs,
using ANTS's antsApplyTransforms tool.
The actual filenames of fMRIPrep derivatives depend on the filenames in the BIDS dataset
(e.g., the name of the task, run numbers, etc.),
but in this example we chose to use the simple example of "sub-01" and "task-rest".
The standard space template in this example is "MNI152NLin2009cAsym", but will depend on fMRIPrep settings in practice.

.. raw:: html

    <script src="https://gist.github.com/tsalo/f9f38e9aba901e99ddb720465bb5222b.js"></script>

************************************
[tedana] ICA has failed to converge.
************************************

The TEDICA step may fail to converge if TEDPCA is either too strict
(i.e., there are too few components) or too lenient (there are too many).

In our experience, this may happen when preprocessing has not been applied to
the data, or when improper steps have been applied to the data (e.g., distortion
correction, rescaling, nuisance regression).
If you are confident that your data have been preprocessed correctly prior to
applying tedana, and you encounter this problem, please submit a question to `NeuroStars`_.


.. _manual classification:

********************************************************************************
[tedana] I think that some BOLD ICA components have been misclassified as noise.
********************************************************************************

``tedana`` allows users to manually specify accepted components when calling the pipeline.
You can use the ``--manacc`` argument to specify the indices of components to accept.


*************************************************************************************
[tedana] Why isn't v3.2 of the component selection algorithm supported in ``tedana``?
*************************************************************************************

There is a lot of solid logic behind the updated version of the TEDICA component
selection algorithm, first added to the original ME-ICA codebase `here`_ by Dr. Prantik Kundu.
However, we (the ``tedana`` developers) have encountered certain difficulties
with this method (e.g., misclassified components) and the method itself has yet
to be validated in any papers, posters, etc., which is why we have chosen to archive
the v3.2 code, with the goal of revisiting it when ``tedana`` is more stable.

Anyone interested in using v3.2 may compile and install an earlier release (<=0.0.4) of ``tedana``.


.. _here: https://bitbucket.org/prantikk/me-ica/commits/906bd1f6db7041f88cd0efcac8a74074d673f4f5

.. _NeuroStars: https://neurostars.org
.. _fMRIPrep: https://fmriprep.readthedocs.io
.. _afni_proc.py: https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html


*******************************************************************
[ME-fMRI] Does multi-echo fMRI require more radio frequency pulses?
*******************************************************************

While multi-echo does lead to collecting more images during each TR (one per echo), there is still only a single
radiofrequency pulse per TR. This means that there is no change in the `specific absorption rate`_ (SAR) limits
for the participant.

.. _specific absorption rate: https://www.mr-tip.com/serv1.php?type=db1&dbs=Specific%20Absorption%20Rate


*********************************************************************************
[ME-fMRI] Can I combine multiband (simultaneous multislice) with multi-echo fMRI?
*********************************************************************************

Yes, these techniques are complementary.
Multiband fMRI leads to collecting multiple slices within a volume  simultaneously, while multi-echo
fMRI is instead related to collecting multiple unique volumes.
These techniques can be combined to reduce the TR in a multi-echo sequence.
