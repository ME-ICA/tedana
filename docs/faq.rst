
###
FAQ
###


.. _collecting fMRIPrepped data:

***************************************************
[tedana] How do I use tedana with fMRIPrepped data?
***************************************************

`fMRIPrep`_ outputs the preprocessed, optimally-combined fMRI data, rather than echo-wise data.
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

.. warning::
    We will try to keep the following gist up-to-date, but there is no guarantee that it will work for a given version.
    Use it with caution!

    If you do find that the gist isn't working for an fMRIPrep version >= 20.2.1,
    please comment on `Issue #717 <https://github.com/ME-ICA/tedana/issues/717>`_ (even if it's closed)
    and we will take a look at the problem.

.. raw:: html

    <script src="https://gist.github.com/tsalo/83828e0c1e9009f3cbd82caed888afba.js"></script>


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


*************************************************
[tedana] What is the warning about ``duecredit``?
*************************************************

``duecredit`` is a python package that is used, but not required by ``tedana``.
These warnings do not affect any of the processing within the ``tedana``.
To avoid this warning, you can install ``duecredit`` with ``pip install duecredit``.
For more information about ``duecredit`` and concerns about
the citation and visibility of software or methods, visit the `duecredit`_ GitHub repository.

.. _duecredit: https://github.com/duecredit/duecredit

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
