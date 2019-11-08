
FAQ
===

tedana
------

ICA has failed to converge.
```````````````````````````
The TEDICA step may fail to converge if TEDPCA is either too strict
(i.e., there are too few components) or too lenient (there are too many).

In our experience, this may happen when preprocessing has not been applied to
the data, or when improper steps have been applied to the data (e.g., distortion
correction, rescaling, nuisance regression).
If you are confident that your data have been preprocessed correctly prior to
applying tedana, and you encounter this problem, please submit a question to `NeuroStars`_.


I think that some BOLD ICA components have been misclassified as noise.
```````````````````````````````````````````````````````````````````````
``tedana`` allows users to manually specify accepted components when calling the pipeline.
You can use the ``--manacc`` argument to specify the indices of components to accept.


Why isn't v3.2 of the component selection algorithm supported in ``tedana``?
````````````````````````````````````````````````````````````````````````````
There is a lot of solid logic behind the updated version of the TEDICA component
selection algorithm, first added to the original ME-ICA codebase `here`_ by Dr. Prantik Kundu.
However, we (the ``tedana`` developers) have encountered certain difficulties
with this method (e.g., misclassified components) and the method itself has yet
to be validated in any papers, posters, etc., which is why we have chosen to archive
the v3.2 code, with the goal of revisiting it when ``tedana`` is more stable.

Anyone interested in using v3.2 may compile and install an earlier release (<=0.0.4) of ``tedana``.

What is the warning about ``duecredit``?
`````````````````````````````````````````
``duecredit`` is a python package that is used, but not required by ``tedana``. These warnings do
not affect any of the processing within the ``tedana``. To avoide this warning, you can install
``duecredit`` with ``pip install duecredit``. For more information about ``duecredit`` and concerns about 
the citation and visibility of software or methods, visit the `duecredit`_ github. 

.. _duecredit: https://github.com/duecredit/duecredit

.. _here: https://bitbucket.org/prantikk/me-ica/commits/906bd1f6db7041f88cd0efcac8a74074d673f4f5

.. _NeuroStars: https://neurostars.org
.. _fMRIPrep: https://fmriprep.readthedocs.io
.. _afni_proc.py: https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html

Multi-echo fMRI
---------------

Will I encounter SAR limits more often with multi-echo fMRI?
````````````````````````````````````````````````````````````
While multi-echo does lead to collecting more images during each TR (one per echo), there is still only a single
radiofrequency pulse. For this reason, there is no change in SAR for participants intrinsic to multi-echo
fMRI. 

Can I combine multiband (simultaneous multislice) with multi-echo fMRI?
```````````````````````````````````````````````````````````````````````
Yes, these techniques are completely seperate. Mutliband fMRI leads to collecting multiple slices within a volume
simultaneouly, while multi-echo fMRI is instead related to collecting multiple unique volumes. These techniques can 
be combined to reduce the TR in a multi-echo sequence. 


