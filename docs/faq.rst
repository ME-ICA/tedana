
FAQ
---

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

.. _here: https://bitbucket.org/prantikk/me-ica/commits/906bd1f6db7041f88cd0efcac8a74074d673f4f5

.. _NeuroStars: https://neurostars.org
.. _fMRIPrep: https://fmriprep.readthedocs.io
.. _afni_proc.py: https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html
