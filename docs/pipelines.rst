ME-EPI preprocessing
====================

tedana must be called in the context of a larger ME-EPI preprocessing pipeline.
Two common pipelines which support ME-EPI processing include `fMRIPrep`_ and `afni_proc.py`_.

.. _fMRIPrep: https://fmriprep.readthedocs.io
.. _afni_proc.py: https://afni.nimh.nih.gov/pub/dist/doc/program_help/afni_proc.py.html

Users can also construct their own preprocessing pipeline for ME-EPI data from which to call tedana.
Any constructed pipeline must include the following steps:

#. Slice time correction
------------------------

Similarly to single-echo EPI data, slice time correction allows use to assume that voxels across
slices represent roughly simultaneous events.
This must be done before multi-echo denoising because slice timing differences might cause problems
with the echo-dependent estimates.

The slice time is generally defined as the excitation pulse time for each slice.
For single-echo EPI data, that excitation time would be the same regardless of the echo time,
and the same is true when one is collecting multiple echoes after a single excitation pulse.
Therefore, we suggest using the same slice timing for all echoes.
