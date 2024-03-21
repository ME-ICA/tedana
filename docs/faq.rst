
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

.. _fMRIPrep: https://fmriprep.readthedocs.io

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

************************************************
[tedana] Why are there so few voxels in my mask?
************************************************

Several steps in ``tedana`` require excluding voxels outside of the brain. If the
provided data are not already masked, ``tedana`` runs
:func:`nilearn.masking.compute_epi_mask` to create a mask. This often works, but
it is a fairly simple method and there are better masking tools in most fMRI
processing pipelines, such as `AFNI`_. If your full preprocessing pipeline already
generates a mask for fMRI data, we recommend inputting that mask to ``tedana`` and using
the ``--mask`` option. ``tedana`` additionally creates an adaptive mask that will
exclude voxels from some steps that have large dropout in later echoes. The adaptive
mask will flag some voxels in common dropout regions, but should not radically alter
the inputted mask. The outputted optimally combined and denoised datasets will include
all voxels in the inputted mask unless the adaptive mask identifies a voxel has having
zero good echoes. Here is more information on the `creation and use of the adaptive mask`_.

.. _AFNI: http://afni.nimh.nih.gov
.. _creation and use of the adaptive mask: approach.html#adaptive-mask-generation

************************************
[tedana] ICA has failed to converge.
************************************

The TEDICA step may fail to converge if TEDPCA is either too strict
(i.e., there are too few components) or too lenient (there are too many).

With updates to the ``tedana`` code, this issue is now rare, but it may happen
when preprocessing has not been applied to the data, or when improper steps have
been applied to the data (e.g. rescaling, nuisance regression).
If you are confident that your data have been preprocessed correctly prior to
applying tedana, and you encounter this problem, please submit a question to `NeuroStars`_.

.. _NeuroStars: https://neurostars.org

.. _manual classification:

********************************************************************************
[tedana] Can I manually reclassify components?
********************************************************************************

``ica_reclassify`` allows users to manually alter component classifications.
This can both be used as a command line tool or as part of other interactive
programs, such as `RICA`_. RICA creates a graphical interface that is similar to
the build-in tedana reports that lets users interactively change component
classifications. Both programs will log which component classifications were
manually altered. If one wants to retain the original denoised time series,
make sure to output the denoised time series into a separate directory.

.. _RICA: https://github.com/ME-ICA/rica

.. _tree differences:

*********************************************************************************************
[tedana] What are the differences between the tedana_orig, meica, and minimal decision trees?
*********************************************************************************************

The decision tree is the series of conditions through which each component is
classified as accepted or rejected. The meica tree (`--tree meica`) was created by Prantik
Kundu for ``MEICA v2.5``, the predecessor to ``tedana``. Tedana's decision tree was based
on this method, but we noticed a difference that affected edge-case components. There were
components that were re-evalued multiple times late in the decision tree in ``meica``,
but, once ``tedana`` rejected them, they were excluded from additional steps. This means
that ``meica`` may accept a few components that ``tedana`` was rejects. When examining
the effects of this divergance, we saw that ``meica`` sometimes accepted high variance
components. While those additionally accepted components often looked like noise, we wanted
to make sure users were aware of this difference. We include options to run the ``meica``
tree and the ``tedana_orig`` tree which has been successfully used for years.
``tedana_orig`` will always remove the same or more components.

Both of the above trees use multiple intersecting metrics and rankings to classify
components. How these steps may interact on specific datasets is opaque. While there is
a kappa (T2*-weighted) elbow threshold and a rho (S0-weighted) elbow threshold, as
discussed in publications, no component is accepted or rejected because of those thresholds.
Users sometimes notice rejected components that clearly should have been accepted. For
example, a component that included a clear T2*-weighted V1 response to a block design
flashing checkerboard was sometimes rejected because the relatively large variance of
that component interacted with a rejection criterion.

The minimal tree (`--tree minimal`) is designed to be easier to understand and less
likely to reject T2* weighted components. There are a few other criteria, but components
with `kappa>kappa elbow` and `rho<rho elbow` should all be accepted, and the rho elbow
threshold is less stringent. If kappa is above threshold and more than 2X rho then it
is also accepted under the assumption that, even if a component contains noise, there
is sufficient T2*-weighted signal to retain. Similarly to the tedana_orig and meica
trees, components with very low variance are retained so that degrees of freedom aren't
wasted by removing them, but `minimal` makes sure that no more than 1% of total variance
is removed this way.

``tedana`` developers still want to examine how the minimal tree performs on a wide
range of datasets, but the primary benefit is that it is possible to describe what it does
in a short paragraph. The minimal tree will retain some components that the other trees
appropriately classifies as noise, and it will reject some components that the other trees
accept. More work is needed to validate the results of the minimal tree. The precise
thresholds and steps in the minimal tree may change as the results from running it are
examined on a wider range of data. The developers are looking for more people to compare
results between the trees. Until it is evaluated more, we recommend that any who
uses ``minimal`` should examine the `tedana report`_ for any misclassifications.

It is also possible for users to view both decision trees and `make their own`_.
This might be useful for general methods development and also for using ``tedana``
on multi-echo datasets with properties different from those of the datasets these trees have been
tested on (i.e. human whole-brain acquisitions). It is also possible, but a bit more
challenging, to add additional metrics for each component so that the selection process
can include additional criteria.

`Flow charts detailing these decision trees are here`_.

.. _Flow charts detailing these decision trees are here: included_decision_trees.html
.. _make their own: building_decision_trees.html
.. _tedana report: outputs.html#ica-components-report

*************************************************************************************
[tedana] What different versions of this method exist?
*************************************************************************************

Dr. Prantik Kundu developed a multi-echo ICA (ME-ICA) denoising method and
`shared code on bitbucket`_ to allow others to use the method. A nearly identical
version of this code is `distributed with AFNI as MEICA v2.5 beta 11`_. Most early
publications that validated the MEICA method used variants of this code. That code
runs only on the now defunct python 2.7.
``tedana`` when run with `--tree meica --tedpca kundu` (or `--tedpca kundu-stabilize`),
uses the same core algorithm as in MEICA v2.5. Since ICA is a nondeterministic
algorithm and ``tedana`` and MEICA use different PCA and ICA code, the algorithm will
mostly be the same, but the results will not be identical.

Prantik Kundu also worked on `MEICA v3.2`_ (also for python v2.7). The underlying ICA
step is very similar, but the component selection process was different. While the
approach in `MEICA v3.2`_ has potentially useful ideas, the early ``tedana`` developers
experienced non-trivial component misclassifications and there were no publications that
validated this method. That is why ``tedana`` replicated the established and validated
MEICA v2.5 method and also includes options to integrate additional component selection
methods. In 2022, Prantik made `MEICA v3.3`_ which runs on for python >=v3.7. It is not
under active development, but it should be possible to run.

.. _shared code on bitbucket: https://bitbucket.org/prantikk/me-ica/src/experimental
.. _distributed with AFNI as MEICA v2.5 beta 11: https://github.com/afni/afni/tree/master/src/pkundu
.. _MEICA v3.2: https://github.com/ME-ICA/me-ica/tree/53191a7e8838788acf837fdf7cb3026efadf49ac
.. _MEICA v3.3: https://github.com/ME-ICA/me-ica/tree/ME-ICA_v3.3.0


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

********************************************************************************
[ME-fMRI] How do field maps and distortion correction work with multi-echo fMRI?
********************************************************************************

There are many different approaches to susceptibility distortion correction out there- far too many to discuss here.
However, the good news is that distortion does not typically change across echoes in multi-echo fMRI.
In most cases, the readout acquisition type and total readout time are the same for each echo,
so distortion will remain relatively constant, even as dropout increases with echo time.

What this means is that, in the vast majority of multi-echo datasets,
standard distortion correction methods will work, and should be applied in the same manner on all echoes.
For example, if you acquire a blip-up/blip-down set of images for all of your echo times,
you should use the first echo time's images to generate the undistortion transform,
as long as that first echo has sufficient gray/white constrast to be useful for alignment
(in which case, use the earliest echo that does have good contrast).

For context, please see
`this NeuroStars thread <https://neurostars.org/t/multi-echo-pepolar-fieldmaps-bids-spec-sdcflows-grayzone/23933/5>`_.
