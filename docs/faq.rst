
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

``tedana_reclassify`` allows users to manually alter component classifications.
This can both be used as a command line tool or as part of other interactive
programs, such as `RICA`_. RICA creates a graphical interface that is similar to
the build-in tedana reports that lets users interactively change component
classifications. Both programs will log which component classifications were
manually altered. If one wants to retain the original denoised time series,
make sure to output the denoised time series into a separate directory.

.. _RICA: https://github.com/ME-ICA/rica

*************************************************************************************
[tedana] What is the difference between the kundu and minimal decision trees?
*************************************************************************************

The decision tree is the series of conditions through which each component is
classified as accepted or rejected. The kundu tree (`--tree kundu`)
was used in Prantik Kundu's MEICA v2.7 is the classification process that has long
been used by ``tedana`` and users have been generally content with the results. The
kundu tree used multiple intersecting metrics and rankings classify components.
How these steps may interact on specific datasets is opaque. While there is a kappa
(T2*-weighted) elbow threshold and a rho (S0-weighted) elbow threshold, as discussed
in publications, no component is accepted or rejected because of those thresholds.
Users sometimes notice rejected components that clearly should been accepted. For
example, a component that included a clear T2*-weighted V1 response to a block design
flashing checkerboard was sometimes rejected because the relatively large variance of
that component interacted with a rejection criterion.

The minimal tree (`--tree minimal`) is designed to be easier to understand and less
likely to reject T2* weighted components. There are a few other critiera, but components
with `kappa>kappa elbow` and `rho<rho eblow` should all be accepted and the rho elbow
threshold is less stringent. If kappa is above threshold and more than 2X rho then it
is also accepted under the assumption that, even if a component contains noise, there
is sufficient T2*-weighted signal to retain. Similarly to the kundu tree, components
with very low variance are retained so that degrees of freedom aren't wasted by
removing them, but `minimal` makes sure that no more than 1% of total variance is
removed this way.

``tedana`` developers still want to examine how the minimal tree performs on a wide
range of datasets, but primary benefit is that it is possible to describe what it does
in a short paragraph. The minimal tree will retain some components that kundu
appropriately classifies as noise, and it will reject some components that kundu
accepts. On balance, we expect it to be a more conservative option that should not
remove noise as agressively as kundu, but will be less likely to reject components that
clearly contain signal-of-interest.

It is also possible for users to view both decision trees and `make their own`_.
This might be useful for general methods development and also for using ``tedana``
on multi-echo datasets with properties that differs from those these trees have been
tested on (i.e. human whole-brain acqusitions). It is also possible, but a bit more
challenging, to add additional metrics for each component so that the selection process
can include additional criteria.

.. _make their own: building\ decision\ trees.html

*************************************************************************************
[tedana] What different versions of this method exist?
*************************************************************************************

Dr. Prantik Kundu developed a multi-echo ICA (ME-ICA) denoising method and
`shared code on bitbucket`_ to allow others to use the method. A nearly identical
version of this code is `distributed with AFNI as MEICA v2.5 beta 11`_. Most early
publications that validated the MEICA method used variants of this code. That code
runs only on the now defunct python 2.7 and is not under active development.
``tedana`` when run with `--tree kundu --tedpca kundu` (or `--tedpca kundu-stabilize`),
uses the same core algorithm as in MEICA v2.5. Since ICA is a nondeterministic
algorithm and ``tedana`` and MEICA use different PCA and ICA code, the algorithm will
mostly be the same, but the results will not be identical.

Prantik Kundu also worked on `MEICA v3.2`_ (also for python v2.7). The underlying ICA
step is very similar, but the component selection process was different. While this
new approach has potentialy useful ideas, the early ``tedana`` developers experienced
non-trivial component misclassifications and there were no publications that
validated this method. That is why ``tedana`` replicated the established and valided
MEICA v2.5 method and also includes options to ingrate additional component selection
methods. Recently Prantik has started to work `MEICA v3.3`_ (for python >=v3.7) so
that this version of the selection process would again be possible to run.

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
