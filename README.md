# tedana

`TE`-`de`pendent `ana`lysis (_tedana_) is a Python module for denoising multi-echo functional magnetic resonance imaging (fMRI) data.

[![CircleCI](https://circleci.com/gh/ME-ICA/tedana.svg?style=shield)](https://circleci.com/gh/ME-ICA/tedana)
[![Documentation Status](https://readthedocs.org/projects/tedana/badge/?version=latest)](http://tedana.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/me-ica/tedana/branch/master/graph/badge.svg)](https://codecov.io/gh/me-ica/tedana)
[![License](https://img.shields.io/badge/License-LGPL%202.0-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![Join the chat at https://gitter.im/ME-ICA/tedana](https://badges.gitter.im/ME-ICA/tedana.svg)](https://gitter.im/ME-ICA/tedana?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![DOI](https://zenodo.org/badge/110845855.svg)](https://zenodo.org/badge/latestdoi/110845855)

   ![](https://user-images.githubusercontent.com/7406227/40031156-57b7cbb8-57bc-11e8-8c51-5b29f2e86a48.png)


## About

`tedana` originally came about as a part of the [`ME-ICA`](https://github.com/me-ica/me-ica) pipeline.
The ME-ICA pipeline orignially performed both pre-processing and TE-dependent analysis of multi-echo fMRI data; however, `tedana` now assumes that you're working with data which has been previously preprocessed.
If you're in need of a pre-processing pipeline, we recommend [`fmriprep`](https://github.com/poldracklab/fmriprep/) which has been tested for compatibility with multi-echo fMRI data and `tedana`.

### Why Multi-Echo?

Multi-echo fMRI data is obtained by acquiring multiple TEs (commonly called [echo times](http://mriquestions.com/tr-and-te.html)) for each MRI volume during data collection.
While fMRI signal contains important neural information (termed the blood oxygen-level dependent, or [BOLD signal](http://www.fil.ion.ucl.ac.uk/spm/course/slides10-zurich/Kerstin_BOLD.pdf)), it also contains "noise" (termed non-BOLD signal) caused by things like participant motion and changes in breathing.
Because the BOLD signal is known to decay at a set rate, collecting multiple echos allows us to assess whether components of the fMRI signal are BOLD- or non-BOLD.
For a comprehensive review, see [Kundu et al. (2017), _NeuroImage_](https://paperpile.com/shared/eH3PPu).

In `tedana`, we take the time series from all the collected TEs, combine them, and decompose the resulting data into components that can be classified as BOLD or non-BOLD. This is performed in a series of steps including:

* Principal components analysis
* Independent components analysis
* Component classification

More information and documentation can be found at https://tedana.readthedocs.io/.

## Installation

You'll need to set up a working development environment to use `tedana`.
We provide a Dockerfile for this purpose (check out [tips on using Docker](https://neurohackweek.github.io/docker-for-scientists/)), but you can also set up your environment locally.
If you choose the latter, the following packages will need to be installed as dependencies:

mdp  
nilearn  
nibabel>=2.1.0  
numpy  
scikit-learn  
scipy

`tedana` will eventually be hosted on PyPi. In the interim, you can still install it with `pip` using:

```
pip install https://github.com/ME-ICA/tedana/archive/master.tar.gz
```

## Getting involved

We :yellow_heart: new contributors !
To get started, check out [our contributing guidelines](https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md).

Want to learn more about our plans for developing `tedana` ?
Have a question, comment, or suggestion ?
Open or comment on one of [our issues](https://github.com/ME-ICA/tedana/issues) !

We ask that all contributions to `tedana` respect our [code of conduct](https://github.com/ME-ICA/tedana/blob/master/Code_of_Conduct.md).
