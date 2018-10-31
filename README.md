tedana: TE Dependent ANAlysis
=============================

The ``tedana`` package is part of the ME-ICA pipeline, performing TE-dependent
analysis of multi-echo functional magnetic resonance imaging (fMRI) data.
``TE``-``de``pendent ``ana``lysis (``tedana``) is a Python module for denoising
multi-echo functional magnetic resonance imaging (fMRI) data.

[![Latest Version](https://img.shields.io/pypi/v/tedana.svg)](https://pypi.python.org/pypi/tedana/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tedana.svg)](https://pypi.python.org/pypi/tedana/)
[![DOI](https://zenodo.org/badge/110845855.svg)](https://zenodo.org/badge/latestdoi/110845855)
[![License](https://img.shields.io/badge/License-LGPL%202.0-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![CircleCI](https://circleci.com/gh/ME-ICA/tedana.svg?style=shield)](https://circleci.com/gh/ME-ICA/tedana)
[![Documentation Status](https://readthedocs.org/projects/tedana/badge/?version=latest)](http://tedana.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/me-ica/tedana/branch/master/graph/badge.svg)](https://codecov.io/gh/me-ica/tedana)
[![Join the chat at https://gitter.im/ME-ICA/tedana](https://badges.gitter.im/ME-ICA/tedana.svg)](https://gitter.im/ME-ICA/tedana?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

About
-----

``tedana`` originally came about as a part of the [ME-ICA](https://github.com/me-ica/me-ica) pipeline.
The ME-ICA pipeline originally performed both pre-processing and TE-dependent
analysis of multi-echo fMRI data; however, ``tedana`` now assumes that you're
working with data which has been previously preprocessed.
If you're in need of a preprocessing pipeline, we recommend
[fmriprep](https://github.com/poldracklab/fmriprep/), which has been tested
for compatibility with multi-echo fMRI data and ``tedana``.

![http://tedana.readthedocs.io/](https://user-images.githubusercontent.com/7406227/40031156-57b7cbb8-57bc-11e8-8c51-5b29f2e86a48.png)

More information and documentation can be found at https://tedana.readthedocs.io/.

## Installation

You'll need to set up a working development environment to use `tedana`.
To set up a local environment, you will need Python >=3.6 and the following packages will need to be installed:

[mdp](https://pypi.org/project/MDP/)  
[numpy](http://www.numpy.org/)   
[scikit-learn](http://scikit-learn.org/stable/)   
[scipy](https://www.scipy.org/)    
[nilearn](https://nilearn.github.io/)     
[nibabel>=2.1.0](http://nipy.org/nibabel/)      

You can then install `tedana` with

```bash
pip install tedana
```

### Creating a miniconda environment for use with tedana
In order to use tedana, you can configure a miniconda environment for your user profile in Mac and Linux systems. (Systems administrators: note that global installation of miniconda in /usr/local/ may cause some trouble for individual users; user installations are recommended; pip installation will require root privileges otherwise. Some users may need you to adjust their home directory permissions during a miniconda install). This also makes it straightforward to install nibabel and nilearn. You can use [miniconda3](https://conda.io/miniconda.html). Then, use the following commands to create the environment:

```
conda create -n ENVIRONMENT_NAME python=3 pip mdp numpy scikit-learn scipy
source activate ENVIRONMENT_NAME
pip install nilearn nibabel tedana
```

At this stage tedana will be added to your path, while allowing any existing tedana installations, such as the one packaged with AfNI, to remain untouched. To leave this environment, use

```
source deactivate
```

Getting involved
----------------

We :yellow_heart: new contributors!
To get started, check out [our contributing guidelines](https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md).

Want to learn more about our plans for developing ``tedana``?
Have a question, comment, or suggestion?
Open or comment on one of [our issues](https://github.com/ME-ICA/tedana/issues)!

We ask that all contributions to ``tedana`` respect our [code of conduct](https://github.com/ME-ICA/tedana/blob/master/CODE_OF_CONDUCT.md).
