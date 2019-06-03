# tedana: TE Dependent ANAlysis
[![All Contributors](https://img.shields.io/badge/all_contributors-10-orange.svg?style=flat-square)](#contributors)

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
[![All Contributors](https://img.shields.io/badge/all_contributors-10-orange.svg?style=flat-square)](#contributors)


## About

``tedana`` originally came about as a part of the [ME-ICA](https://github.com/me-ica/me-ica) pipeline.
The ME-ICA pipeline originally performed both pre-processing and TE-dependent
analysis of multi-echo fMRI data; however, ``tedana`` now assumes that you're
working with data which has been previously preprocessed.

![http://tedana.readthedocs.io/](https://user-images.githubusercontent.com/7406227/40031156-57b7cbb8-57bc-11e8-8c51-5b29f2e86a48.png)

More information and documentation can be found at https://tedana.readthedocs.io/.

## Installation

You'll need to set up a working development environment to use `tedana`.
To set up a local environment, you will need Python >=3.5 and the following packages will need to be installed:

[numpy](http://www.numpy.org/)
[scipy](https://www.scipy.org/)
[scikit-learn](http://scikit-learn.org/stable/)
[nilearn](https://nilearn.github.io/)
[nibabel>=2.1.0](http://nipy.org/nibabel/)

You can then install `tedana` with

```bash
pip install tedana
```

### Creating a miniconda environment for use with `tedana`
In using `tedana`, you can optionally configure [a conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html).

We recommend using [miniconda3](https://conda.io/miniconda.html).
After installation, you can use the following commands to create an environment for `tedana`:

```bash
conda create -n ENVIRONMENT_NAME python=3 pip mdp numpy scikit-learn scipy
conda activate ENVIRONMENT_NAME
pip install nilearn nibabel
pip install tedana
```

`tedana` will then be available in your path.
This will also allow any previously existing tedana installations to remain untouched.

To exit this conda environment, use

```bash
conda deactivate
```

NOTE: Conda < 4.6 users will need to use the soon-to-be-deprecated option
`source` rather than `conda` for the activation and deactivation steps.
You can read more about managing conda environments and this discrepancy here:
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Getting involved

We :yellow_heart: new contributors!
To get started, check out [our contributing guidelines](https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md).

Want to learn more about our plans for developing ``tedana``?
Have a question, comment, or suggestion?
Open or comment on one of [our issues](https://github.com/ME-ICA/tedana/issues)!
If you're not sure where to begin,
feel free to pop into [Gitter](https://gitter.im/ME-ICA/tedana) and
introduce yourself! We will be happy to help you find somewhere to get
started.

We ask that all contributors to ``tedana`` across all project-related
spaces (including but not limited to: GitHub, Gitter, and project emails),
adhere to our
[code of conduct](https://github.com/ME-ICA/tedana/blob/master/CODE_OF_CONDUCT.md).

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table><tr><td align="center"><a href="https://github.com/dowdlelt"><img src="https://avatars2.githubusercontent.com/u/15126366?v=4" width="100px;" alt="Logan Dowdle"/><br /><sub><b>Logan Dowdle</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=dowdlelt" title="Code">💻</a> <a href="#question-dowdlelt" title="Answering Questions">💬</a> <a href="#design-dowdlelt" title="Design">🎨</a> <a href="https://github.com/ME-ICA/tedana/issues?q=author%3Adowdlelt" title="Bug reports">🐛</a></td><td align="center"><a href="http://emdupre.me"><img src="https://avatars3.githubusercontent.com/u/15017191?v=4" width="100px;" alt="Elizabeth DuPre"/><br /><sub><b>Elizabeth DuPre</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Code">💻</a> <a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Documentation">📖</a> <a href="#ideas-emdupre" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-emdupre" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#review-emdupre" title="Reviewed Pull Requests">👀</a> <a href="#example-emdupre" title="Examples">💡</a> <a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Tests">⚠️</a> <a href="#question-emdupre" title="Answering Questions">💬</a></td><td align="center"><a href="https://github.com/javiergcas"><img src="https://avatars1.githubusercontent.com/u/7314358?v=4" width="100px;" alt="Javier Gonzalez-Castillo"/><br /><sub><b>Javier Gonzalez-Castillo</b></sub></a><br /><a href="#ideas-javiergcas" title="Ideas, Planning, & Feedback">🤔</a></td><td align="center"><a href="https://github.com/handwerkerd"><img src="https://avatars3.githubusercontent.com/u/7406227?v=4" width="100px;" alt="Dan Handwerker"/><br /><sub><b>Dan Handwerker</b></sub></a><br /><a href="#design-handwerkerd" title="Design">🎨</a> <a href="https://github.com/ME-ICA/tedana/commits?author=handwerkerd" title="Documentation">📖</a> <a href="#example-handwerkerd" title="Examples">💡</a> <a href="#review-handwerkerd" title="Reviewed Pull Requests">👀</a></td><td align="center"><a href="https://github.com/prantikk"><img src="https://avatars0.githubusercontent.com/u/1636689?v=4" width="100px;" alt="Prantik Kundu"/><br /><sub><b>Prantik Kundu</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=prantikk" title="Code">💻</a> <a href="#ideas-prantikk" title="Ideas, Planning, & Feedback">🤔</a></td></tr><tr><td align="center"><a href="http://rossmarkello.me"><img src="https://avatars0.githubusercontent.com/u/14265705?v=4" width="100px;" alt="Ross Markello"/><br /><sub><b>Ross Markello</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=rmarkello" title="Code">💻</a> <a href="#infra-rmarkello" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#question-rmarkello" title="Answering Questions">💬</a></td><td align="center"><a href="http://tsalo.github.io"><img src="https://avatars3.githubusercontent.com/u/8228902?v=4" width="100px;" alt="Taylor Salo"/><br /><sub><b>Taylor Salo</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=tsalo" title="Code">💻</a> <a href="#ideas-tsalo" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ME-ICA/tedana/commits?author=tsalo" title="Documentation">📖</a> <a href="#tutorial-tsalo" title="Tutorials">✅</a> <a href="#question-tsalo" title="Answering Questions">💬</a></td><td align="center"><a href="https://github.com/jbteves"><img src="https://avatars3.githubusercontent.com/u/26722533?v=4" width="100px;" alt="Joshua Teves"/><br /><sub><b>Joshua Teves</b></sub></a><br /><a href="#projectManagement-jbteves" title="Project Management">📆</a> <a href="https://github.com/ME-ICA/tedana/commits?author=jbteves" title="Documentation">📖</a> <a href="#review-jbteves" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-jbteves" title="Maintenance">🚧</a></td><td align="center"><a href="https://whitakerlab.github.io"><img src="https://avatars1.githubusercontent.com/u/3626306?v=4" width="100px;" alt="Kirstie Whitaker"/><br /><sub><b>Kirstie Whitaker</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=KirstieJane" title="Documentation">📖</a> <a href="#projectManagement-KirstieJane" title="Project Management">📆</a> <a href="#review-KirstieJane" title="Reviewed Pull Requests">👀</a> <a href="#talk-KirstieJane" title="Talks">📢</a></td><td align="center"><a href="https://github.com/monicayao"><img src="https://avatars1.githubusercontent.com/u/35382166?v=4" width="100px;" alt="monicayao"/><br /><sub><b>Monica Yao</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=monicayao" title="Documentation">📖</a></td></tr></table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
