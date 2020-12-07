# tedana: TE Dependent ANAlysis

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
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/ME-ICA/tedana.svg)](http://isitmaintained.com/project/ME-ICA/tedana "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/ME-ICA/tedana.svg)](http://isitmaintained.com/project/ME-ICA/tedana "Percentage of issues still open")
[![Join the chat at https://gitter.im/ME-ICA/tedana](https://badges.gitter.im/ME-ICA/tedana.svg)](https://gitter.im/ME-ICA/tedana?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Join our tinyletter mailing list](https://img.shields.io/badge/receive-our%20newsletter%20❤%EF%B8%8F-blueviolet.svg)](https://tinyletter.com/tedana-devs)
[![All Contributors](https://img.shields.io/badge/all_contributors-14-orange.svg?style=flat-square)](#contributors)


## About

``tedana`` originally came about as a part of the [ME-ICA](https://github.com/me-ica/me-ica) pipeline.
The ME-ICA pipeline originally performed both pre-processing and TE-dependent analysis of multi-echo fMRI data; however, ``tedana`` now assumes that you're working with data which has been previously preprocessed.

![http://tedana.readthedocs.io/](https://user-images.githubusercontent.com/7406227/40031156-57b7cbb8-57bc-11e8-8c51-5b29f2e86a48.png)

More information and documentation can be found at https://tedana.readthedocs.io.

## Installation

### Use `tedana` with your local Python environment

You'll need to set up a working development environment to use `tedana`.
To set up a local environment, you will need Python >=3.5 and the following packages will need to be installed:

* [numpy>=1.14](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [nilearn](https://nilearn.github.io/)
* [nibabel>=2.1.0](http://nipy.org/nibabel/)

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
This will also allow any previously existing `tedana` installations to remain untouched.

To exit this conda environment, use

```bash
conda deactivate
```

NOTE: Conda < 4.6 users will need to use the soon-to-be-deprecated option `source` rather than `conda` for the activation and deactivation steps.
You can read more about managing conda environments and this discrepancy [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Use and contribute to `tedana` as a developer

If you aim to contribute to the `tedana` code base and/or documentation, please first read the developer installation instructions in [our contributing section](https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md). You can then continue to set up your preferred development environment.

## Getting involved

We :yellow_heart: new contributors!
To get started, check out [our contributing guidelines](https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md)
and our [developer's guide](https://tedana.readthedocs.io/en/latest/developing.html).

Want to learn more about our plans for developing ``tedana``?
Have a question, comment, or suggestion?
Open or comment on one of [our issues](https://github.com/ME-ICA/tedana/issues)!

If you're not sure where to begin, feel free to pop into [Gitter](https://gitter.im/ME-ICA/tedana) and introduce yourself!
We will be happy to help you find somewhere to get started.

If you don't want to get lots of notifications, we send out newsletters approximately once per month though our TinyLetter mailing list.
You can view the [previous newsletters](https://tinyletter.com/tedana-devs/archive) and/or sign up to receive future ones at [https://tinyletter.com/tedana-devs](https://tinyletter.com/tedana-devs).

We ask that all contributors to ``tedana`` across all project-related spaces (including but not limited to: GitHub, Gitter, and project emails), adhere to our [code of conduct](https://github.com/ME-ICA/tedana/blob/master/CODE_OF_CONDUCT.md).

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/dowdlelt"><img src="https://avatars2.githubusercontent.com/u/15126366?v=4" width="100px;" alt=""/><br /><sub><b>Logan Dowdle</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=dowdlelt" title="Code">💻</a> <a href="#question-dowdlelt" title="Answering Questions">💬</a> <a href="#design-dowdlelt" title="Design">🎨</a> <a href="https://github.com/ME-ICA/tedana/issues?q=author%3Adowdlelt" title="Bug reports">🐛</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Adowdlelt" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="http://emdupre.me"><img src="https://avatars3.githubusercontent.com/u/15017191?v=4" width="100px;" alt=""/><br /><sub><b>Elizabeth DuPre</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Code">💻</a> <a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Documentation">📖</a> <a href="#ideas-emdupre" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-emdupre" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Aemdupre" title="Reviewed Pull Requests">👀</a> <a href="#example-emdupre" title="Examples">💡</a> <a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Tests">⚠️</a> <a href="#question-emdupre" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://github.com/javiergcas"><img src="https://avatars1.githubusercontent.com/u/7314358?v=4" width="100px;" alt=""/><br /><sub><b>Javier Gonzalez-Castillo</b></sub></a><br /><a href="#ideas-javiergcas" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ME-ICA/tedana/commits?author=javiergcas" title="Code">💻</a> <a href="#design-javiergcas" title="Design">🎨</a></td>
    <td align="center"><a href="https://github.com/handwerkerd"><img src="https://avatars3.githubusercontent.com/u/7406227?v=4" width="100px;" alt=""/><br /><sub><b>Dan Handwerker</b></sub></a><br /><a href="#design-handwerkerd" title="Design">🎨</a> <a href="https://github.com/ME-ICA/tedana/commits?author=handwerkerd" title="Documentation">📖</a> <a href="#example-handwerkerd" title="Examples">💡</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Ahandwerkerd" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/prantikk"><img src="https://avatars0.githubusercontent.com/u/1636689?v=4" width="100px;" alt=""/><br /><sub><b>Prantik Kundu</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=prantikk" title="Code">💻</a> <a href="#ideas-prantikk" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://rossmarkello.me"><img src="https://avatars0.githubusercontent.com/u/14265705?v=4" width="100px;" alt=""/><br /><sub><b>Ross Markello</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=rmarkello" title="Code">💻</a> <a href="#infra-rmarkello" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#question-rmarkello" title="Answering Questions">💬</a></td>
    <td align="center"><a href="http://tsalo.github.io"><img src="https://avatars3.githubusercontent.com/u/8228902?v=4" width="100px;" alt=""/><br /><sub><b>Taylor Salo</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=tsalo" title="Code">💻</a> <a href="#ideas-tsalo" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ME-ICA/tedana/commits?author=tsalo" title="Documentation">📖</a> <a href="#tutorial-tsalo" title="Tutorials">✅</a> <a href="#question-tsalo" title="Answering Questions">💬</a> <a href="https://github.com/ME-ICA/tedana/issues?q=author%3Atsalo" title="Bug reports">🐛</a> <a href="https://github.com/ME-ICA/tedana/commits?author=tsalo" title="Tests">⚠️</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Atsalo" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/jbteves"><img src="https://avatars3.githubusercontent.com/u/26722533?v=4" width="100px;" alt=""/><br /><sub><b>Joshua Teves</b></sub></a><br /><a href="#projectManagement-jbteves" title="Project Management">📆</a> <a href="https://github.com/ME-ICA/tedana/commits?author=jbteves" title="Documentation">📖</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Ajbteves" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-jbteves" title="Maintenance">🚧</a> <a href="https://github.com/ME-ICA/tedana/commits?author=jbteves" title="Code">💻</a></td>
    <td align="center"><a href="https://whitakerlab.github.io"><img src="https://avatars1.githubusercontent.com/u/3626306?v=4" width="100px;" alt=""/><br /><sub><b>Kirstie Whitaker</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=KirstieJane" title="Documentation">📖</a> <a href="#projectManagement-KirstieJane" title="Project Management">📆</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3AKirstieJane" title="Reviewed Pull Requests">👀</a> <a href="#talk-KirstieJane" title="Talks">📢</a></td>
    <td align="center"><a href="https://github.com/monicayao"><img src="https://avatars1.githubusercontent.com/u/35382166?v=4" width="100px;" alt=""/><br /><sub><b>Monica Yao</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=monicayao" title="Documentation">📖</a> <a href="https://github.com/ME-ICA/tedana/commits?author=monicayao" title="Tests">⚠️</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://www.fmrwhy.com/"><img src="https://avatars0.githubusercontent.com/u/10141237?v=4" width="100px;" alt=""/><br /><sub><b>Stephan Heunis</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=jsheunis" title="Documentation">📖</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/benoit-beranger/"><img src="https://avatars0.githubusercontent.com/u/16976839?v=4" width="100px;" alt=""/><br /><sub><b>Benoît Béranger</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=benoitberanger" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/eurunuela"><img src="https://avatars0.githubusercontent.com/u/13706448?v=4" width="100px;" alt=""/><br /><sub><b>Eneko Uruñuela</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=eurunuela" title="Code">💻</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Aeurunuela" title="Reviewed Pull Requests">👀</a> <a href="#ideas-eurunuela" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/CesarCaballeroGaudes"><img src="https://avatars1.githubusercontent.com/u/7611340?v=4" width="100px;" alt=""/><br /><sub><b>Cesar Caballero Gaudes</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=CesarCaballeroGaudes" title="Documentation">📖</a> <a href="https://github.com/ME-ICA/tedana/commits?author=CesarCaballeroGaudes" title="Code">💻</a></td>
    <td align="center"><a href="http://isla.st"><img src="https://avatars2.githubusercontent.com/u/23707851?v=4" width="100px;" alt=""/><br /><sub><b>Isla</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3AIslast" title="Reviewed Pull Requests">👀</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mjversluis"><img src="https://avatars0.githubusercontent.com/u/32125111?v=4" width="100px;" alt=""/><br /><sub><b>mjversluis</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=mjversluis" title="Documentation">📖</a></td>
    <td align="center"><a href="https://mvaziri.github.io/"><img src="https://avatars2.githubusercontent.com/u/4219325?v=4" width="100px;" alt=""/><br /><sub><b>Maryam</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=mvaziri" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/aykhojandi"><img src="https://avatars1.githubusercontent.com/u/38105040?v=4" width="100px;" alt=""/><br /><sub><b>aykhojandi</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=aykhojandi" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/smoia"><img src="https://avatars3.githubusercontent.com/u/35300580?v=4" width="100px;" alt=""/><br /><sub><b>Stefano Moia</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=smoia" title="Code">💻</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Asmoia" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/ME-ICA/tedana/commits?author=smoia" title="Documentation">📖</a></td>
    <td align="center"><a href="https://www.notzaki.com/"><img src="https://avatars1.githubusercontent.com/u/9019681?v=4" width="100px;" alt=""/><br /><sub><b>Zaki A.</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/issues?q=author%3AnotZaki" title="Bug reports">🐛</a> <a href="https://github.com/ME-ICA/tedana/commits?author=notZaki" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
