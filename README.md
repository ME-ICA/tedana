# tedana: TE Dependent ANAlysis

[![Latest Version](https://img.shields.io/pypi/v/tedana.svg)](https://pypi.python.org/pypi/tedana/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tedana.svg)](https://pypi.python.org/pypi/tedana/)
[![JOSS DOI](https://joss.theoj.org/papers/10.21105/joss.03669/status.svg)](https://doi.org/10.21105/joss.03669)
[![Zenodo DOI](https://zenodo.org/badge/110845855.svg)](https://zenodo.org/badge/latestdoi/110845855)
[![License](https://img.shields.io/badge/License-LGPL%202.0-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![CircleCI](https://circleci.com/gh/ME-ICA/tedana.svg?style=shield)](https://circleci.com/gh/ME-ICA/tedana)
[![Documentation Status](https://readthedocs.org/projects/tedana/badge/?version=latest)](http://tedana.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/me-ica/tedana/branch/main/graph/badge.svg)](https://codecov.io/gh/me-ica/tedana)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/ME-ICA/tedana.svg)](http://isitmaintained.com/project/ME-ICA/tedana "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/ME-ICA/tedana.svg)](http://isitmaintained.com/project/ME-ICA/tedana "Percentage of issues still open")
[![Join the chat on Mattermost](https://img.shields.io/badge/Chat%20on-Mattermost-purple.svg)](https://mattermost.brainhack.org/brainhack/channels/tedana)
[![Join our Google Group mailing list](https://img.shields.io/badge/receive-our%20newsletter%20❤%EF%B8%8F-blueviolet.svg)](https://groups.google.com/g/tedana-newsletter)
[![All Contributors](https://img.shields.io/badge/all_contributors-20-orange.svg?style=flat-square)](#contributors)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

``TE``-``de``pendent ``ana``lysis (``tedana``) is a Python library for denoising multi-echo functional magnetic resonance imaging (fMRI) data.
``tedana`` originally came about as a part of the [ME-ICA](https://github.com/me-ica/me-ica) pipeline, although it has since diverged.
An important distinction is that while the ME-ICA pipeline originally performed both pre-processing and TE-dependent analysis of multi-echo fMRI data,
``tedana`` now assumes that you're working with data which has been previously preprocessed.

![http://tedana.readthedocs.io/](https://user-images.githubusercontent.com/7406227/40031156-57b7cbb8-57bc-11e8-8c51-5b29f2e86a48.png)

More information and documentation can be found at https://tedana.readthedocs.io.

## Citing `tedana`

If you use `tedana`, please cite the following papers, as well as our [most recent Zenodo release](https://zenodo.org/badge/latestdoi/110845855):

- DuPre, E. M., Salo, T., Ahmed, Z., Bandettini, P. A., Bottenhorn, K. L.,
  Caballero-Gaudes, C., Dowdle, L. T., Gonzalez-Castillo, J., Heunis, S.,
  Kundu, P., Laird, A. R., Markello, R., Markiewicz, C. J., Moia, S.,
  Staden, I., Teves, J. B., Uruñuela, E., Vaziri-Pashkam, M.,
  Whitaker, K., & Handwerker, D. A. (2021).
  [TE-dependent analysis of multi-echo fMRI with tedana.](https://doi.org/10.21105/joss.03669)
  _Journal of Open Source Software_, _6(66)_, 3669.
  doi:10.21105/joss.03669.
- Kundu, P., Inati, S. J., Evans, J. W., Luh, W. M., & Bandettini, P. A. (2011).
  [Differentiating BOLD and non-BOLD signals in fMRI time series using multi-echo EPI.](https://doi.org/10.1016/j.neuroimage.2011.12.028)
  _NeuroImage_, _60_, 1759-1770.
- Kundu, P., Brenowitz, N. D., Voon, V., Worbe, Y., Vértes, P. E., Inati, S. J.,
  Saad, Z. S., Bandettini, P. A., & Bullmore, E. T. (2013).
  [Integrated strategy for improving functional connectivity mapping using multiecho fMRI.](https://doi.org/10.1073/pnas.1301725110)
  _Proceedings of the National Academy of Sciences_, _110_, 16187-16192.

## Installation

### Use `tedana` with your local Python environment

You'll need to set up a working development environment to use `tedana`.
To set up a local environment, you will need Python >=3.8 and the following packages will need to be installed:

* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [nilearn](https://nilearn.github.io/stable/)
* [nibabel](http://nipy.org/nibabel/)
* [mapca](https://github.com/ME-ICA/mapca)

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

You can confirm that ``tedana`` has successfully installed by launching a Python instance and running:

```python
import tedana
```

You can check that it is available through the command line interface (CLI) with:

```bash
tedana --help
```

If no error occurs, ``tedana`` has correctly installed in your environment!

### Use and contribute to `tedana` as a developer

If you aim to contribute to the `tedana` code base and/or documentation, please first read the developer installation instructions in [our contributing section](https://github.com/ME-ICA/tedana/blob/main/CONTRIBUTING.md). You can then continue to set up your preferred development environment.

## Getting involved

We :yellow_heart: new contributors!
To get started, check out [our contributing guidelines](https://github.com/ME-ICA/tedana/blob/main/CONTRIBUTING.md)
and our [developer's guide](https://tedana.readthedocs.io/en/latest/contributing.html#developer-guidelines).

Want to learn more about our plans for developing ``tedana``?
Have a question, comment, or suggestion?
Open or comment on one of [our issues](https://github.com/ME-ICA/tedana/issues)!

If you're not sure where to begin, feel free to pop into [Mattermost](https://mattermost.brainhack.org/brainhack/channels/tedana) and introduce yourself!
We will be happy to help you find somewhere to get started.

If you don't want to get lots of notifications, we send out newsletters approximately once per month though our Google Group mailing list.
You can view the [previous newsletters](https://groups.google.com/g/tedana-newsletter) and/or sign up to receive future ones by joining at [https://groups.google.com/g/tedana-newsletter](https://groups.google.com/g/tedana-newsletter).

We ask that all contributors to ``tedana`` across all project-related spaces (including but not limited to: GitHub, Mattermost, and project emails), adhere to our [code of conduct](https://github.com/ME-ICA/tedana/blob/main/CODE_OF_CONDUCT.md).

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/dowdlelt"><img src="https://avatars2.githubusercontent.com/u/15126366?v=4?s=100" width="100px;" alt="Logan Dowdle"/><br /><sub><b>Logan Dowdle</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=dowdlelt" title="Code">💻</a> <a href="#question-dowdlelt" title="Answering Questions">💬</a> <a href="#design-dowdlelt" title="Design">🎨</a> <a href="https://github.com/ME-ICA/tedana/issues?q=author%3Adowdlelt" title="Bug reports">🐛</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Adowdlelt" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="20%"><a href="http://emdupre.me"><img src="https://avatars3.githubusercontent.com/u/15017191?v=4?s=100" width="100px;" alt="Elizabeth DuPre"/><br /><sub><b>Elizabeth DuPre</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Code">💻</a> <a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Documentation">📖</a> <a href="#ideas-emdupre" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-emdupre" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Aemdupre" title="Reviewed Pull Requests">👀</a> <a href="#example-emdupre" title="Examples">💡</a> <a href="https://github.com/ME-ICA/tedana/commits?author=emdupre" title="Tests">⚠️</a> <a href="#question-emdupre" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/marco7877"><img src="https://avatars.githubusercontent.com/u/56403434?v=4?s=100" width="100px;" alt="Marco Flores-Coronado"/><br /><sub><b>Marco Flores-Coronado</b></sub></a><br /><a href="#ideas-marco7877" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ME-ICA/tedana/commits?author=marco7877" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/javiergcas"><img src="https://avatars1.githubusercontent.com/u/7314358?v=4?s=100" width="100px;" alt="Javier Gonzalez-Castillo"/><br /><sub><b>Javier Gonzalez-Castillo</b></sub></a><br /><a href="#ideas-javiergcas" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ME-ICA/tedana/commits?author=javiergcas" title="Code">💻</a> <a href="#design-javiergcas" title="Design">🎨</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/handwerkerd"><img src="https://avatars3.githubusercontent.com/u/7406227?v=4?s=100" width="100px;" alt="Dan Handwerker"/><br /><sub><b>Dan Handwerker</b></sub></a><br /><a href="#design-handwerkerd" title="Design">🎨</a> <a href="https://github.com/ME-ICA/tedana/commits?author=handwerkerd" title="Documentation">📖</a> <a href="#example-handwerkerd" title="Examples">💡</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Ahandwerkerd" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/ME-ICA/tedana/commits?author=handwerkerd" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/prantikk"><img src="https://avatars0.githubusercontent.com/u/1636689?v=4?s=100" width="100px;" alt="Prantik Kundu"/><br /><sub><b>Prantik Kundu</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=prantikk" title="Code">💻</a> <a href="#ideas-prantikk" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="20%"><a href="http://rossmarkello.me"><img src="https://avatars0.githubusercontent.com/u/14265705?v=4?s=100" width="100px;" alt="Ross Markello"/><br /><sub><b>Ross Markello</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=rmarkello" title="Code">💻</a> <a href="#infra-rmarkello" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#question-rmarkello" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/pmolfese"><img src="https://avatars.githubusercontent.com/u/3665743?v=4?s=100" width="100px;" alt="Pete Molfese"/><br /><sub><b>Pete Molfese</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=pmolfese" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/n-reddy"><img src="https://avatars.githubusercontent.com/u/58482773?v=4?s=100" width="100px;" alt="Neha Reddy"/><br /><sub><b>Neha Reddy</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/issues?q=author%3An-reddy" title="Bug reports">🐛</a> <a href="https://github.com/ME-ICA/tedana/commits?author=n-reddy" title="Documentation">📖</a> <a href="#ideas-n-reddy" title="Ideas, Planning, & Feedback">🤔</a> <a href="#question-n-reddy" title="Answering Questions">💬</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3An-reddy" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="20%"><a href="http://tsalo.github.io"><img src="https://avatars3.githubusercontent.com/u/8228902?v=4?s=100" width="100px;" alt="Taylor Salo"/><br /><sub><b>Taylor Salo</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=tsalo" title="Code">💻</a> <a href="#ideas-tsalo" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ME-ICA/tedana/commits?author=tsalo" title="Documentation">📖</a> <a href="#tutorial-tsalo" title="Tutorials">✅</a> <a href="#question-tsalo" title="Answering Questions">💬</a> <a href="https://github.com/ME-ICA/tedana/issues?q=author%3Atsalo" title="Bug reports">🐛</a> <a href="https://github.com/ME-ICA/tedana/commits?author=tsalo" title="Tests">⚠️</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Atsalo" title="Reviewed Pull Requests">👀</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/jbteves"><img src="https://avatars3.githubusercontent.com/u/26722533?v=4?s=100" width="100px;" alt="Joshua Teves"/><br /><sub><b>Joshua Teves</b></sub></a><br /><a href="#projectManagement-jbteves" title="Project Management">📆</a> <a href="https://github.com/ME-ICA/tedana/commits?author=jbteves" title="Documentation">📖</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Ajbteves" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-jbteves" title="Maintenance">🚧</a> <a href="https://github.com/ME-ICA/tedana/commits?author=jbteves" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://whitakerlab.github.io"><img src="https://avatars1.githubusercontent.com/u/3626306?v=4?s=100" width="100px;" alt="Kirstie Whitaker"/><br /><sub><b>Kirstie Whitaker</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=KirstieJane" title="Documentation">📖</a> <a href="#projectManagement-KirstieJane" title="Project Management">📆</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3AKirstieJane" title="Reviewed Pull Requests">👀</a> <a href="#talk-KirstieJane" title="Talks">📢</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/monicayao"><img src="https://avatars1.githubusercontent.com/u/35382166?v=4?s=100" width="100px;" alt="Monica Yao"/><br /><sub><b>Monica Yao</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=monicayao" title="Documentation">📖</a> <a href="https://github.com/ME-ICA/tedana/commits?author=monicayao" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="20%"><a href="http://www.fmrwhy.com/"><img src="https://avatars0.githubusercontent.com/u/10141237?v=4?s=100" width="100px;" alt="Stephan Heunis"/><br /><sub><b>Stephan Heunis</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=jsheunis" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://www.linkedin.com/in/benoit-beranger/"><img src="https://avatars0.githubusercontent.com/u/16976839?v=4?s=100" width="100px;" alt="Benoît Béranger"/><br /><sub><b>Benoît Béranger</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=benoitberanger" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/eurunuela"><img src="https://avatars0.githubusercontent.com/u/13706448?v=4?s=100" width="100px;" alt="Eneko Uruñuela"/><br /><sub><b>Eneko Uruñuela</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=eurunuela" title="Code">💻</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Aeurunuela" title="Reviewed Pull Requests">👀</a> <a href="#ideas-eurunuela" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/CesarCaballeroGaudes"><img src="https://avatars1.githubusercontent.com/u/7611340?v=4?s=100" width="100px;" alt="Cesar Caballero Gaudes"/><br /><sub><b>Cesar Caballero Gaudes</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=CesarCaballeroGaudes" title="Documentation">📖</a> <a href="https://github.com/ME-ICA/tedana/commits?author=CesarCaballeroGaudes" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="http://isla.st"><img src="https://avatars2.githubusercontent.com/u/23707851?v=4?s=100" width="100px;" alt="Isla"/><br /><sub><b>Isla</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3AIslast" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/mjversluis"><img src="https://avatars0.githubusercontent.com/u/32125111?v=4?s=100" width="100px;" alt="mjversluis"/><br /><sub><b>mjversluis</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=mjversluis" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://mvaziri.github.io/"><img src="https://avatars2.githubusercontent.com/u/4219325?v=4?s=100" width="100px;" alt="Maryam"/><br /><sub><b>Maryam</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=mvaziri" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/aykhojandi"><img src="https://avatars1.githubusercontent.com/u/38105040?v=4?s=100" width="100px;" alt="aykhojandi"/><br /><sub><b>aykhojandi</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=aykhojandi" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/smoia"><img src="https://avatars3.githubusercontent.com/u/35300580?v=4?s=100" width="100px;" alt="Stefano Moia"/><br /><sub><b>Stefano Moia</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=smoia" title="Code">💻</a> <a href="https://github.com/ME-ICA/tedana/pulls?q=is%3Apr+reviewed-by%3Asmoia" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/ME-ICA/tedana/commits?author=smoia" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://www.notzaki.com/"><img src="https://avatars1.githubusercontent.com/u/9019681?v=4?s=100" width="100px;" alt="Zaki A."/><br /><sub><b>Zaki A.</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/issues?q=author%3AnotZaki" title="Bug reports">🐛</a> <a href="https://github.com/ME-ICA/tedana/commits?author=notZaki" title="Code">💻</a> <a href="https://github.com/ME-ICA/tedana/commits?author=notZaki" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/manfredg"><img src="https://avatars.githubusercontent.com/u/1173430?v=4?s=100" width="100px;" alt="Manfred G Kitzbichler"/><br /><sub><b>Manfred G Kitzbichler</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=manfredg" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/giadaan"><img src="https://avatars.githubusercontent.com/u/118978909?v=4?s=100" width="100px;" alt="giadaan"/><br /><sub><b>giadaan</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=giadaan" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="20%"><a href="https://github.com/bpinsard"><img src="https://avatars.githubusercontent.com/u/1155388?v=4?s=100" width="100px;" alt="Basile"/><br /><sub><b>Basile</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=bpinsard" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/effigies"><img src="https://avatars.githubusercontent.com/u/83442?v=4?s=100" width="100px;" alt="Chris Markiewicz"/><br /><sub><b>Chris Markiewicz</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=effigies" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/goodalse2019"><img src="https://avatars.githubusercontent.com/u/60117796?v=4?s=100" width="100px;" alt="Sarah Goodale"/><br /><sub><b>Sarah Goodale</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=goodalse2019" title="Documentation">📖</a> <a href="#ideas-goodalse2019" title="Ideas, Planning, & Feedback">🤔</a> <a href="#question-goodalse2019" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/martinezeguiluz"><img src="https://avatars.githubusercontent.com/u/129765987?v=4?s=100" width="100px;" alt="Maitane Martinez Eguiluz"/><br /><sub><b>Maitane Martinez Eguiluz</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=martinezeguiluz" title="Code">💻</a></td>
      <td align="center" valign="top" width="20%"><a href="https://github.com/martaarbizu"><img src="https://avatars.githubusercontent.com/u/127724722?v=4?s=100" width="100px;" alt="Marta Arbizu Gómez"/><br /><sub><b>Marta Arbizu Gómez</b></sub></a><br /><a href="https://github.com/ME-ICA/tedana/commits?author=martaarbizu" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
To see what contributors feel they've done in their own words, please see our [contribution recognition page][contribution].

[contribution]: <contributions.md>
