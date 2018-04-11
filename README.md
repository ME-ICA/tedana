# tedana

TE-Dependent Analysis (_tedana_) is a Python module for denoising multi-echo fMRI data.

tedana is part of the ME-ICA pipeline, and therefore assumes that you're working with already preprocessed data. If you're in need of a preprocessing pipeline, we recommend [FMRIPREP](https://github.com/poldracklab/fmriprep/), which has been tested for compatibility with multi-echo fMRI data.

## About

Multi-echo fMRI data collection entails acquires multiple TEs (commonly called [echo times](http://mriquestions.com/tr-and-te.html)) for each collected fMRI volume.
Our signal of interest, Blood Oxygen-Level Dependent or [BOLD signal](http://www.fil.ion.ucl.ac.uk/spm/course/slides10-zurich/Kerstin_BOLD.pdf), is known to decay at a set rate within each fMRI volume.
Collecting multiple echos therefore allows us to infer if components of fMRI signal are BOLD-related or driven by acquisition artifacts, like participant motion.
For a review, see [Kundu et al. (2017), _NeuroImage_](https://paperpile.com/shared/eH3PPu).

In tedana, we combine all collected echos, then decompose the resulting time series into components that can be classified as BOLD or non-BOLD based. This is performed in a series of steps including:

* Principle components analysis
* Independent components analysis
* Component classification

More information and documentation can be found at https://tedana.readthedocs.io/.

## Installation

You'll need to set up a working development environment to use tedana. We provide a Dockerfile for this purpose (check out [tips on using Docker](https://neurohackweek.github.io/docker-for-scientists/)), or you can set up your environment locally. If you choose the latter, make sure the following packages are installed:

mdp  
nilearn  
nibabel>=2.1.0  
numpy  
pybids>=0.4.0  
scikit-learn  

tedana will eventually be hosted on PyPi. In the mean time, you can still install it with `pip` using:

```
pip install https://github.com/ME-ICA/tedana/archive/master.tar.gz
```

## Development

We :yellow_heart: new contributors ! To get started, check out [our contributing guidelines](https://github.com/emdupre/tedana/blob/master/CONTRIBUTING.md).

Want to learn more about our plans for developing tedana ? Check out [our roadmap](https://github.com/emdupre/tedana/projects). Have a question, comment, or suggestion ? Open or comment on one of [our issues](https://github.com/emdupre/tedana/issues) !

We ask that all contributions to tedana respect our [code of conduct](https://github.com/emdupre/tedana/blob/master/Code_of_Conduct.md).

### Mozilla Global Sprint (10-11 May, 2018)

This year, tedana will be participating in the [Mozilla Global Sprint](https://foundation.mozilla.org/opportunity/global-sprint/) !
Look out for issues tagged `global-sprint` for good places to get started during the sprint.
