# tedana

TE-Dependent Analysis (_tedana_) is a Python module for denoising multi-echo fMRI data.

tedana is part of the ME-ICA pipeline, and therefore assumes that you're working with already preprocessed data. If you're in need of a preprocessing pipeline, we recommend [FMRIPREP](https://github.com/poldracklab/fmriprep/), which has been tested for compatibility with multi-echo fMRI data.

## Installation

You'll need to set up a working development environment to use tedana. We provide a Dockerfile for this purpose (see here for [tips on using Docker](https://neurohackweek.github.io/docker-for-scientists/)), or you can set up your environment locally. If you choose the latter, make sure the following packages are installed:

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

Want to learn more about our plans for developing tedana? Check out [our roadmap](https://github.com/emdupre/tedana/projects). Have a question, comment, or suggestion? Open or comment on one of [our issues](https://github.com/emdupre/tedana/issues)!

We ask that all contributions to tedana respect our [code of conduct](https://github.com/emdupre/tedana/blob/master/Code_of_Conduct.md).
