# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""
from pathlib import Path
readme_path = Path(__file__).parent.parent.joinpath("README.md")

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = 'tedana developers'
__copyright__ = 'Copyright 2018, tedana developers'
__credits__ = ['Elizabeth DuPre', 'Prantik Kundu', 'Ross Markello',
               'Taylor Salo', 'Kirstie Whitaker', 'Dan Handwerker']
__license__ = 'LGPL 2.1'
__maintainer__ = 'Elizabeth DuPre'
__email__ = 'emd222@cornell.edu'
__status__ = 'Prototype'
__url__ = 'https://github.com/me-ica/tedana'
__packagename__ = 'tedana'
__description__ = ('TE-Dependent Analysis (tedana) of multi-echo functional '
                   'magnetic resonance imaging (fMRI) data.')
__longdesc__ = readme_path.open().read()

DOWNLOAD_URL = (
    'https://github.com/ME-ICA/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))

REQUIRES = [
    'bokeh',
    'matplotlib',
    'nibabel>=2.5.1',
    'nilearn>=0.5.2',
    'numpy>=1.15',
    'pandas',
    'scikit-learn>=0.22',
    'scipy>=1.3.3',
    'threadpoolctl'
]

TESTS_REQUIRES = [
    'codecov',
    'coverage<5.0',
    'flake8>=3.7',
    'pytest',
    'pytest-cov',
    'requests',
]

EXTRA_REQUIRES = {
    'dev': ['versioneer'],
    'doc': [
        'sphinx>=1.5.3',
        'sphinx_rtd_theme',
        'sphinx-argparse',
    ],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = list(set([
    v for deps in EXTRA_REQUIRES.values() for v in deps]))

# Supported Python versions using PEP 440 version specifiers
# Should match the same set of Python versions as classifiers
PYTHON_REQUIRES = ">=3.5"

# Package classifiers
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
]
