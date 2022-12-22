# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
tedana: A Python package for TE-dependent analysis of multi-echo data.
"""

import warnings

from ._version import get_versions

__version__ = get_versions()["version"]

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")
warnings.filterwarnings("ignore", r"Failed to import duecredit due to No module named 'duecredit'")

del get_versions
