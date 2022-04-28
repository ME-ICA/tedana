# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
tedana: A Python package for TE-dependent analysis of multi-echo data.
"""

import warnings

from .due import BibTeX, Doi, due
from ._version import get_versions

__version__ = get_versions()["version"]

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

# Citation for the package JOSS paper.
due.cite(
    Doi("10.21105/joss.03669"),
    description="Publication introducing tedana.",
    path="tedana",
    cite_module=True,
)

# Citation for the algorithm.
due.cite(
    Doi("10.1016/j.neuroimage.2011.12.028"),
    description="Introduces MEICA and tedana.",
    path="tedana",
    cite_module=True,
)
due.cite(
    Doi("10.1073/pnas.1301725110"),
    description="Improves MEICA and tedana.",
    path="tedana",
    cite_module=True,
)

# Citation for package version.
due.cite(
    Doi("10.5281/zenodo.1250561"),
    description="The tedana package",
    version=__version__,
    path="tedana",
    cite_module=True,
)

del get_versions
