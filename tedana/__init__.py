# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tedana: A Python package for TE-dependent analysis of multi-echo data."""
import warnings

from tedana.__about__ import __copyright__, __credits__, __packagename__, __version__

# cmp is not used, so ignore nipype-generated warnings
warnings.filterwarnings("ignore", r"cmp not installed")

__all__ = [
    "__copyright__",
    "__credits__",
    "__packagename__",
    "__version__",
]
