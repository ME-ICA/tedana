Contributing to tedana
======================

This document explains how to set up a development environment for contributing
to tedana and code style conventions we follow within the project.
For a more general guide to the tedana development, please see our
`contributing guide`_.

.. _contributing guide: https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md

Style Guide
-----------

Code
####

Docstrings should follow `numpydoc`_ convention. We encourage extensive
documentation.

The code itself should follow `PEP8`_ convention as much as possible, with at
most about 500 lines of code (not including docstrings) per script.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _PEP8: https://www.python.org/dev/peps/pep-0008/

Pull Requests
#############

We encourage the use of standardized tags for categorizing pull requests.
When opening a pull request, please use one of the following prefixes:

    + **[ENH]** for enhancements
    + **[FIX]** for bug fixes
    + **[TST]** for new or updated tests
    + **[DOC]** for new or updated documentation
    + **[STY]** for stylistic changes
    + **[RF]** for refactoring existing code

Pull requests should be submitted early and often!
If your pull request is not yet ready to be merged, please also include the **[WIP]** prefix.
This tells the development team that your pull request is a "work-in-progress",
and that you plan to continue working on it.
