Contributing to tedana
======================

This document explains how to set up a development environment for contributing to tedana
and code style conventions we follow within the project.
For a more general guide to the tedana development, please see our `contributing guide`_.

Development in docker_ is encouraged, for the sake of consistency and portability.
By default, work should be built off of ``emdupre/meica-docker:0.0.3``
(see the :doc:`installation` guide for grabbing the docker image).

.. _contributing guide: https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md
.. _docker: https://www.docker.com/

Style Guide
-----------

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
