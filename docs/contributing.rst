Contributing to tedana
======================

This document explains how to set up a development environment for contributing
to tedana and code style conventions we follow within the project.
For a more general guide to the tedana development, please see our
`contributing guide`_. Please also follow our `code of conduct`_.

.. _contributing guide: https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md
.. _code of conduct: https://github.com/ME-ICA/tedana/blob/master/Code_of_Conduct.md


Style Guide
-----------

Code
````

Docstrings should follow `numpydoc`_ convention. We encourage extensive
documentation.

The code itself should follow `PEP8`_ convention as much as possible, with at
most about 500 lines of code (not including docstrings) per script.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html
.. _PEP8: https://www.python.org/dev/peps/pep-0008/

Pull Requests
`````````````

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

Release Checklist
`````````````````

This is the checklist of items that must be completed when cutting a new release of tedana.
These steps can only be completed by a project maintainer, but they a good resource for
releasing your own Python projects!

    #. All continuous integration must be passing and docs must be building successfully
    #. Record all merged pull requests since the last release; these will be included in
       the change log for the new release.
    #. Create a new release on GitHub, logging the changes since the last release
       in the release notes. Here's a `guide for creating a release on GitHub`_.
    #. Pulling from the ``master`` branch, locally build a new copy of tedana and
        `upload it to PyPi`_.
    #. Update the `Read the Docs landing page`_ to reference the lastest Zenodo DOI.
       We have set up tedana so that releases automatically mint a new DOI with Zenodo;
       a guide for doing this integration is available `here`_.

    .. _`upload it to PyPi`: https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives
    .. _`guide for creating a release on GitHub`: https://help.github.com/articles/creating-releases/
    .. _`Read the Docs landing page`: https://tedana.readthedocs.io/
    .. _here: https://guides.github.com/activities/citable-code/
