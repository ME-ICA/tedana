Contributing to tedana
======================

This document explains contributing to ``tedana`` at a very high level,
with a focus on project governance and development philosophy.
For a more practical guide to the tedana development, please see our
`contributing guide`_.

.. _contributing guide: https://github.com/ME-ICA/tedana/blob/master/CONTRIBUTING.md

Code of conduct
```````````````

All ``tedana`` community members are expected to follow our code of conduct
during any interaction with the project. `The full code of conduct is here`_
That includes---but is not limited to---online conversations,
in-person workshops or development sprints, and when giving talks about the software.

As stated in the code, severe or repeated violations by community members may result in exclusion
from collective decision-making and rejection of future contributions to the ``tedana`` project.

.. _The full code of conduct is here: https://github.com/ME-ICA/tedana/blob/master/CODE_OF_CONDUCT.md

tedana's development philosophy
--------------------------------------

In contributing to any open source project,
we have found that it is hugely valuable to understand the core maintainers' development philosophy.
In order to aid other contributors in on-boarding to ``tedana`` development,
we have therefore laid out our shared opinion on several major decision points.
These are:

#. :ref:`exposing options to the user`,
#. :ref:`prioritizing project developments`,
#. :ref:`backwards compatibility with meica`,
#. :ref:`future-proofing for continuous development`, and
#. :ref:`when to release new software versions`


.. _exposing options to the user:

Which options are available to users?
`````````````````````````````````````

The ``tedana``  developers are committed to providing useful and interpretable outputs
for a majority of use cases.

In doing so, we have made a decision to embrace defaults which support the broadest base of users.
For example, the choice of an independent component analysis (ICA) cost function is part of the
``tedana`` pipeline that can have a significant impact on the results and is difficult for
individual researchers to form an opinion on.

The ``tedana`` "opinionated approach" is therefore to provide reasonable defaults and to hide some
options from the top level workflows.

This decision has two key benefits:

1. By default, users should get high quality results from running the pipelines, and
2. The work required of the ``tedana``  developers to maintain the project is more focused
   and somewhat restricted.

It is important to note that ``tedana``  is shipped under `an LGPL2 license`_ which means that
the code can---at all times---be cloned and re-used by anyone for any purpose.

"Power users" will always be able to access and extend all of the options available.
We encourage those users to feed back their work into ``tedana``  development,
particularly if they have good evidence for updating the default values.

We understand that it is possible to build the software to provide more
options within the existing framework, but we have chosen to focus on `the 80 percent use cases`_.

You can provide feedback on this philosophy through any of the channels
listed on the ``tedana`` :ref:`support_ref` page.

.. _an LGPL2 license: https://github.com/ME-ICA/tedana/blob/master/LICENSE
.. _the 80 percent use cases: https://en.wikipedia.org/wiki/Pareto_principle#In_software


.. _prioritizing project developments:

Structuring project developments
````````````````````````````````

The ``tedana``  developers have chosen to structure ongoing development around specific goals.
When implemented successfully, this focuses the direction of the project and helps new contributors
prioritize what work needs to be completed.

We have outlined our goals for ``tedana`` in our :doc:`roadmap`,
which we encourage all contributors to read and give feedback on.
Feedback can be provided through any of the channels listed on our :ref:`support_ref` page.

In order to more directly map between our :doc:`roadmap` and ongoing `project issues`_,
we have also created `milestones in our github repository`_.

.. _project issues: https://github.com/ME-ICA/tedana/issues
.. _milestones in our github repository: https://github.com/me-ica/tedana/milestones

This allows us to:

1. Label individual issues as supporting specific aims, and
2. Measure progress towards each aim's concrete deliverable(s).


.. _backwards compatibility with meica:

Is ``tedana`` backwards compatible with MEICA?
``````````````````````````````````````````````

The short answer is No.

There are two main reasons why.
The first is that `mdp`_, the python library used to run the ICA decomposition core to the original
MEICA method, is no longer supported.

In November 2018, the ``tedana`` developers made the decision to switch to `scikit-learn`_ to
perform these analyses.
``scikit-learn`` is well supported and under long term development.
``tedana`` will be more stable and have better performance going forwards as a result of
this switch, but it also means that exactly reproducing previous MEICA analyses is not possible.

The other reason is that the core developers have chosen to look forwards rather than maintaining
an older code base.
As described in the :ref:`governance` section, ``tedana`` is maintained by a small team of
volunteers with limited development time.
If you'd like to use MEICA as has been previously published the code is available on
`bitbucket`_ and freely available under a LGPL2 license.

.. _mdp: http://mdp-toolkit.sourceforge.net
.. _scikit-learn: http://scikit-learn.org/stable
.. _bitbucket: https://bitbucket.org/prantikk/me-ica


.. _future-proofing for continuous development:

How does ``tedana`` future-proof its development?
`````````````````````````````````````````````````

``tedana``  is a reasonably young project that is run by volunteers.
No one involved in the development is paid for their time.
In order to focus our limited time, we have made the decision to not let future possibilities limit
or over-complicate the most immediately required features.
That is, to `not let the perfect be the enemy of the good`_.

.. _not let the perfect be the enemy of the good: https://en.wikipedia.org/wiki/Perfect_is_the_enemy_of_good

While this stance will almost certainly yield ongoing refactoring as the scope of the software expands,
the team's commitment to transparency, reproducibility, and extensive testing
mean that this work should be relatively manageable.

We hope that the lessons we learn building something useful in the short term will be
applicable in the future as other needs arise.


.. _when to release new software versions:

When to release a new version
`````````````````````````````

In the broadest sense, we have adopted a "you know it when you see it" approach
to releasing new versions of the software.

To try to be more concrete, if a change to the project substantially changes the user's experience
of working with ``tedana``, we recommend releasing an updated version.
Additional functionality and bug fixes are very clear opportunities to release updated versions,
but there will be many other reasons to update the software as hosted on `PyPi`_.

.. _PyPi: https://pypi.org/project/tedana/

To give two concrete examples of slightly less obvious cases:

1. A substantial update to the documentation that makes ``tedana``  easier to use **would** count as
a substantial change to ``tedana``  and a new release should be considered.

2. In contrast, updating code coverage with additional unit tests does not affect the
**user's** experience with ``tedana``  and therefore does not require a new release.

Any member of the ``tedana``  community can propose that a new version is released.
They should do so by opening an issue recommending a new release and giving a
1-2 sentence explanation of why the changes are sufficient to update the version.
More information about what is required for a release to proceed is available
in the :ref:`release checklist`.


Release Checklist
`````````````````

This is the checklist of items that must be completed when cutting a new release of tedana.
These steps can only be completed by a project maintainer, but they are a good resource for
releasing your own Python projects!

    #. All continuous integration must be passing and docs must be building successfully.
    #. Create a new release, using the GitHub `guide for creating a release on GitHub`_.
       `Release-drafter`_ should have already drafted release notes listing all
       changes since the last release; check to make sure these are correct.

We have set up tedana so that releases automatically mint a new DOI with Zenodo;
a guide for doing this integration is available `here`_.
We have also set up the repository so that tagged releases automatically deploy
to PyPi (for pip installation).

.. _`guide for creating a release on GitHub`: https://help.github.com/articles/creating-releases/
.. _`Release-drafter`: https://github.com/apps/release-drafter
.. _here: https://guides.github.com/activities/citable-code/
