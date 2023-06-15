######################
Contributing to tedana
######################

This document explains contributing to ``tedana`` at a very high level,
with a focus on project governance and development philosophy.
For a more practical guide to the tedana development, please see our
`contributing guide`_.

.. _contributing guide: https://github.com/ME-ICA/tedana/blob/main/CONTRIBUTING.md


***************
Code of conduct
***************

All ``tedana`` community members are expected to follow our code of conduct
during any interaction with the project. `The full code of conduct is here`_.
That includes---but is not limited to---online conversations,
in-person workshops or development sprints, and when giving talks about the software.

As stated in the code, severe or repeated violations by community members may result in exclusion
from collective decision-making and rejection of future contributions to the ``tedana`` project.

.. _The full code of conduct is here: https://github.com/ME-ICA/tedana/blob/main/CODE_OF_CONDUCT.md


*******************************
tedana's development philosophy
*******************************

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
=====================================

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

.. _an LGPL2 license: https://github.com/ME-ICA/tedana/blob/main/LICENSE
.. _the 80 percent use cases: https://en.wikipedia.org/wiki/Pareto_principle#In_software


.. _prioritizing project developments:

Structuring project developments
================================

The ``tedana``  developers have chosen to structure ongoing development around specific goals.
When implemented successfully, this focuses the direction of the project and helps new contributors
prioritize what work needs to be completed.

We have outlined our goals for ``tedana`` in our :doc:`roadmap`,
which we encourage all contributors to read and give feedback on.
Feedback can be provided through any of the channels listed on our :ref:`support_ref` page.

In order to more directly map between our :doc:`roadmap` and ongoing `project issues`_,
we have also created `milestones in our github repository`_.

This allows us to:

1. Label individual issues as supporting specific aims, and
2. Measure progress towards each aim's concrete deliverable(s).

.. _project issues: https://github.com/ME-ICA/tedana/issues
.. _milestones in our github repository: https://github.com/me-ica/tedana/milestones


.. _backwards compatibility with meica:

Is ``tedana`` backwards compatible with MEICA?
==============================================

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
As described in the `governance`_ section, ``tedana`` is maintained by a small team of
volunteers with limited development time.
If you'd like to use MEICA as has been previously published the code is available on
`bitbucket`_ and freely available under a LGPL2 license.

.. _mdp: http://mdp-toolkit.sourceforge.net
.. _scikit-learn: http://scikit-learn.org/stable
.. _bitbucket: https://bitbucket.org/prantikk/me-ica


.. _future-proofing for continuous development:

How does ``tedana`` future-proof its development?
=================================================

``tedana``  is a reasonably young project that is run by volunteers.
No one involved in the development is paid for their time.
In order to focus our limited time, we have made the decision to not let future possibilities limit
or over-complicate the most immediately required features.
That is, to `not let the perfect be the enemy of the good`_.

While this stance will almost certainly yield ongoing refactoring as the scope of the software expands,
the team's commitment to transparency, reproducibility, and extensive testing
mean that this work should be relatively manageable.

We hope that the lessons we learn building something useful in the short term will be
applicable in the future as other needs arise.

.. _not let the perfect be the enemy of the good: https://en.wikipedia.org/wiki/Perfect_is_the_enemy_of_good


.. _when to release new software versions:

When to release a new version
=============================

In the broadest sense, we have adopted a "you know it when you see it" approach
to releasing new versions of the software.

To try to be more concrete, if a change to the project substantially changes the user's experience
of working with ``tedana``, we recommend releasing an updated version.
Additional functionality and bug fixes are very clear opportunities to release updated versions,
but there will be many other reasons to update the software as hosted on `PyPi`_.

To give two concrete examples of slightly less obvious cases:

1. A substantial update to the documentation that makes ``tedana``  easier to use **would** count as
a substantial change to ``tedana``  and a new release should be considered.

2. In contrast, updating code coverage with additional unit tests does not affect the
**user's** experience with ``tedana``  and therefore does not require a new release.

Any member of the ``tedana``  community can propose that a new version is released.
They should do so by opening an issue recommending a new release and giving a
1-2 sentence explanation of why the changes are sufficient to update the version.
More information about what is required for a release to proceed is available
in the :ref:`release-checklist`.

.. _PyPi: https://pypi.org/project/tedana/


.. _release-checklist:

Release Checklist
=================

This is the checklist of items that must be completed when cutting a new release of tedana.
These steps can only be completed by a project maintainer, but they are a good resource for
releasing your own Python projects!

    #. All continuous integration must be passing and docs must be building successfully.
    #. Create a new release, using the GitHub `guide for creating a release on GitHub`_.
       `Release-drafter`_ should have already drafted release notes listing all
       changes since the last release; check to make sure these are correct.

  .. warning::
    Do not directly release the `Release-drafter`_-generated release draft.
    You **must** copy the contents of the auto-generated draft to a new draft to be released.
    `Release-drafter`_-generated releases **will not** deploy to PyPi.

We have set up tedana so that releases automatically mint a new DOI with Zenodo;
a guide for doing this integration is available in `GitHub's citable code guide`_.
We have also set up the repository so that tagged releases automatically deploy
to PyPi (for pip installation).

.. _`guide for creating a release on GitHub`: https://help.github.com/articles/creating-releases/
.. _`Release-drafter`: https://github.com/apps/release-drafter
.. _`GitHub's citable code guide`: https://guides.github.com/activities/citable-code/


********************
Developer guidelines
********************

This section is intended to guide users through making making changes to
``tedana``'s codebase, in particular working with tests.
The worked example also offers some guidelines on approaching testing when
adding new functions.
Please check out our `contributing guide`_ for getting started.


Monthly Developer Calls
=======================

We run monthly developer calls via Zoom.
You can see the schedule via the tedana `google calendar`_.

Everyone is welcome.
We look forward to meeting you there!


Adding and Modifying Tests
==========================

Testing is an important component of development.
For simplicity, we have migrated all tests to ``pytest``.
There are two basic kinds of tests: unit and integration tests.
Unit tests focus on testing individual functions, whereas integration tests focus on making sure
that the whole workflow runs correctly.


Unit Tests
----------

For unit tests, we try to keep tests from the same module grouped into one file.
Make sure the function you're testing is imported, then write your test.
Good tests will make sure that edge cases are accounted for as well as common cases.
You may also use ``pytest.raises`` to ensure that errors are thrown for invalid inputs to a
function.


Integration Tests
-----------------

Adding integration tests is relatively rare.
An integration test will be a complete multi-echo dataset called with some set of options to ensure
end-to-end pipeline functionality.
These tests are relatively computationally expensive but aid us in making sure the pipeline is
stable during large sets of changes.
If you believe you have a dataset that will test ``tedana`` more completely, please open an issue
before attempting to add an integration test.
After securing the appropriate permission from the dataset owner to share it with ``tedana``, you
can use the following procedure:

(1) Make a ``tar.gz`` file which will unzip to be only the files you'd like to
run a workflow on.
You can do this with the following, which would make an archive ``my_data.tar.gz``:

.. code-block:: bash

    tar czf my_data.tar.gz my_data/*.nii.gz

(2) Run the workflow with a known-working version, and put the outputs into a text file inside
``$TEDANADIR/tedana/tests/data/``, where ``TEDANADIR`` is your local ``tedana repository``.
We encourage using the convention ``<DATASET>_<n_echoes>_echo_outputs.txt``, appending ``verbose``
to the filename if the integration test uses ``tedana`` in the verbose mode.

(3) Write a test function in ``test_integration.py``.
To write the test function you can follow the model of our `five echo set`_, which takes the following steps:

1. Check if a pytest user is skipping integration, skip if so
#. Use ``download_test_data`` to retrieve the test data from OSF
#. Run a workflow
#. Use ``resources_filename`` and ``check_integration_outputs`` to compare your expected output to
   actual output.

(4) If you need to upload new data, you will need to contact the maintainers and ask them to either add
it to the `tedana OSF project`_ or give you permission to add it.

(5) Once you've tested your integration test locally and it is working, you will need to add it to the
CircleCI config and the ``Makefile``.
Following the model of the three-echo and five-echo sets, define a name for your integration test
and on an indented line below put

.. code-block:: bash

    @py.test --cov-append --cov-report term-missing --cov=tedana -k TEST

with ``TEST`` your test function's name.
This call basically adds code coverage reports to account for the new test, and runs the actual
test in addition.

(6) Using the five-echo set as a template, you should then edit ``.circlec/config.yml`` to add your
test, calling the same name you define in the ``Makefile``.


Viewing CircleCI Outputs
------------------------

If you need to take a look at a failed test on CircleCI rather than locally, you can use the
following block to retrieve artifacts (see CircleCI documentation here_)

.. code-block:: bash

    export CIRCLE_TOKEN=':your_token'

    curl https://circleci.com/api/v1.1/project/:vcs-type/:username/:project/$build_number/artifacts?circle-token=$CIRCLE_TOKEN \
       | grep -o 'https://[^"]*' \
       | sed -e "s/$/?circle-token=$CIRCLE_TOKEN/" \
       | wget -v -i -

To get a CircleCI token, follow the instructions for `getting one`_.
You cannot do this unless you are part of the ME-ICA/tedana organization.
If you don't want all of the artifacts, you can go to the test details and use the browser to
manually select the files you would like.


Worked Example
==============

Suppose we want to add a function in ``tedana`` that creates a file called ```hello_world.txt`` to
be stored along the outputs of the ``tedana`` workflow.

First, we merge the repository's ``main`` branch into our own to make sure we're up to date, and
then we make a new branch called something like ``feature/say_hello``.
Any changes we make will stay on this branch.
We make the new function and call it ``say_hello`` and locate this function inside of ``io.py``.
We'll also need to make a unit test.
(Some developers actually make the unit test before the new function; this is a great way to make
sure you don't forget to create it!)
Since the function lives in ``io.py``, its unit test should go into ``test_io.py``.
The job of this test is exclusively to tell if the function we wrote does what it claims to do
without errors.
So, we define a new function in ``test_io.py`` that looks something like this:

.. code-block:: python

    def test_say_hello():
        # run the function
        say_hello()
        # test the function
        assert op.exists('hello_world.txt')
        # clean up
        os.remove('hello_world.txt')

We should see that our unit test is successful via

.. code-block:: bash

    pytest $TEDANADIR/tedana/tests/test_io.py -k test_say_hello

If not, we should continue editing the function until it passes our test.
Let's suppose that suddenly, you realize that what would be even more useful is a function that
takes an argument, ``place``, so that the output filename is actually ``hello_PLACE``, with
``PLACE`` the value passed and ``'world'`` as the default value.
We merge any changes from the upstream main branch into our branch via

.. code-block:: bash

    git checkout feature/say_hello
    git fetch upstream main
    git merge upstream/main

and then begin work on our test.
We need to our unit test to be more complete, so we update it to look more like the following,
adding several cases to make sure our function is robust to the name supplied:

.. code-block:: python

    def test_say_hello():
        # prefix of all files to be checked
        prefix = 'hello_'
        # suffix of all files to be checked
        suffix  = '.txt'
        # run the function with several cases
        for x in ['world', 'solar system', 'galaxy', 'universe']:
            # current test name
            outname = prefix + x + suffix
            # call the function
            say_hello(x)
            # test the function
            assert op.exists(outname)
            # clean up from this call
            os.remove(outname)

Once that test is passing, we may need to adjust the integration test.
Our program creates a file, ``hello_world.txt``, which the older version would not have produced.
Therefore, we need to add the file to ``$TEDANADIR/tedana/tests/data/tedana_outputs.txt`` and its
counterpart, R2-D2-- uh, we mean, ``tedana_outputs_verbose.txt``.
With that edit complete, we can run the full ``pytest`` suite via

.. code-block:: bash

    pytest $TEDANADIR/tedana/tests

Once that filename is added, all of the tests should be passing and we should open a PR to have our
change reviewed.

From here, others working on the project may request changes and we'll have to make sure that our
tests are kept up to date with any changes made as we did before updating the unit test.
For example, if a new parameter is added, ``greeting``, with a default of ``hello``, we'll need to
adjust the unit test.
However, since this doesn't change the typical workflow of ``tedana``, there's no need to change
the integration test; we're still matching the original filename.
Once we are happy with the changes and some members of ``tedana`` have approved the changes, our
changes will be merged!

We should then do the following cleanup with our git repository:

.. code-block:: bash

    git checkout main
    git fetch upstream main
    git merge upstream/main
    git branch -d feature/say_hello
    git push --delete origin feature/say_hello

and we're good to go!

.. _`tedana OSF project`: https://osf.io/bpe8h/
.. _git: https://git-scm.com/
.. _`git pro`: https://git-scm.com/book/en/v2
.. _Fork: https://help.github.com/en/github/getting-started-with-github/fork-a-repo
.. _`pull request`: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request
.. _GitKraken: https://www.gitkraken.com/
.. _`GitHub Desktop`: https://desktop.github.com/
.. _SourceTree: https://www.sourcetreeapp.com/
.. _`GitHub UI`: https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository
.. _this: https://github.com/ME-ICA/tedana/tree/main/docs
.. _ReStructuredText: http://docutils.sourceforge.net/rst.html#user-documentation
.. _`five echo set`: https://github.com/ME-ICA/tedana/blob/37368f802f77b4327fc8d3f788296ca0f01074fd/tedana/tests/test_integration.py#L71-L95
.. _here: https://circleci.com/docs/2.0/artifacts/#downloading-all-artifacts-for-a-build-on-circleci
.. _`getting one`: https://circleci.com/docs/2.0/managing-api-tokens/?gclid=CjwKCAiAqqTuBRBAEiwA7B66heDkdw6l68GAYAHtR2xS1xvDNNUzy7l1fmtwQWvVN0OIa97QL8yfhhoCejoQAvD_BwE#creating-a-personal-api-token
.. _`google calendar`: https://calendar.google.com/calendar/embed?src=pl6vb4t9fck3k6mdo2mok53iss%40group.calendar.google.com
.. _`contributing guide`: https://github.com/ME-ICA/tedana/blob/main/CONTRIBUTING.md


**********
Governance
**********

Governance is a hugely important part of any project.
It is especially important to have clear processes and communication channels
for open source projects that rely on a distributed network of volunteers,
such as ``tedana``.


Overview
========

Tedana is a relatively small open source project that requires specialized
knowledge in multiple domains.
This leads to several challenges.
No one
person on the current tedana development team has a combination of
available time plus expertise in collaborative software development, MRI
physics, and advanced data processing methods to assume a primary project
leader role.
Even if such a person was interested, it may not benefit the
project to overly rely on the existence of one person.
Instead, we developed the
following system with several goals in mind:

- Grow the community.
- Strive for consensus.
- Provide a path for when consensus cannot be achieved.
- Minimize the administrative burden.
- Maximize the `bus factor`_ of the project.
- Acknowledge the leadership and time multiple people contribute to our
  community without demanding more time than any individual can offer.
  Dividing leadership responsibilities into multiple smaller roles also
  makes it easier to encourage new people to take on a leadership role
  without fearing that too much work will be required of them.
- Openness as a priority:

  - Promote open discussions.
  - Openness is critical to building trust with the broader community
  - Openness provides a mechanism for non-leaders to identify and address
    oversights or mistakes
  - Openness provides a smoother transition to onboard future leaders
  - Leadership meetings should be open and notes should be shared unless
    there are discussions about sensitive personal matters.

This governance structure is a work-in-progress.
We welcome both people
who want to take on a leadership role as well as ideas to improve
this structure.


Leadership
==========

Contributor
  A contributor is someone who has made a contribution to tedana.
  A contribution can be code, documentation, or conceptual.
  All contributors are listed in the `all-contributors file`_.
  The community decides on the content of this file using the same process
  as any other change to the `Repository`_ (see below) allowing the
  meaning of "Contributor" to evolve independently of the Decision-making
  rules.
  Contributors also have the option to be added to the Zenodo file which
  may be used for authorship credit for tedana.


Maintainer
  A Maintainer is responsible for the long term health of the project and
  the community.
  Maintainers have additional authority (see `Decision Making Process`_)
  helping them to resolve conflicts and increase the pace of the
  development when necessary.
  Any maintainer can remove themselves.
  Any contributor can become a maintainer by request, or by nomination of
  a current maintainer,  and with the support of the majority of the
  current maintainers.

  Current Maintainers:

  +-------------------------------------------+
  | `Logan Dowdle`_ (@dowdlelt)               |
  +-------------------------------------------+
  | `Elizabeth DuPre`_ (@emdupre)             |
  +-------------------------------------------+
  | `Javier Gonzalez-Castillo`_ (@javiergcas) |
  +-------------------------------------------+
  | `Dan Handwerker`_ (@handwerkerd)          |
  +-------------------------------------------+
  | `Taylor Salo`_ (@tsalo)                   |
  +-------------------------------------------+
  | `Eneko Uruñuela`_ (@eurunuela)            |
  +-------------------------------------------+

Steering committee
  The :ref:`Steering Committee` is made up of a subset of maintainers who
  help guide the project.

  Current Steering Committee members:

  +--------------------------------------+
  | `Logan Dowdle`_ (@dowdlelt)          |
  +--------------------------------------+
  | `Elizabeth DuPre`_ (@emdupre)        |
  +--------------------------------------+
  | `Dan Handwerker`_ (@handwerkerd)     |
  +--------------------------------------+
  | `Taylor Salo`_ (@tsalo)              |
  +--------------------------------------+
  | `Eneko Uruñuela`_ (@eurunuela)       |
  +--------------------------------------+

Focused Leadership Roles
  We have identified key responsibilities or skills that help advance
  tedana development and created roles for each of these responsibilities.
  One person can fill more than one role and more than one person can
  decide to share or split the responsibilities of a role.
  Any contributor can propose the creation of new focused leadership roles.
  A person can take on a leadership role without being a Maintainer or
  Steering Committee member

  - | Task manager & record keeper: `Dan Handwerker`_

    |   Helps write & keep track of notes from meetings
    |   Keeps track of issues or items that should be addressed
    |   Follows up with people who volunteered to address an item or
        alerts the broader community of known tasks that could use a
        volunteer
  - | MR physics leader: `César Caballero-Gaudes`_

    |   Someone who can make sure calculations fit within our
        understanding of MR physics
    |   Someone who can either answer MRI physics questions related to
        multi-echo or direct people to where they can find answers
  - | Processing algorithms leaders: `Eneko Uruñuela`_ (Decomposition) &  `Dan Handwerker`_ (Metrics & Decision Tree)

    |   Someone who can make sure algorithms are appropriately implemented
        (or knows enough to delegate to someone who can make sure
        implementation is good)
    |   Someone who can either answer processing algorithm questions or
        direct people to where they can find answers
  - | Collaborative programming leaders: `Elizabeth DuPre`_
    |   Helps make sure tedana is following best practices for Python code
        design, testing, and communication for issues/pull requests etc.
  - | New contributors leader: `Taylor Salo`_
    |   Leads efforts to make contributor documentation more welcoming
    |   Is a point of contact for potential contributors to make them feel
        welcome and direct them to relevant resources or issues
  - | Multi-echo fMRI support leader: `Logan Dowdle`_
    |   Monitors places where people may ask questions about tedana or
        multi-echo fMRI and tries to find someone to answer those questions
  - | Enforcer(s) of the `code of conduct`_: `Elizabeth DuPre`_ &  `Dan Handwerker`_ & `Stefano Moia`_
    |   People someone can go to if they want to report a code of conduct
        violation


Changing leaders
----------------

Any leader can remove themselves for a role at any time and open up a call
for a new self-nomination.
Anyone can request to take on a leadership role at any time.
Once per year, there should be an explicit call to the larger contributor
community asking if anyone wants to self nominate for a leadership role.
If individuals cannot reach consensus on who steps back and who assumes
new roles, then a majority vote of contributors from the previous 3 years
will assign people to roles where there are conflicts.

If there are concerns with a tedana leader, any enforcer of the code of
conduct can ask anyone to step down from a leadership role.
If a person refuses to step down, then an enforcer of the code of conduct
will consult with the other code of conduct enforcers.
If they reach a concensus that a person shouldn't have a tedana leadership
position, then they should be removed.
If a code of conduct enforcer has a conflict of interest, then the
remaining code of conduct enforcers will identify someone without a
conflict to include in deliberations.


Decision Making Process
=======================

The rules outlined below are inspired by the
`decision-making rules for the BIDS standard <https://github.com/bids-standard/bids-specification/blob/master/DECISION-MAKING.md>`_,
which in turn were inspired by the
`lazy consensus system used in the Apache Foundation <https://www.apache.org/foundation/voting.html>`_,
and heavily depend on the
`GitHub Pull Request review system <https://help.github.com/articles/about-pull-requests/>`_.

1. Potential modifications to the Repository should first be proposed via
   an Issue.
2. Every modification (including a correction of a typo, adding a new
   Contributor, an extension or others) or proposal to release a new
   version needs to be done via a Pull Request (PR) to the Repository.
3. Anyone can open an Issue or a PR (this action is not limited to
   Contributors).
4. A PR is eligible to be merged if and only if these conditions are met:

   a) The PR features at least two
      `Reviews that Approve <https://help.github.com/articles/about-pull-request-reviews/#about-pull-request-reviews>`_
      the PR of which neither is the author of the PR.
      The reviews should be made after the last commit in the PR
      (equivalent to
      `Stale review dismissal <https://help.github.com/articles/enabling-required-reviews-for-pull-requests/>`_
      option on GitHub).
      If a second review requests minor changes after
      another reviewer approved the PR, the first review does not need
      to re-review.
   b) Does not feature any
      `Reviews that Request changes <https://help.github.com/articles/about-required-reviews-for-pull-requests/>`_.
      That is, if someone asked for changes, the PR should not be merged
      just because two other people approve it.
   c) Is not a Draft PR.
      That is, the PR author says it is ready for review.
   d) Passes all automated tests.
   e) Is not proposing a new release.
   f) The steering committee has not added extra restrictions.
      For example, if a PR is a non-trival change, the steering committee
      can create a system to get feedback from more than just two reviewers
      before merging.
5. After consultation with contributors, the steering committee can decide
   to merge any PR - even if it's not eligible to merge according to Rule 4.
6. Anyone can Review a PR and request changes.
   If a community member requests changes they need to provide an
   explanation regarding what changes should be made and justification of
   their importance.
   Reviews requesting changes can also be used to request more time to
   review a PR.
7. A reviewer who requested changes can dismiss their own review, if they
   decide their requested changes are no longer necessary, or approve
   changes that address the issue underlying their change request.
8. If the author of a PR and a reviewer who requests changes cannot find a
   solution that would lead to:

   (1) The author closing the PR without merging
   (2) The reviewer accepting requested changes or
   (3) The reviewer dismissing their review, so that the PR can be approved and
       merged, then the disagreement will be resolved with a vote.
9. Rules governing voting:

   a) A vote can be triggered by any Maintainer, but only after 5 working
      days from the time a Review Requesting Changes is made.
      A PR can only have one open vote at a time.
      If disagreements over a PR results in more than one
      vote, the Steering Committee has the authority to create a voting
      process to help resolve disagreements in a more efficient and
      respectful manner.
   b) Only Contributors can vote and each Contributor gets one vote.
   c) A vote ends after 15 working days or when all Contributors have
      voted or abstained (whichever comes first).
   d) A vote freezes the PR - no new commits or Reviews Requesting Changes
      can be added to it while a vote is ongoing.
      If a commit is accidentally made during that period it should be
      reverted.
      Comments are allowed.
   e) The quorum for a vote is five votes.
   f) The outcome of the vote is decided based on a simple majority.


.. _Steering Committee:

Steering Committee
------------------

The steering committee steers.
The goal of the steering committee is to help guide the direction of the
project.
Decisions in the steering committee will focus on how to present project
issues to the broader community in a clear way rather than making project
decisions without community input.

The steering committee can decide:

- An issue should be prioritized for wider communal discussion.
- A pull request requires more discussion or reviews than standard before
  merging.
- How a breaking change (something that changes existing user function calls
  or program outputs) will be presented to the developer and user base for
  discussion, before decisions are made.
- Criteria for cutting a new version release and when those criteria are met.

Steering committee decisions should strive for consensus.
If consensus cannot be reached, the members of the steering committee
should vote.
Voting will take place over 7 days or until every steering committee member
votes or abstains.
The outcome of a vote is based on a simple majority.

.. _César Caballero-Gaudes: https://github.com/CesarCaballeroGaudes
.. _Logan Dowdle: https://github.com/dowdlelt
.. _Elizabeth DuPre: https://github.com/emdupre
.. _Javier Gonzalez-Castillo: https://github.com/javiergcas
.. _Dan Handwerker: https://github.com/handwerkerd
.. _Stefano Moia: https://github.com/smoia
.. _Taylor Salo: https://tsalo.github.io
.. _Joshua Teves: https://github.com/jbteves
.. _Eneko Uruñuela: https://github.com/eurunuela
.. _Kirstie Whitaker: https://github.com/KirstieJane
.. _code of conduct: https://github.com/ME-ICA/tedana/blob/main/CODE_OF_CONDUCT.md
.. _all-contributors file: https://github.com/ME-ICA/tedana/blob/main/.all-contributorsrc
.. _bus factor: https://en.wikipedia.org/wiki/Bus_factor
.. _Repository: https://github.com/ME-ICA/tedana>
