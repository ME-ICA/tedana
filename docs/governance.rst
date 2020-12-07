Governance
==========
Governance is a hugely important part of any project.
It is especially important to have clear processes and communication channels
for open source projects that rely on a distributed network of volunteers,
such as ``tedana``.

Overview
--------

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
----------

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
  | `Joshua Teves`_ (@jbteves)                |
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
  | `Joshua Teves`_ (@jbteves)           |
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
  - | Collaborative programming leaders: `Elizabeth DuPre`_ & `Joshua Teves`_
    |   Helps make sure tedana is following best practices for Python code
        design, testing, and communication for issues/pull requests etc.
  - | Communications leader: `Joshua Teves`_
    |   Mailing list manager & other outward-facing communication about
        the project
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
````````````````
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
-----------------------

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
```````````````````
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
.. _code of conduct: https://github.com/ME-ICA/tedana/blob/master/CODE_OF_CONDUCT.md
.. _all-contributors file: https://github.com/ME-ICA/tedana/blob/master/.all-contributorsrc
.. _bus factor: https://en.wikipedia.org/wiki/Bus_factor
.. _Repository: https://github.com/ME-ICA/tedana>
