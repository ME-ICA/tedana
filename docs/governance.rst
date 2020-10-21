Governance
==========
Governance is a hugely important part of any project.
It is especially important to have clear process and communication channels
for open source projects that rely on a distributed network of volunteers, such as ``tedana``.

Overview
--------

Tedana is a relatively small open source project that requires specialized
knowledge in multiple domains. This leads to several challenges. No one
person on the current tedana development team has a combination of 
available time plus expertise in collaborative software development, MRI
physics, and advanced data processing methods to assume a primary project
leader role. Even if such a person was interested, it may not benefit the
project to overly rely on the existence of one person. We developed the
following system the following goals in mind:

- Promote open discussions.
- Minimize the administrative burden.
- Strive for consensus.
- Provide a path for when consensus cannot be achieved.
- Grow the community.
- Maximize the `bus factor`_ of the project.
- Acknowledge the leadership and time multiple people contribute to our
  community without demanding more time than any individual can offer.
  Dividing leadership responsabilies into multiple smaller roles also
  makes it easier to encourage new people to take on a leadership role
  without fearing that too much work will be required of them.

This governance structure is a work-in-progress. We welcome both people
who want to take on a leadership role as well as ideas for our to improve
this structure.



Steering Committee and Leadership Roles
---------------------------------------

Organizational Principles
`````````````````````````
The steering committee steers. The goal of the steering committee is to help
guide the direction of the project. Decisions in the steering committee will 
focus on how to present project issues to the broader community in a clear way
rather than making project decisions without community input. 

Issues or pull requests that can be reviewed by any community member can be
reviewed and approved by the steering committee or steering committee members.
The steering committee can decide that an issue should be prioritized for wider
communal discussion or that a pull request requires more discussion or reviews
than standard before merging. The steering committee can also decide how a
breaking change (something that changes existing user function calls or program
outputs) will be presented to the developer and user base for discussion, before
decisions are made.

Steering committee decisions should strive for consensus and accept the
majority’s decision. If voting is necessary, it can happen asynchronously and is
defined as more than half of the members of the steering committee. If a member
of the steering committee is unable to spend the time necessary to understand
an issue & vote, they can recuse from a decision so that the number of committee
members is reduced

Openness is a priority:

- Openness is critical to building trust with the broader community
- Openness provides a mechanism for non-steering committee members to identify
  and address steering committee blindspots
- Openness provides a smoother transition to onboard future steering committee
  members
- If steering committee discussions (written or verbal) could include
  community members without compromising efficiency, they should be invited. The
  steering committee can schedule discussions without needing to ask about the
  availability for people outside of the steering committee. If notes from
  meetings can be openly shared without compromising personal privacy, they
  should be.

The current members of the tedana steering committee are:

- Name 1 & link to github profile
- Name 2 & link to github profile
- Name 3 & link to github profile
- Name 4 & link to github profile
- Name 5 & link to github profile

If you are interested in joining the steering committee, please let the current members know.

Leadership roles
````````````````
We have identified key skills that help advance tedana development and
volunteers who will be the point-people for each of these roles. One person
can fill more than one role and more than one person can decide to share or
split the responsabilities of a role. The steering committee will include
people who also take on leadership roles, but a person can take on a leadership
role without also volunteering to be on the steering committee.

If you are interested in taking over one of these roles, splitting a role, or
creating a new leadership role, please talk to some of the current leaders.

- | Task manager & record keeper: `Dan Handwerker`_
  |   Helps write & keep track of notes from meetings
  |   Keeps track of issues or items that should be addressed
  |   Follows up with people who volunteered to address an item or alerts the broader community of known tasks that could use a volunteer
- | MR physics leader `César Caballero-Gaudes`_
  |   Someone who can make sure calculations fit within our understanding of MR physics
  |   Someone who can either answer MRI physics questions related to multi-echo or direct people to where they can find answers
- | Processing algorithms leader `Eneko Uruñuela`_ (Decomposiiton) & `Dan Handwerker`_ (Metrics & Decision Tree)
  |   Someone who can make sure algorithms are appropriately implemented (or knows enough to delegate to someone who can make sure implementation is good)
  |   Someone who can either answer processing algorithm questions or direct people to where they can find answers
- | Collaborative programming leader `Elizabeth DuPre`_ & `Joshua Teves`_
  |   Helps make sure tedana is following best practices for Python code design, testing, and communication for issues/pull requests etc.
- | Communications leader `Joshua Teves`_
  |   Mailing list manager & other outward-facing communication about the project
- | New contributors leader `Taylor Salo`_
  |   Leads efforts to make contributor documentation more welcoming
  |   Is a point of contact for potential contributors to make them feel welcome and direct them to relevant resources or issues
- | Multi-echo fMRI support leader `Logan Dowdle`_
  |   Monitors places where people may ask questions about tedana or multi-echo fMRI and tries to find someone to answer those questions
- | Enforcer(s) of the `code of conduct`_ `Elizabeth DuPre`_ & `Dan Handwerker`_ & another volunteer
  |   A person or people someone can go to if they want to report a code of conduct violation
  |   If this is one person, that person should NOT be on the steering committee
  |   If this is more than one person, at least one should not be on the steering committee
  |   Ideal is someone who cares about tedana but DOESN’T know contributors well enough to say, ”Person X would never do that”



Changing leaders
````````````````
Steering committee members can remove themselves from the steering committee at
any time and open up a call for a new self-nomination. Anyone can request to take
on a leadership role at any time. Once per year, there should be an explicit call
to the larger contributor community asking if anyone wants to self nominate for
membership on the steering committee or other leadership roles. If individuals
cannot reach consensus on who steps back and who assumes new roles, then a
majority vote of contributors from the previous 3 years will assign people to
roles where there are conflicts.

If there are concerns with a tedana steering committee member or leader, any
enforcer of the code of conduct can ask anyone to step down from a leadership role.
If a person refuses to step down, then an enforcer of the code of conduct can call
a vote of contributors to remove an individual from a leadership role in tedana.


Decision Making Process
-----------------------

These rules outlined below are inspired by the 
`decision-making rules for the BIDS standard <https://github.com/bids-standard/bids-specification/blob/master/DECISION-MAKING.md>`_, which in turn were inspired by the
`lazy consensus system used in the Apache Foundation <https://www.apache.org/foundation/voting.html>`_,
and heavily depend on the
`GitHub Pull Request review system <https://help.github.com/articles/about-pull-requests/>`_.

Definitions
```````````

Repository
  `https://github.com/ME-ICA/tedana <https://github.com/ME-ICA/tedana>`_

Contributor
  Person listed in the `all-contributors file`_.
  The community decides on the content of this file using the same process as any
  other change to the Repository (see below) allowing the meaning of "Contributor"
  to evolve independently of the Decision-making rules.

Maintainer
  A Contributor responsible for the long term health of the project and the
  community. Maintainers have additional rights (see Rules) helping them to
  resolve conflicts and increase the pace of the development when necessary.
  Any maintainer can self-remove themselves. Any contributor can become a
  maintainer by request and with the support of the majority of the current
  maintainers. Current Maintainers:

  +-----------------------------------+-----------------+
  | Name                              | Time commitment |
  +===================================+=================+
  | `Logan Dowdle`_ (@dowdlelt)       | 0.5h/week       |
  +-----------------------------------+-----------------+
  | `Elizabeth DuPre`_ (@emdupre)     | 0.5h/week       |
  +-----------------------------------+-----------------+
  | `Dan Handwerker`_ (@handwerkerd)  | 0.5h/week       |
  +-----------------------------------+-----------------+
  | `Ross Markello`_ (@rmarkello)     | 0.5h/week       |
  +-----------------------------------+-----------------+
  | `Taylor Salo`_ (@tsalo)           | 3h/week         |
  +-----------------------------------+-----------------+
  | `Joshua Teves`_ (@jbteves)        | 0.5h/week       |
  +-----------------------------------+-----------------+
  | `Eneko Uruñuela`_ (@eurunuela)    | 0.5h/week       |
  +-----------------------------------+-----------------+
  | `Kirstie Whitaker`_ (@KirstieJane)| 0.5h/week       |
  +-----------------------------------+-----------------+


Rules
`````

1. Potential modifications to the Repository should first be proposed via an 
   Issue. Rules regarding Votes apply to both Pull Requests and Issues.

   - Every modification of the specification (including a correction of a typo, adding a new Contributor, an extension adding support for a new data type, or others) or proposal to release a new version needs to be done via a Pull Request (PR) to the Repository.
2. Anyone can open a PR (this action is not limited to Contributors).
3. PRs adding new Contributors must also add their GitHub names to the 
   `all-contributors file`_. 
   This should be done with the allcontributors bot.

   - Contributors may also add themselves to the Zenodo file if they wish, but this is not mandatory.
4. A PR is eligible to be merged if and only if these conditions are met:

   a) The PR features at least two `Reviews that Approve <https://help.github.com/articles/about-pull-request-reviews/#about-pull-request-reviews>`_
      the PR from Maintainers of which neither is the author of the PR. 
      The reviews need to be made after the last commit in the PR (equivalent to 
      `Stale review dismissal <https://help.github.com/articles/enabling-required-reviews-for-pull-requests/>`_
      option on GitHub).
   b) Does not feature any `Reviews that Request changes <https://help.github.com/articles/about-required-reviews-for-pull-requests/>`_.
   c) Does not feature "WIP" in the title (Work in Progress).
   d) Passes all automated tests.
   e) Is not proposing a new release or has been approved by at least one
      Maintainer (i.e., PRs proposing new releases need to be approved by
      at least one Maintainer).
5. After consultation with contributors, the steering committee can decide
   to merge any PR - even if it's not eligible to merge according to Rule 4.
6. Any Maintainer can Review a PR and request changes. If a Maintainer Requests
   changes they need to provide an explanation regarding what changes should
   be added and justification of their importance. Reviews requesting
   changes can also be used to request more time to review a PR.
7. A Maintainer who Requested changes can Dismiss their own review or Approve
   changes added by the Contributor who opened the PR.
8. If the author of a PR and Maintainer who provided Review that Requests
   changes cannot find a solution that would lead to the Maintainer dismissing
   their review or accepting the changes the Review can be Dismissed with a vote.
9. Rules governing voting:

   a) A Vote can be triggered by any Maintainer, but only after 5 working days
      from the time a Review Requesting Changes has been raised and in case a
      Vote has been triggered previously no sooner than 15 working days since
      its conclusion.
   b) Only Maintainers can vote and each Maintainer gets one vote.
   c) A Vote ends after 7 working days or when all Maintainers have voted
      (whichever comes first).
   d) A Vote freezes the PR - no new commits or Reviews Requesting changes can
      be added to it while a vote is ongoing. If a commit is accidentally made
      during that period it should be reverted.
   e) The quorum for a Vote is five votes.
   f) The outcome of the vote is decided based on a simple majority.

.. _César Caballero-Gaudes: https://github.com/CesarCaballeroGaudes
.. _Logan Dowdle: https://github.com/dowdlelt
.. _Elizabeth DuPre: https://github.com/emdupre
.. _Dan Handwerker: https://github.com/handwerkerd
.. _Ross Markello: https://github.com/rmarkello
.. _Taylor Salo: https://tsalo.github.io
.. _Joshua Teves: https://github.com/jbteves
.. _Eneko Uruñuela: https://github.com/eurunuela
.. _Kirstie Whitaker: https://github.com/KirstieJane
.. _all-contributors file: https://github.com/ME-ICA/tedana/blob/master/.all-contributorsrc
.. _bus factor: https://en.wikipedia.org/wiki/Bus_factor
