# Decision-making rules

## Introduction

The tedana community set out the following decision-making rules with the intention to:

- Strive for consensus.
- Promote open discussions.
- Minimize the administrative burden.
- Provide a path for when consensus cannot be made.
- Grow the community.
- Maximize the [bus factor](https://en.wikipedia.org/wiki/Bus_factor) of the project.

The rules outlined below are inspired by the [decision-making rules for the BIDS standard](https://github.com/bids-standard/bids-specification/blob/master/DECISION-MAKING.md), which in turn were inspired by the [lazy consensus system used in the Apache Foundation](https://www.apache.org/foundation/voting.html),
and heavily depends on [GitHub Pull Request review system](https://help.github.com/articles/about-pull-requests/).

## Definitions

**Repository** - [https://github.com/ME-ICA/tedana](https://github.com/ME-ICA/tedana)

**Contributor** - a person listed in the [all-contributors file](.all-contributorsrc).
The community decides on the content of this file using the same process as any
other change to the Repository (see below) allowing the meaning of "Contributor"
to evolve independently of the Decision-making rules.

**Maintainer** - a Contributor responsible for the long term health of the
project and the community. Maintainers have additional rights (see Rules)
helping them to resolve conflicts and increase the pace of the development
when necessary. Current Maintainers:

| Name                                                            | Time commitment |
|-----------------------------------------------------------------|-----------------|
| Logan Dowdle ([@dowdlelt](https://github.com/dowdlelt))         | 0.5h/week       |
| Dan Handwerker ([@handwerkerd](https://github.com/handwerkerd)) | 0.5h/week       |
| Ross Markello ([@rmarkello](https://github.com/rmarkello))      | 0.5h/week       |
| Taylor Salo ([@tsalo](https://github.com/tsalo))                | 3h/week         |
| Joshua Teves ([@jbteves](https://github.com/jbteves))           | 0.5h/week       |
| Eneko Urunuela ([@eurunuela](https://github.com/eurunuela))     | 0.5h/week       |
| Kirstie Whitaker ([@KirstieJane](https://github.com/KirstieJane)) | 0.5h/week       |

**Project Leader** - the Contributor responsible for final decisions in cases of
stalled issues or pull requests. The Project Leader has additional rights
(see Rules) helping them to resolve conflicts and increase the pace of the development
when necessary. Current Project Leader:

| Name                                                            | Time commitment |
|-----------------------------------------------------------------|-----------------|
| Elizabeth DuPre ([@emdupre](https://github.com/emdupre))        | 0.5h/week       |

## Rules

1. Potential modifications to the Repository should first be proposed via an
   Issue. Rules regarding Votes apply to both Pull Requests and Issues.
1. Every modification of the specification (including a correction of a typo,
   adding a new Contributor, an extension adding support for a new data type, or
   others) or proposal to release a new version needs to be done via a Pull
   Request (PR) to the Repository.
1. Anyone can open a PR (this action is not limited to Contributors).
1. PRs adding new Contributors must also add their GitHub names to the
   [all-contributors file](.all-contributorsrc) file.
   This should be done with the allcontributors bot.
   1. Contributors may also add themselves to the Zenodo file if they wish, but
      this is not mandatory.
1. A PR is eligible to be merged if and only if these conditions are met:
   1. The PR features at least two [Reviews that Approve](https://help.github.com/articles/about-pull-request-reviews/#about-pull-request-reviews)
      the PR from Maintainers of which neither is the author of the PR. The reviews
      need to be made after the last commit in the PR (equivalent to
      [Stale review dismissal](https://help.github.com/articles/enabling-required-reviews-for-pull-requests/)
      option on GitHub).
   1. Does not feature any [Reviews that Request changes](https://help.github.com/articles/about-required-reviews-for-pull-requests/).
   1. Does not feature "WIP" in the title (Work in Progress).
   1. Passes all automated tests.
   1. Is not proposing a new release or has been approved by at least one
      Maintainer (i.e., PRs proposing new releases need to be approved by at
      least one Maintainer).
1. The Project Leader can merge any PR - even if it's not eligible to merge
   according to Rule 4.
1. Any Maintainer can Review a PR and request changes. If a Maintainer
   Requests changes they need to provide an explanation regarding what changes
   should be added and justification of their importance. Reviews requesting
   changes can also be used to request more time to review a PR.
1. A Maintainer who Requested changes can Dismiss their own review or Approve
   changes added by the Contributor who opened the PR.
1. If the author of a PR and Maintainer who provided Review that Requests
   changes cannot find a solution that would lead to the Maintainer dismissing
   their review or accepting the changes the Review can be Dismissed with a
   vote.
1. Rules governing voting:
   1. A Vote can be triggered by any Maintainer, but only after 5 working days
      from the time a Review Requesting Changes has been raised and in case a
      Vote has been triggered previously no sooner than 15 working days since
      its conclusion.
   1. Only Maintainers can vote and each Maintainer gets one vote.
   1. A Vote ends after 7 working days or when all Maintainers have voted
      (whichever comes first).
   1. A Vote freezes the PR - no new commits or Reviews Requesting changes can
      be added to it while a vote is ongoing. If a commit is accidentally made
      during that period it should be reverted.
   1. The quorum for a Vote is five votes.
   1. The outcome of the vote is decided based on a simple majority.

## Comments

1. Releases are triggered the same way as any other change - via a PR.
1. PRs MUST be merged using the "Squash and merge" option in GitHub.
   See the [GitHub help page](https://help.github.com/en/articles/about-merge-methods-on-github)
   for information on merge methods.
