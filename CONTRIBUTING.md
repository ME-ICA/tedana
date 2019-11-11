# Contributing to `tedana`

Welcome to the `tedana` repository! We're excited you're here and want to contribute.

These guidelines are designed to make it as easy as possible to get involved.
If you have any questions that aren't discussed below, please let us know by opening an [issue][link_issues]!

Before you start you'll need to set up a free [GitHub][link_github] account and sign in.
Here are some [instructions][link_signupinstructions].

Already know what you're looking for in this guide? Jump to the following sections:

* [Joining the conversation](#joining-the-conversation)
  * [Monthly developer calls](#monthly-developer-calls)
* [Contributing small documentation changes](#contributing-small-documentation-changes)
* [Contributing through Github](#contributing-through-github)
* [Understanding issues, milestones, and project boards](#understanding-issues-milestones-and-project-boards)
* [Making a change](#making-a-change)
* [Structuring contributions](#style-guide)
* [Recognizing contributors](#recognizing-contributions)

Don't know where to get started? 
Read [Joining the conversation](#joining-the-conversation) and pop into 
Gitter to introduce yourself! Let us know what your interests are and we 
will help you find an issue to contribute to. Thanks so much!

## Joining the conversation

`tedana` is a young project maintained by a growing group of enthusiastic developers&mdash; and we're excited to have you join!
Most of our discussions will take place on open [issues][link_issues].
We also maintain a [gitter chat room][link_gitter] for more informal conversations and general project updates.

There is significant cross-talk between these two spaces, and we look forward to hearing from you in either venue!
As a reminder, we expect all contributions to `tedana` to adhere to our [code of conduct][link_coc].

### Monthly developer calls

We run monthly developer calls via Zoom.
You can see the schedule via the `tedana` [google calendar](https://calendar.google.com/calendar/embed?src=pl6vb4t9fck3k6mdo2mok53iss%40group.calendar.google.com).
An agenda will be circulated in the gitter channel in advance of the meeting.

Everyone is welcome.
We look forward to meeting you there :hibiscus:

## Contributing small documentation changes
If you are new to GitHub and just have a small documentation change
recommendation, please submit it to [our e-mail address](mailto:tedana.devs@gmail.com)
and one of our developers will add it to the documentation directly.

## Contributing through GitHub

[git][link_git] is a really useful tool for version control.
[GitHub][link_github] sits on top of git and supports collaborative and distributed working.

You'll use [Markdown][markdown] to chat in issues and pull requests on GitHub.
You can think of Markdown as a few little symbols around your text that will allow GitHub
to render the text with a little bit of formatting.
For example you could write words as bold (`**bold**`), or in italics (`*italics*`),
or as a [link][rick_roll] (`[link](https://https://youtu.be/dQw4w9WgXcQ)`) to another webpage.

GitHub has a helpful page on
[getting started with writing and formatting Markdown on GitHub][writing_formatting_github].


## Understanding issues, milestones and project boards

Every project on GitHub uses [issues][link_issues], [milestones][link_milestones],
and [project boards][link_project_boards] slightly differently.

The following outlines how the ``tedana`` developers think about these different tools.

* **Issues** are individual pieces of work that need to be completed to move the project forwards.
A general guideline: if you find yourself tempted to write a great big issue that
is difficult to describe as one unit of work, please consider splitting it into two or more issues.

    Issues are assigned [labels](#issue-labels) which explain how they relate to the overall project's
    goals and immediate next steps.

    Sometimes issues may not produce action items, and conversation will stall after a few months.
    When this happens, they may be marked stale by [stale-bot][link_stale-bot],
    and will be closed after a week unless there is more discussion.
    This helps us keep the issue tracker organized.
    Any new discussion on the issue will remove the `stale` label, and prevent it from closing.
    So, if theres's a discussion you think it not yet resolved, please jump in !

* **Milestones** are the link between the issues and the high level strategy for the ``tedana`` project.
Contributors new and old are encouraged to take a look at the milestones to see how we are progressing
towards ``tedana``'s shared vision.

    Issues are assigned to these milestones by the maintainers.
    If you feel that an issue should be assigned to a specific milestone but the maintainers have not done so, just ask!
    We might have just missed it, or we might not (yet) see how it aligns with the overall project structure.
    These conversations are important to have, and we are excited to hear your perspective!

* The **project board** is an automated [Kanban board][link_kanban] to keep track of what is currently underway
(in progress), what has been completed (done), and what remains to be done for a specific release.
The ``tedana``  maintainers use this board to keep an eye on how tasks are progressing week by week.


### Issue labels

The current list of labels are [here][link_labels] and include:

* [![Help Wanted](https://img.shields.io/badge/-help%20wanted-159818.svg)][link_helpwanted] *These issues contain a task that a member of the team has determined we need additional help with.*

    If you feel that you can contribute to one of these issues, we especially encourage you to do so!

* [![Paused](https://img.shields.io/badge/-paused-%23ddcc5f.svg)][link_paused] *These issues should not be worked on until the resolution of other issues or Pull Requests.*

    These are issues that are paused pending resolution of a related issue or Pull Request.
    Please do not open any Pull Requests to resolve these issues.

* [![Bugs](https://img.shields.io/badge/-bugs-fc2929.svg)][link_bugs] *These issues point to problems in the project.*

    If you find new a bug, please give as much detail as possible in your issue, including steps to recreate the error.
    If you experience the same bug as one already listed, please add any additional information that you have as a comment.

* [![Enhancement](https://img.shields.io/badge/-enhancement-84b6eb.svg)][link_enhancement] *These issues are asking for enhancements to be added to the project.*

    Please try to make sure that your enhancement is distinct from any others that have already been requested or implemented.
    If you find one that's similar but there are subtle differences please reference the other request in your issue.

## Making a change

We appreciate all contributions to `tedana`, but those accepted fastest will follow a workflow similar to the following:

### 1. Comment on an existing issue or [open a new issue][link_createissue] referencing your addition

This allows other members of the `tedana` development team to confirm that you aren't overlapping with work that's currently underway and that everyone is on the same page with the goal of the work you're going to carry out.

[This blog][link_pushpullblog] is a nice explanation of why putting this work in up front is so useful to everyone involved.

### 2. [Fork][link_fork] the [tedana repository][link_tedana] to your GitHub profile

This is now your own unique and online copy of `tedana`. Changes here won't affect anyone else's work, so it's a safe space to explore edits to the code!

Remember to [clone your fork][link_clonerepo] of `tedana` to your local machine, which will allow you to make local changes to `tedana`.

Make sure to always [keep your fork up to date][link_updateupstreamwiki] with the master repository before and after making changes.

### 3. Run the developer setup

To test a change, you may need to set up your local repository to run a `tedana` workflow. 
To do so, run
```
pip install -e .
```
from within your local `tedana` repository. This should ensure all packages are correctly organized and linked on your user profile. 

Once you've run this, your repository should be set for most changes (i.e., you do not have to re-run with every change). 

### 4. Make the changes you've discussed

Try to keep the changes focused to the issue. We've found that working on a [new branch][link_branches] for each issue makes it easier to keep your changes targeted. 

Using a new branch allows you to follow the standard "fork/branch/commit/pull-request/merge" GitHub workflow when making changes. [This guide][link_gitworkflow] provides a useful overview for this workflow.

Before creating your pull request, please make sure to review the `tedana` [style conventions](#style-guide).

### 5. Test your changes

#### Changes to code

For changes to the codebase, we suggest using our development Docker container which will run all the necessary checks and tests to ensure your code is ready to be merged into `tedana`!
(This does require that you have a local install of [Docker](https://www.docker.com/products/docker-desktop).)
You can run all the checks with:

```
docker run --tty --rm -v ${PWD}:/tedana tedana/tedana-dev:latest run_all_tests
```

from within your local `tedana` repository.
(**N.B.** It is possible that, depending on your Docker setup, you may need to increase the amount of memory available to Docker in order to run the `tedana` test suite. 
You can either do this permanently by editing your Docker settings or temporarily by adding `--memory=4g` to the above `docker run` command.)

This will print out a number of different status update messages as the tests run, but if you see `"FINISHED RUNNING ALL TESTS! GREAT SUCCESS"` then it means everything finished succesfully.
If not, there should be some helpful outputs that specify which tests failed.

#### Changes to documentation

For changes to documentation, we suggest rendering the HTML files locally in order to review the changes before submitting a pull request. This can be done by running
```
make html
```
from the `docs` directory in your local `tedana` repository. You should then be able to access the rendered files in the `docs/_build` directory, and view them in your browser.

### 6. Submit a [pull request][link_pullrequest]

When opening the pull request, we ask that you follow some [specific conventions](#pull-requests). We outline these below.

After you have submitted the pull request, a member of the development team will review your changes to confirm that they can be merged into the main code base.

After successful merging of the pull request, remember to [keep your fork up to date][link_updateupstreamwiki] with the master `tedana` repository and to delete the branch on your fork that was used for the merged pull request.

### Pull Requests

To improve understanding pull requests "at a glance", we encourage the use of several standardized tags.
When opening a pull request, please use at least one of the following prefixes:

* **[BRK]** for changes which break existing builds or tests
* **[DOC]** for new or updated documentation
* **[ENH]** for enhancements
* **[FIX]** for bug fixes
* **[REF]** for refactoring existing code
* **[STY]** for stylistic changes
* **[TST]** for new or updated tests, and
* **[WIP]** for changes which are not yet ready to be merged

Pull requests should be submitted early and often!
If your pull request is not yet ready to be merged, please also include the **[WIP]** prefix.
This tells the development team that your pull request is a "work-in-progress",
and that you plan to continue working on it.
We request that you do not use the Draft PR feature at this time, 
as it interferes with our Continuous Integration tool, Travis.

You can also combine the tags above, for example if you are updating both a test and
the documentation: **[TST, DOC]**.
If you're still working on the pull request that prefix would be **[WIP, TST, DOC]**.

## Style Guide

Docstrings should follow [numpydoc][link_numpydoc] convention.
We encourage extensive documentation.

The python code itself should follow [PEP8][link_pep8] convention
whenever possible, with at most about 500 lines of code (not including docstrings) per script.

Our documentation is written in [ReStructuredText](#writing-in-restructuredtext),
which we explain in more detail below.

### Writing in ReStructuredText

The documentation for `tedana` is written using [ReStructuredText][restructuredtext].
Using this markup language allows us to create an online site using the [Sphinx][sphinx]
documentation generator.
We then host the generated Sphinx site on [ReadTheDocs][readthedocs],
to provide an easily accessible space for accessing `tedana` documentation.

What this means is that we need to add any updates to the documentation in ReStructuredText,
or `rst`.
The resulting text looks slightly different from the markdown formatting you'll
[use on github](#contributing-through-github), but we're excited to help you get started!
Here's [one guide we've found particularly helpful][link_rst_guide] for starting with `rst`.
And, if you have any questions, please don't hesitate to ask!


## Recognizing contributors

We welcome and recognize [all contributions][link_all-contributors-spec] 
from documentation to testing to code development.
You can see a list of current contributors in the 
README
(kept up to date by the [all contributors bot][link_all-contributors-bot]). 
You can see [here][link_all-contributors-bot-usage] for instructions on
how to use the bot.

## Thank you!

You're awesome. :wave::smiley:

<br>

*&mdash; Based on contributing guidelines from the [STEMMRoleModels][link_stemmrolemodels] project.*

[link_git]: https://git-scm.com
[link_github]: https://github.com/
[link_tedana]: https://github.com/ME-ICA/tedana
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account

[writing_formatting_github]: https://help.github.com/articles/getting-started-with-writing-and-formatting-on-github
[markdown]: https://daringfireball.net/projects/markdown
[rick_roll]: https://www.youtube.com/watch?v=dQw4w9WgXcQ
[restructuredtext]: http://docutils.sourceforge.net/rst.html#user-documentation
[sphinx]: http://www.sphinx-doc.org/en/master/index.html
[readthedocs]: https://docs.readthedocs.io/en/latest/index.html

[link_issues]: https://github.com/ME-ICA/tedana/issues
[link_milestones]: https://github.com/ME-ICA/tedana/milestones/
[link_project_boards]: https://github.com/ME-ICA/tedana/projects
[link_gitter]: https://gitter.im/me-ica/tedana
[link_coc]: https://github.com/ME-ICA/tedana/blob/master/CODE_OF_CONDUCT.md
[link_stale-bot]: https://github.com/probot/stale

[link_labels]: https://github.com/ME-ICA/tedana/labels
[link_paused]: https://github.com/ME-ICA/tedana/labels/paused
[link_bugs]: https://github.com/ME-ICA/tedana/labels/bug
[link_helpwanted]: https://github.com/ME-ICA/tedana/labels/help%20wanted
[link_enhancement]: https://github.com/ME-ICA/tedana/labels/enhancement

[link_kanban]: https://en.wikipedia.org/wiki/Kanban_board
[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request/
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_pushpullblog]: https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/
[link_updateupstreamwiki]: https://help.github.com/articles/syncing-a-fork/
[link_branches]: https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/
[link_createissue]: https://help.github.com/articles/creating-an-issue/
[link_clonerepo]: https://help.github.com/articles/cloning-a-repository/
[link_gitworkflow]: https://guides.github.com/introduction/flow/

[link_numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[link_pep8]: https://www.python.org/dev/peps/pep-0008/
[link_rst_guide]: http://docs.sphinxdocs.com/en/latest/step-1.html

[link_contributors]: https://github.com/ME-ICA/tedana/graphs/contributors
[link_all-contributors-spec]: https://allcontributors.org/docs/en/specification
[link_all-contributors-bot]: https://allcontributors.org/docs/en/bot/overview
[link_all-contributors-bot-usage]: https://allcontributors.org/docs/en/bot/usage
[link_stemmrolemodels]: https://github.com/KirstieJane/STEMMRoleModels
