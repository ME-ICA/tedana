Developer Guidelines
====================

Basics
------
.. _basics:
``tedana`` uses git_ version control on the popular open source platform GitHub.
This allows the developers to easily manage changes even if they're very complex.
The most comprehensive tutorial on using git_ is `git pro`_.
For `small documentation patches`_ read the below section; otherwise, you will need to have a basic understanding of git.
When making code changes, you will use the `git workflow`_ described.
Once you're happy with your progress and open a pull request, a team member will review your changes via GitHub.
They may ask you to make changes or make suggestions.
Additionally, your changes will trigger automatic testing.
If your tests fail, you can inspect their results to see the cause of failure.
Common failures include:

- Linting (code style; we adhere to PEP8)
- Unit Test failure (a module does not work as expected)
- Integration Test failure (a change causes the program as a whole to fail)

If all tests are successful, you can squash and merge your changes with an approving review.
Thanks for your contribution!


Git Workflow
------------
.. _`git workflow`:
When making changes to the code or large documentation changes, you will want to use the following workflow:

1. Fork_ the repository_ on GitHub. 
#. Create a new branch locally. 
#. Frequently fetch and merge the ``tedana`` master branch into your work (this avoids merge conflicts). 
#. Open a `pull request`_ to the ``tedana`` project. 

There are a variety of ways to do this on a GUI'd ``git`` client.
If you're using the command line and ssh keys, the following will approximately follow the above:

.. code-block:: none

    # clone your fork
    git clone git@github.com:YOUR_PROFILE/tedana.git
    # enter your fork
    cd tedana
    # add the repository as an upstream
    git remote add upstream git@github.com:ME-ICA/tedana.git
    # create a new feature branch
    git checkout -b feature/my_awesome_contribution
    # work, work, work, ...
    # stay up to date
    git fetch upstream
    git merge upstream/master
    # push to your remote
    git push -u origin feature/my_awesome_contribution
    # the command line makes a link you can use to open a pull request
    # maybe you'll continue working
    git push
    # your pull request is now up-to-date

NOTE: if you don't understand the above steps, consider using a git tutorial and a practice repository until you get the hang of it.
Feel free also to ask questions and use a ``git`` client.
Several popular ones include

- GitKraken_
- `GitHub Desktop`_
- SourceTree_

Small Documentation Patches
---------------------------
.. _`small documentation patches`:
You can use the `GitHub UI`_ to make small documentation patches directly on the repository.
Please create a new branch as part of step 7 in the linked article.
Note that the documentation is mostly hosted inside of this_ folder using ReStructuredText_.



.. _git: https://git-scm.com/
.. _`git pro`: https://git-scm.com/book/en/v2
.. _repository: https://github.com/ME-ICA/tedana
.. _Fork: https://help.github.com/en/github/getting-started-with-github/fork-a-repo
.. _`pull request`: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request
.. _GitKraken: https://www.gitkraken.com/
.. _`GitHub Desktop`: https://desktop.github.com/
.. _SourceTree: https://www.sourcetreeapp.com/
.. _`GitHub UI`: https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository
.. _this: https://github.com/ME-ICA/tedana/tree/master/docs
.. _ReStructuredText: http://docutils.sourceforge.net/rst.html#user-documentation
