====================
Developer Guidelines
====================

Adding and Modifying Tests
==========================
Testing is an important component of development.
For simplicity, we have migrated all tests to ``pytest``.
There are two basic kinds of tests:
unit and integration tests.
Unit tests focus on testing individual functions,
whereas integration tests focus on making sure that the whole workflow
runs correctly.

For unit tests,
we try to keep tests on the same module grouped into one file.
Make sure the function you're testing is imported,
then write your test.
Good tests will make sure that edge cases are accounted for as well as
common cases.
You may also use ``pytest.raises`` to ensure that errors are thrown for
invalid inputs to a function.

For integration tests,
make a ``tar.gz`` file which will unzip to be only the files you'd like to run a workflow on.
You can do this with the following, which would make an archive ``my_data.tar.gz``:

.. code-block:: none
    tar czf my_data.tar.gz my_data/*.nii.gz

Run the workflow with a known-working version, and put the outputs into a
text file inside ``$TEDANADIR/tedana/tests/data/``,
with ``TEDANADIR`` the local ``tedana repository``.
You can follow the model our `five echo set`_,
which has the following steps:

1. Check if a pytest user is skipping integration, skip if so
#. Use ``download_test_data`` to retrieve the test data from OSF
#. Run a workflow
#. Use ``resources_filename`` and ``check_integration_outputs`` to compare your expected output to actual

If you need to upload new data, you will need to contact the maintainers
and ask them to either add it or give you permission to add it.
Once you've tested your integration test locally and it is working,
you will need to add it to the CircleCI config and the ``Makefile``.
Following the model of the three-echo and five-echo sets,
define a name for your integration test and on an indented line below put 

.. code-block:: none

    @py.test --cov-append --cov-report term-missing --cov=tedana -k TEST

with ``TEST`` your test function's name. 
This call basically adds code coverage reports to account for the new test,
and runs the actual test in addition.
Using the five-echo set as a template,
you should then edit ``.circlec/config.yml`` to add your test,
calling the same name you define in the ``Makefile``.

If you need to take a look at a failed test on CircleCI rather than
locally, you can use the following block to retrieve artifacts
(see CircleCI documentation here_)

.. code-block:: none
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
Suppose that a 

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
.. _`five echo set`: https://github.com/ME-ICA/tedana/blob/37368f802f77b4327fc8d3f788296ca0f01074fd/tedana/tests/test_integration.py#L71-L95
.. _here: https://circleci.com/docs/2.0/artifacts/#downloading-all-artifacts-for-a-build-on-circleci
.. _`getting one`: https://circleci.com/docs/2.0/managing-api-tokens/?gclid=CjwKCAiAqqTuBRBAEiwA7B66heDkdw6l68GAYAHtR2xS1xvDNNUzy7l1fmtwQWvVN0OIa97QL8yfhhoCejoQAvD_BwE#creating-a-personal-api-token
