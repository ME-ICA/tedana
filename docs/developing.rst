====================
Developer Guidelines
====================

This webpage is intended to guide users through making making changes to
``tedana``'s codebase, in particular working with tests.
The worked example also offers some guidelines on approaching testing when
adding new functions.

Adding and Modifying Tests
==========================
Testing is an important component of development.
For simplicity, we have migrated all tests to ``pytest``.
There are two basic kinds of tests: unit and integration tests.
Unit tests focus on testing individual functions, whereas integration tests focus on making sure
that the whole workflow runs correctly.

For unit tests, we try to keep tests on the same module grouped into one file.
Make sure the function you're testing is imported, then write your test.
Good tests will make sure that edge cases are accounted for as well as common cases.
You may also use ``pytest.raises`` to ensure that errors are thrown for invalid inputs to a
function.

For integration tests, make a ``tar.gz`` file which will unzip to be only the files you'd like to
run a workflow on.
You can do this with the following, which would make an archive ``my_data.tar.gz``:

.. code-block:: bash

    tar czf my_data.tar.gz my_data/*.nii.gz

Run the workflow with a known-working version, and put the outputs into a text file inside
``$TEDANADIR/tedana/tests/data/``, where ``TEDANADIR`` is your local ``tedana repository``.
To write the test function you can follow the model of our `five echo set`_, which takes the following steps:

1. Check if a pytest user is skipping integration, skip if so
#. Use ``download_test_data`` to retrieve the test data from OSF
#. Run a workflow
#. Use ``resources_filename`` and ``check_integration_outputs`` to compare your expected output to
   actual output.

If you need to upload new data, you will need to contact the maintainers and ask them to either add
it or give you permission to add it.
Once you've tested your integration test locally and it is working, you will need to add it to the
CircleCI config and the ``Makefile``.
Following the model of the three-echo and five-echo sets, define a name for your integration test
and on an indented line below put

.. code-block:: bash

    @py.test --cov-append --cov-report term-missing --cov=tedana -k TEST

with ``TEST`` your test function's name.
This call basically adds code coverage reports to account for the new test, and runs the actual
test in addition.
Using the five-echo set as a template, you should then edit ``.circlec/config.yml`` to add your
test, calling the same name you define in the ``Makefile``.

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

First, we merge the repository's ``master`` branch into our own to make sure we're up to date, and
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
We merge any changes from the upstream master branch into our branch via

.. code-block:: bash

    git checkout feature/say_hello  # unless you're already there
    git fetch upstream master
    git merge upstream/master

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

    git checkout master
    git fetch upstream master
    git merge upstream/master
    git branch -d feature/say_hello
    git push --delete origin feature/say_hello

and we're good to go!

Monthly Developer Calls
=======================
We run monthly developer calls via Zoom.
You can see the schedule via the tedana `google calendar`_.

Everyone is welcome.
We look forward to meeting you there :hibiscus:

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
.. _`google calendar`: https://calendar.google.com/calendar/embed?src=pl6vb4t9fck3k6mdo2mok53iss%40group.calendar.google.com
