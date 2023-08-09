############
Installation
############

You'll need to set up a working development environment to use ``tedana``.
To set up a local environment, you will need Python >=3.8 and the following
packages will need to be installed:

- nilearn
- nibabel
- numpy
- scikit-learn
- scipy
- mapca

After installing relevant dependencies, you can then install ``tedana`` with:

.. code-block:: bash

  pip install tedana

In addition to the Python package, installing ``tedana`` will add the ``tedana``
and ``t2smap`` workflow CLIs to your path.
You can confirm that ``tedana`` has successfully installed by launching a Python instance and running:

.. code-block:: python

  import tedana

You can check that it is available through the command line interface (CLI) with:

.. code-block:: bash

  tedana --help

If no error occurs, ``tedana`` has correctly installed in your environment!
