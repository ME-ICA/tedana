.. include:: links.rst

.. _fetching tools:

Fetching resources from the internet
====================================
:mod:`tedana.datasets`

tedana's ``datasets`` module contains a number of functions for downloading multi-echo data from the internet.

.. topic:: Where do downloaded resources end up?

    The fetching functions in tedana use the same approach as ``nilearn``.
    Namely, data fetched using tedana's functions will be downloaded to the disk.
    These files will be saved to one of the following directories:

    - the folder specified by ``data_dir`` parameter in the fetching function
    - the global environment variable ``TEDANA_SHARED_DATA``
    - the user environment variable ``TEDANA_DATA``
    - the ``tedana_data`` folder in the user home folder

    The two different environment variables (``TEDANA_SHARED_DATA`` and ``TEDANA_DATA``) are provided for multi-user systems,
    to distinguish a global dataset repository that may be read-only at the user-level.
    Note that you can copy that folder to another user's computers to avoid the initial dataset download on the first fetching call.

    You can check in which directory tedana will store the data with the function :func:`tedana.datasets.utils.get_data_dirs`.
