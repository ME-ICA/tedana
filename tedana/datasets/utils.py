"""Utilities for downloading selected multi-echo neuroimaging datasets.

Adapted from the nilearn dataset fetchers.
"""
import os


def _readlinkabs(link):
    """Return an absolute path for the destination of a symlink."""
    path = os.readlink(link)
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(link), path)


def get_data_dirs(data_dir=None):
    """Return the directories in which tedana looks for data.

    This is typically useful for the end-user to check where the data is downloaded and stored.

    Parameters
    ----------
    data_dir : :obj:`pathlib.Path` or :obj:`str` or None, optional
        Path of the data directory. Used to force data storage in a specified location.
        Default: None

    Returns
    -------
    paths : :obj:`list` of :obj:`str`
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the dataset's directory using the following priority :

    1. the keyword argument ``data_dir``
    2. the global environment variable ``TEDANA_SHARED_DATA``
    3. the user environment variable ``TEDANA_DATA``
    4. ``tedana_data`` in the user home folder
    """
    # We build an array of successive paths by priority
    # The boolean indicates if it is a pre_dir: in that case, we won't add the
    # dataset name to the path.
    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend(str(data_dir).split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:
        global_data = os.getenv("TEDANA_SHARED_DATA")
        if global_data is not None:
            paths.extend(global_data.split(os.pathsep))

        local_data = os.getenv("TEDANA_DATA")
        if local_data is not None:
            paths.extend(local_data.split(os.pathsep))

        paths.append(os.path.expanduser("~/tedana_data"))
    return paths


def _get_dataset_dir(dataset_name, data_dir=None, default_paths=None, verbose=1):
    """Create if necessary and return data directory of given dataset.

    Parameters
    ----------
    dataset_name : :obj:`str`
        The unique name of the dataset.
    data_dir : :obj:`pathlib.Path` or :obj:`str` or None, optional
        Path of the data directory. Used to force data storage in a specified location.
        Default: None
    default_paths : :obj:`list` of :obj:`str`, optional
        Default system paths in which the dataset may already have been installed by a third party
        software. They will be checked first.
    verbose : :obj:`int`, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data_dir : :obj:`str`
        Path of the given dataset directory.

    Notes
    -----
    This function retrieves the dataset's directory (or data directory) using the following
    priority :

    1. defaults system paths
    2. the keyword argument ``data_dir``
    3. the global environment variable ``TEDANA_SHARED_DATA``
    4. the user environment variable ``TEDANA_DATA``
    5. ``tedana_data`` in the user home folder
    """
    paths = []
    # Search possible data-specific system paths
    if default_paths is not None:
        for default_path in default_paths:
            paths.extend([(d, True) for d in default_path.split(os.pathsep)])

    paths.extend([(d, False) for d in get_data_dirs(data_dir=data_dir)])

    if verbose > 2:
        print(f"Dataset search paths: {paths}")

    # Check if the dataset exists somewhere
    for path, is_pre_dir in paths:
        if not is_pre_dir:
            path = os.path.join(path, dataset_name)

        if os.path.islink(path):
            # Resolve path
            path = _readlinkabs(path)

        if os.path.exists(path) and os.path.isdir(path):
            if verbose > 1:
                print(f"\nDataset found in {path}\n")

            return path

    # If not, create a folder in the first writeable directory
    errors = []
    for (path, is_pre_dir) in paths:
        if not is_pre_dir:
            path = os.path.join(path, dataset_name)

        if not os.path.exists(path):
            try:
                os.makedirs(path)
                if verbose > 0:
                    print(f"\nDataset created in {path}\n")
                return path

            except Exception as exc:
                short_error_message = getattr(exc, "strerror", str(exc))
                errors.append(f"\n -{path} ({short_error_message})")

    raise OSError(
        f"tedana tried to store the dataset in the following directories, but: {', '.join(errors)}"
    )
