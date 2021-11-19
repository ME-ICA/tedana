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


def query_files():
    VALID_ENTITIES = {
        "coordinates.tsv.gz": ["data", "version"],
        "metadata.tsv.gz": ["data", "version"],
        "features.npz": ["data", "version", "vocab", "source", "type"],
        "vocabulary.txt": ["data", "version", "vocab"],
        "metadata.json": ["data", "version", "vocab"],
        "keys.tsv": ["data", "version", "vocab"],
    }


def _find_entities(filename, search_pairs, log=False):
    """Search file for any matching patterns of entities."""
    # Convert all string-based kwargs to lists
    search_pairs = {k: [v] if isinstance(v, str) else v for k, v in search_pairs.items()}
    search_pairs = [[f"{k}-{v_i}" for v_i in v] for k, v in search_pairs.items()]
    searches = list(itertools.product(*search_pairs))

    file_parts = filename.split("_")
    suffix = file_parts[-1]
    valid_entities_for_suffix = VALID_ENTITIES[suffix]
    for search in searches:
        temp_search = [term for term in search if term.split("-")[0] in valid_entities_for_suffix]
        if all(term in file_parts for term in temp_search):
            return True

    return False


def _fetch_dataset(config_file, data_dir, search_pairs, overwrite=False):
    """Fetch generic database."""
    import json

    with open(config_file, "r") as fo:
        config = json.load(fo)

    os.makedirs(data_dir, exist_ok=True)

    found_files = []
    for filename in config:
        if _find_entities(filename, search_pairs):
            found_files.append(filename)

    return found_files



def fetch_neurosynth(data_dir=None, overwrite=False, **kwargs):
    """Download the latest data files from NeuroSynth.

    Parameters
    ----------
    data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
        Path where data should be downloaded. By default, files are downloaded in home directory.
        A subfolder, named ``neurosynth``, will be created in ``data_dir``, which is where the
        files will be located.
    version : str or list, optional
        The version to fetch. The default is "7" (Neurosynth's latest version).
    overwrite : bool, optional
        Whether to overwrite existing files or not. Default is False.
    kwargs : dict, optional
        Keyword arguments to select relevant feature files.
        Valid kwargs include: source, vocab, type.
        Each kwarg may be a string or a list of strings.
        If no kwargs are provided, all feature files for the specified database version will be
        downloaded.

    Returns
    -------
    found_databases : :obj:`list` of :obj:`dict`
        List of dictionaries indicating datasets downloaded.
        Each list entry is a different database, containing a dictionary with three keys:
        "coordinates", "metadata", and "features". "coordinates" and "metadata" will be filenames.
        "features" will be a list of dictionaries, each containing "id", "vocab", and "features"
        keys with associated files.

    Notes
    -----
    This function was adapted from neurosynth.base.dataset.download().

    Warning
    -------
    Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
    format. Old code using this function **will not work** with the new version.
    """
    URL = (
        "https://github.com/neurosynth/neurosynth-data/blob/"
        "209c33cd009d0b069398a802198b41b9c488b9b7/"
    )
    dataset_name = "cambridge"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)

    config_file = f"data/{dataset_name}.json"

    found_files = _fetch_dataset(config_file, data_dir, kwargs, overwrite=overwrite)

    return found_databases
