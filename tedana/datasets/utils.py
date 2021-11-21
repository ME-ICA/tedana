"""Utilities for downloading selected multi-echo neuroimaging datasets.

Adapted from the nilearn dataset fetchers.
"""
import json
import os
import re


def build_json_from_manifest(manifest_file, out_file):
    """Generate a tedana-format json file from an OSF manifest file.

    This function reorganizes the data stored in the manifest file and annotates each file
    according to the groups with which it should be associated.
    These groups are strongly linked to fMRIPrep derivatives, as well as dev-generated
    low-resolution versions of those derivatives.

    Parameters
    ----------
    manifest_file : :obj:`str`
        Path to the OSF manifest JSON file.
    out_file : :obj:`str`
        Path to the output JSON file.

    Notes
    -----
    The three groups currently included are:

    - "minimal_nativeres": The minimal files necessary for analyses of native-resolution echo-wise
      functional data. This includes the semi-preprocessed functional files, the native-space
      brain mask, confounds, transforms necessary to warp to standard space, and the dataset
      description file.
    - "minimal_lowres": The minimal files necessary for analyses of downsampled (5mm3) echo-wise
      functional data. This includes the downsampled, semi-preprocessed functional files,
      confounds, and the dataset description file.
    - "all_nativeres": All of the fMRIPrep derivatives, except for dev-generated downsampled files.
    """
    DROP_ENTITIES_FOR_TYPE = ["sub", "ses", "echo"]
    IGNORE_FOLDERS_FOR_TYPE = ["figures", "log"]

    with open(manifest_file, "r") as fo:
        data = json.load(fo)

    new_data = {}
    for fn, link in data.items():
        new_data[fn] = {}
        new_data[fn]["groups"] = []
        new_data[fn]["key"] = os.path.basename(link)

        # Preprocessed echo-wise data and brain mask go in associated minimal groups
        if ("desc-partialPreproc_bold" in fn) and ("res-5mm" not in fn):
            new_data[fn]["groups"].append("minimal_nativeres")
        elif ("desc-partialPreproc_bold" in fn) and ("res-5mm" in fn):
            new_data[fn]["groups"].append("minimal_lowres")
        elif "space-scanner_desc-brain_mask" in fn:
            new_data[fn]["groups"].append("minimal_nativeres")

        # Transforms go in native-res minimal group
        if fn.endswith("from-T1w_to-scanner_mode-image_xfm.txt"):
            new_data[fn]["groups"].append("minimal_nativeres")
        elif fn.endswith("from-scanner_to-T1w_mode-image_xfm.txt"):
            new_data[fn]["groups"].append("minimal_nativeres")
        elif fn.endswith("from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"):
            new_data[fn]["groups"].append("minimal_nativeres")

        # Confounds go in both minimal groups
        if "desc-confounds_timeseries" in fn:
            new_data[fn]["groups"].append("minimal_nativeres")
            new_data[fn]["groups"].append("minimal_lowres")

        # Dataset description goes in both minimal groups
        if "dataset_description.json" in fn:
            new_data[fn]["groups"].append("minimal_nativeres")
            new_data[fn]["groups"].append("minimal_lowres")

        # All files not flagged as low-resolution go in the "all" group
        if "res-5mm" not in fn:
            new_data[fn]["groups"].append("all_nativeres")

        # Now to apply "type" labels
        if any(ignore_folder in fn.split("/") for ignore_folder in IGNORE_FOLDERS_FOR_TYPE):
            new_data[fn]["type"] = ""
        elif fn.endswith(".html"):
            new_data[fn]["type"] = "report"
        else:
            mapped_name = os.path.basename(fn)
            for ent in DROP_ENTITIES_FOR_TYPE:
                mapped_name = re.sub(f"{ent}-[0-9a-zA-Z]+_", "", mapped_name)

            new_data[fn]["type"] = mapped_name

    with open(out_file, "w") as fo:
        json.dump(new_data, fo, sort_keys=True, indent=4)


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


def _fetch_dataset_by_groups(config_file, data_dir, groups, overwrite=False):
    """Fetch generic database by group labels."""
    import json

    with open(config_file, "r") as fo:
        config = json.load(fo)

    os.makedirs(data_dir, exist_ok=True)

    found_files = []
    for filename, info in config.items():
        if any([g in info["groups"] for g in groups]):
            found_files.append(filename)

    if not found_files:
        valid_groups = [info["groups"] for info in config.values()]
        # From https://stackoverflow.com/a/952952/2589328
        valid_groups = [item for sublist in valid_groups for item in sublist]
        valid_groups = sorted(list(set(valid_groups)))
        raise ValueError(f"No files found for group {groups}. Supported groups are {valid_groups}")

    return found_files


def fetch_neurosynth(data_dir=None, overwrite=False, groups=("nativeres",)):
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
    groups : tuple of str
        A list of groups you want to download data from.

    Returns
    -------
    found_files : list of str

    Notes
    -----
    This function was adapted from neurosynth.base.dataset.download().

    Warning
    -------
    Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
    format. Old code using this function **will not work** with the new version.
    """
    dataset_name = "cambridge"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)

    config_file = f"data/{dataset_name}.json"

    found_files = _fetch_dataset_by_groups(config_file, data_dir, groups, overwrite=overwrite)

    return found_files
