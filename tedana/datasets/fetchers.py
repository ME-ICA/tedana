"""Functions for fetching datasets."""
import json
import logging
import os
import re

from sklearn.utils import Bunch

from ..due import Doi, due
from .utils import _get_dataset_dir

LGR = logging.getLogger("GENERAL")


@due.dcite(Doi("10.1073/pnas.1720985115"), description="Introduces the Cambridge dataset.")
@due.dcite(Doi("10.1038/s41592-018-0235-4"), description="Original citation for fMRIPrep.")
@due.dcite(Doi("10.1038/s41596-020-0327-3"), description="Updated citation for fMRIPrep.")
def fetch_cambridge(
    n_subjects=None,
    groups=("minimal_nativeres",),
    data_dir=None,
    resume=True,
    verbose=1,
):
    """Fetch Cambridge multi-echo data.

    See Notes below for more information on this dataset.
    Please cite [1]_ if you are using this dataset, as well as [2]_ and [3]_ because it was
    preprocessed with fMRIPrep.

    Parameters
    ----------
    n_subjects : :obj:`int` or None, optional
        The number of subjects to load. If None, all the subjects are loaded. Total 88 subjects.
    groups : tuple of {"minimal_nativeres", "minimal_lowres", "all_nativeres"}
        Which groups of files to download. Default = ``("minimal_nativeres",)``.
    data_dir : :obj:`str`, optional
        Path of the data directory. Used to force data storage in a specified location.
        If None, data are stored in home directory.
        A folder called ``cambridge`` will be created within the ``data_dir``.
    resume : :obj:`bool`, optional
        Whether to resume download of a partly-downloaded file. Default=True.
    verbose : :obj:`int`, optional
        Defines the level of verbosity of the output. Default=1.

    Returns
    -------
    data : :obj:`sklearn.utils.Bunch`
        Dictionary-like object, the attributes of interest are :

        - ``'func'``: List of paths to nifti files containing downsampled functional MRI data (4D)
          for each subject.
        - ``'confounds'``: List of paths to tsv files containing confounds related to each subject.

    Notes
    -----
    This fetcher downloads preprocessed data that are available on Open Science Framework (OSF):
    https://osf.io/9wcb8/ .
    These data have been partially preprocessed with fMRIPrep v20.2.1.
    Specifically, the "func" files are in native BOLD space and have been slice-timing corrected
    and motion corrected.

    The original, raw data are available on OpenNeuro at
    https://openneuro.org/datasets/ds000258/versions/1.0.0 .

    Warning
    -------
    The grouping and labeling approach we use for our datasets will not work well with the
    inheritance principle. We also ignore figure and log files in the returned Bunch objects.

    References
    ----------
    .. [1] Power, J., Plitt, M., Gotts, S., Kundu, P., Voon, V., Bandettini, P., & Martin, A.
           (2018).
           Ridding fMRI data of motion-related influences: removal of signals with distinct
           spatial and physical bases in multi-echo data.
           PNAS, 115(9), E2105-2114.
           www.pnas.org/content/115/9/E2105
    .. [2] Esteban, O., Markiewicz, C. J., Blair, R. W., Moodie, C. A., Isik, A. I., Erramuzpe, A.,
           ... & Gorgolewski, K. J. (2019).
           fMRIPrep: a robust preprocessing pipeline for functional MRI.
           Nature methods, 16(1), 111-116.
           https://doi.org/10.1038/s41592-018-0235-4
    .. [3] Esteban, O., Ciric, R., Finc, K., Blair, R. W., Markiewicz, C. J., Moodie, C. A.,
           ...  & Gorgolewski, K. J. (2020).
           Analysis of task-based functional MRI data preprocessed with fMRIPrep.
           Nature protocols, 15(7), 2186-2202.
           https://doi.org/10.1038/s41596-020-0327-3

    See Also
    --------
    tedana.datasets.utils.get_data_dirs
    """
    DATASET_NAME = "cambridge"
    data_dir = _get_dataset_dir(DATASET_NAME, data_dir=data_dir, verbose=verbose)

    # Dataset description
    package_directory = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(package_directory, "data", DATASET_NAME + ".rst"), "r") as rst_file:
            fdescr = rst_file.read()
    except IOError:
        fdescr = ""

    config_file = os.path.join(package_directory, "data", DATASET_NAME + ".json")
    with open(config_file, "r") as fo:
        config_data = json.load(fo)

    if isinstance(groups, str):
        groups = (groups,)

    # files_to_download = select_files(config_data, n_subjects=n_subjects, groups=groups)
    # Reduce by groups
    reduced_config_data = {
        k: v for k, v in config_data.items() if any(g in v["groups"] for g in groups)
    }

    all_files = sorted(list(reduced_config_data.keys()))
    subject_search = "(sub-[a-zA-Z0-9]+)"
    general_files = [f for f in all_files if not re.findall(subject_search, f)]
    subject_files = [f for f in all_files if re.findall(subject_search, f)]
    subjects = [re.findall(subject_search, f)[0] for f in subject_files]
    subjects = sorted(list(set(subjects)))

    if n_subjects > len(subjects):
        LGR.warning(
            f"{n_subjects} requested, but only {len(subjects)} available. "
            f"Downloading {len(subjects)}."
        )
        n_subjects = len(subjects)

    subjects = subjects[:n_subjects]
    selected_files = [f for f in subject_files if any(sub in f for sub in subjects)]
    selected_files += general_files
    selected_files = sorted(selected_files)

    # Should probably *download* the selected files here.

    # bunch = group_files(selected_files, description=fdescr)
    grouped_files = {}
    reduced_config_data_again = {
        k: v for k, v in reduced_config_data.items() if k in selected_files
    }
    unique_types = sorted(list(set([v["type"] for v in reduced_config_data.values()])))
    unique_types = [v for v in unique_types if v]  # Drop empty ("") types
    for type_ in unique_types:
        grouped_files[type_] = []
        type_files = [f for f in selected_files if reduced_config_data_again[f]["type"] == type_]

        type_general_files = sorted([f for f in type_files if f in general_files])
        if len(type_general_files) == 1:
            type_general_files = type_general_files[0]

        if type_general_files:
            # must have been a subject-wise file
            grouped_files[type_].append(type_general_files)
        else:
            for subject in subjects:
                subject_type_files = sorted([f for f in type_files if subject in f])
                if len(subject_type_files) == 1:
                    subject_type_files = subject_type_files[0]

                grouped_files[type_].append(subject_type_files)

    grouped_files["participant_id"] = subjects

    # Would be great to extract useful metadata here (esp. EchoTimes) and add it to the Bunch.

    bunch = Bunch(description=fdescr, **grouped_files)

    return bunch
