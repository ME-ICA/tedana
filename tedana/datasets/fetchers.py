import os
import numpy as np
from sklearn.utils import Bunch
from nilearn.datasets.utils import _fetch_files
from nilearn._utils.numpy_conversions import csv_to_array

from .utils import _get_dataset_dir


def _reduce_confounds(regressors, keep_confounds):
    reduced_regressors = []
    for in_file in regressors:
        out_file = in_file.replace('desc-confounds',
                                   'desc-reducedConfounds')
        if not os.path.isfile(out_file):
            confounds = np.recfromcsv(in_file, delimiter='\t')
            selected_confounds = confounds[keep_confounds]
            header = '\t'.join(selected_confounds.dtype.names)
            np.savetxt(out_file, np.array(selected_confounds.tolist()),
                       fmt=('%.18e %.18e %.18e %.18e %.18e %.18e %f '
                            '%.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e'),
                       header=header, delimiter='\t', comments='')
        reduced_regressors.append(out_file)
    return reduced_regressors


def _fetch_cambridge_functional(n_subjects, data_dir, url, resume,
                                verbose):
    """Helper function to fetch_cambridge.
    This function helps in downloading multi-echo functional MRI data
    for each subject in the Cambridge dataset.
    Files are downloaded from Open Science Framework (OSF).
    For more information on the data and its preprocessing, see:
    https://osf.io/9wcb8/

    Parameters
    ----------
    n_subjects : int
        The number of subjects to load. If None, all the subjects are
        loaded. Total 88 subjects.
    data_dir : str
        Path of the data directory. Used to force data storage in a specified
        location. If None is given, data are stored in home directory.
    url : str
        Override download URL. Used for test only (or if you setup a mirror of
        the data).
    resume : bool
        Whether to resume download of a partly-downloaded file.
    verbose : int
        Defines the level of verbosity of the output.
    Returns
    -------
    func : list of str (Nifti files)
        Paths to functional MRI data (4D) for each subject.
    """
    dataset_name = 'cambridge'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    if url is None:
        # Download from the relevant OSF project, using hashes generated
        # from the OSF API. Note the trailing slash. For more info, see:
        # https://gist.github.com/emdupre/3cb4d564511d495ea6bf89c6a577da74
        url = 'https://osf.io/download/{}/'

    func = '{0}_task-rest_{1}_space-scanner_desc-partialPreproc_bold.nii.gz'

    # The gzip contains unique download keys per Nifti file and confound
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('participant_id', 'U12'), ('echo_id', 'U12'),
             ('key_bold', 'U24')]
    names = ['participant_id', 'echo_id', 'key_b']
    # csv file contains download information related to OpenScience(osf)
    osf_data = csv_to_array(os.path.join(package_directory, "data",
                                         "cambridge_echos.csv"),
                            skip_header=True, dtype=dtype, names=names)
    funcs = []
    participant_id, echo_id, uuid = zip(*osf_data)
    participants = np.unique(participant_id)[:n_subjects]

    for participant_id in participants:
        this_osf_id = osf_data[osf_data['participant_id'] == participant_id]
        participant_funcs = []

        for entry in this_osf_id:
            echo_id = entry['echo_id']
            # Download bold images for each echo
            func_url = url.format(entry['key_b'])
            func_file = [(func.format(participant_id, echo_id),
                          func_url,
                          {'move': func.format(participant_id, echo_id)})]
            path_to_func = _fetch_files(data_dir, func_file, resume=resume,
                                        verbose=verbose)[0]
            participant_funcs.append(path_to_func)
        funcs.append[tuple(participant_funcs)]
    return funcs


def _fetch_cambridge_regressors(n_subjects, data_dir, url, resume,
                                verbose):
    """Helper function to fetch_cambridge.
    This function helps in downloading the regressor time series for each run
    of multi-echo functional MRI data in the Cambridge dataset.
    Files are downloaded from Open Science Framework (OSF).
    For more information on the data and its preprocessing, see:
    https://osf.io/9wcb8/

    Parameters
    ----------
    n_subjects : int
        The number of subjects to load. If None, all the subjects are
        loaded. Total 88 subjects.
    data_dir : str
        Path of the data directory. Used to force data storage in a specified
        location. If None is given, data are stored in home directory.
    url : str
        Override download URL. Used for test only (or if you setup a mirror of
        the data).
    resume : bool
        Whether to resume download of a partly-downloaded file.
    verbose : int
        Defines the level of verbosity of the output.
    Returns
    -------
    regressors : list of str (tsv files)
        Paths to regressors related to each subject.
    """
    dataset_name = 'cambridge'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    if url is None:
        # Download from the relevant OSF project, using hashes generated
        # from the OSF API. Note the trailing slash. For more info, see:
        # https://gist.github.com/emdupre/3cb4d564511d495ea6bf89c6a577da74
        url = 'https://osf.io/download/{}/'

    regr = '{0}_task-rest_desc-confounds_timeseries.tsv'

    # The gzip contains unique download keys per Nifti file and confound
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('participant_id', 'U12'), ('key_regressor', 'U24')]
    names = ['participant_id', 'key_r']
    # csv file contains download information related to OpenScience(osf)
    osf_data = csv_to_array(os.path.join(package_directory, "data",
                                         "cambridge_confounds.csv"),
                            skip_header=True, dtype=dtype, names=names)
    regressors = []

    for participant_id, key_r in osf_data[:n_subjects]:
        # Download regressor files
        regr_url = url.format(key_r)
        regr_file = [(regr.format(participant_id), regr_url,
                      {'move': regr.format(participant_id)})]
        path_to_regr = _fetch_files(data_dir, regr_file, resume=resume,
                                    verbose=verbose)[0]
        regressors.append(path_to_regr)
    return regressors


def fetch_cambridge(n_subjects=None, reduce_confounds=True,
                    data_dir=None, resume=True, verbose=1):
    """Fetch Cambridge multi-echo data.
    See Notes below for more information on this dataset.
    Please cite [1]_ if you are using this dataset.

    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load. If None, all the subjects are
        loaded. Total 88 subjects.
    reduce_confounds : bool, optional
        If True, the returned confounds only include 6 motion parameters,
        mean framewise displacement, signal from white matter, csf, and
        6 anatomical compcor parameters. This selection only serves the
        purpose of having realistic examples. Depending on your research
        question, other confounds might be more appropriate.
        If False, returns all fmriprep confounds.
        Default=True.
    data_dir : str, optional
        Path of the data directory. Used to force data storage in a specified
        location. If None, data are stored in home directory.
    resume : bool, optional
        Whether to resume download of a partly-downloaded file.
        Default=True.
    verbose : int, optional
        Defines the level of verbosity of the output. Default=1.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        - 'func': list of str (Nifti files)
            Paths to downsampled functional MRI data (4D) for each subject.
        - 'confounds': list of str (tsv files)
            Paths to confounds related to each subject.

    Notes
    -----
    The original data is downloaded from OpenNeuro
    https://openneuro.org/datasets/ds000258/versions/1.0.0
    This fetcher downloads preprocessed data that are available on Open
    Science Framework (OSF): https://osf.io/9wcb8/

    References
    ----------
    .. [1] Power, J., Plitt, M., Gotts, S., Kundu, P., Voon, V.,
       Bandettini, P., & Martin, A. (2018).
       Ridding fMRI data of motion-related influences:
       removal of signals with distinct spatial and physical bases
       in multi-echo data.
       PNAS, 115(9), E2105-2114.
       www.pnas.org/content/115/9/E2105
    """
    dataset_name = 'cambridge'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=1)
    keep_confounds = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y',
                      'rot_z', 'framewise_displacement', 'a_comp_cor_00',
                      'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
                      'a_comp_cor_04', 'a_comp_cor_05', 'csf',
                      'white_matter']

    # Dataset description
    package_directory = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(
                    package_directory, 'data', dataset_name + '.rst'),
                  'r') as rst_file:
            fdescr = rst_file.read()
    except IOError:
        fdescr = ''

    funcs = _fetch_cambridge_functional(
        n_subjects, data_dir=data_dir, url=None,
        resume=resume, verbose=verbose)

    regressors = _fetch_cambridge_regressors(
        n_subjects, data_dir=data_dir, url=None,
        resume=resume, verbose=verbose)
    if reduce_confounds:
        regressors = _reduce_confounds(regressors, keep_confounds)

    return Bunch(func=funcs, confounds=regressors, description=fdescr)
