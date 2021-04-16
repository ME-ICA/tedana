import os
from sklearn.utils import Bunch
from nilearn.datasets.utils import (_get_dataset_dir, _fetch_files,
                                    _get_dataset_descr, _read_md5_sum_file,
                                    _tree, _filter_columns, _fetch_file,
                                    _uncompress_file)
from nilearn._utils.numpy_conversions import csv_to_array


def _fetch_cambridge_functional(participants, data_dir, url, resume,
                                verbose):
    """Helper function to fetch_cambridge.
    This function helps in downloading multi-echo functional MRI data
    for each subject in the Cambridge dataset.
    Files are downloaded from Open Science Framework (OSF).
    For more information on the data and its preprocessing, see:
    https://osf.io/9wcb8/

    Parameters
    ----------
    participants : numpy.ndarray
        Should contain column participant_id which represents subjects id. The
        number of files are fetched based on ids in this column.
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

    for participant_id in participants['participant_id']:
        this_osf_id = osf_data[osf_data['participant_id'] == participant_id]
        # Download bold images
        func_url = url.format(this_osf_id['key_b'][0])
        func_file = [(func.format(participant_id, participant_id), func_url,
                      {'move': func.format(participant_id)})]
        path_to_func = _fetch_files(data_dir, func_file, resume=resume,
                                    verbose=verbose)[0]
        funcs.append(path_to_func)
    return funcs


def fetch_cambridge(n_subjects=None, reduce_confounds=True,
                    data_dir=None, resume=True, verbose=1):
    """Fetch Cambridge multi-echo data.
    See Notes below for more information on this dataset.
    Please cite [1]_ if you are using this dataset.

    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load. If None, all the subjects are
        loaded. Total 155 subjects.
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
    https://openneuro.org/datasets/ds000228/versions/1.0.0
    This fetcher downloads preprocessed data that are available on Open
    Science Framework (OSF): https://osf.io/9wcb8/

    References
    ----------
    .. [1] Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018).
       Development of the social brain from age three to twelve years.
       Nature communications, 9(1), 1027.
       https://www.nature.com/articles/s41467-018-03399-2
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
    fdescr = _get_dataset_descr(dataset_name)
    funcs= _fetch_cambridge_functional(participants,
                                       data_dir=data_dir,
                                       url=None,
                                       resume=resume,
                                       verbose=verbose)

    if reduce_confounds:
        regressors = _reduce_confounds(regressors, keep_confounds)
    return Bunch(func=funcs, confounds=regressors, phenotypic=participants,
                 description=fdescr)