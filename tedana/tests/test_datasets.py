"""Tests for tedana.datasets."""
import os

import sklearn

from tedana import datasets
from tedana.datasets.utils import _get_dataset_dir


def test__get_dataset_dir(tmp_path_factory):
    """Check that tedana.datasets.utils._get_dataset_dir finds and creates the expected folders."""
    dataset_name = "test_dataset"
    tmpdir = str(tmp_path_factory.mktemp("test__get_dataset_dir"))
    dataset_dir = _get_dataset_dir(dataset_name, data_dir=tmpdir)
    assert dataset_dir == os.path.join(tmpdir, dataset_name)
    assert os.path.isdir(dataset_dir)
    os.rmdir(dataset_dir)

    os.environ["TEDANA_SHARED_DATA"] = tmpdir
    dataset_dir = _get_dataset_dir(dataset_name)
    assert dataset_dir == os.path.join(tmpdir, dataset_name)
    assert os.path.isdir(dataset_dir)
    os.rmdir(dataset_dir)
    del os.environ["TEDANA_SHARED_DATA"]

    os.environ["TEDANA_DATA"] = tmpdir
    dataset_dir = _get_dataset_dir(dataset_name)
    assert dataset_dir == os.path.join(tmpdir, dataset_name)
    assert os.path.isdir(dataset_dir)
    os.rmdir(dataset_dir)
    del os.environ["TEDANA_DATA"]


def test_fetch_cambridge(tmp_path_factory):
    """Check that tedana.datasets.fetch_cambridge downloads the right files."""
    tmpdir = str(tmp_path_factory.mktemp("test_fetch_cambridge"))
    cambridge_bunch = datasets.fetch_cambridge(
        n_subjects=1,
        data_dir=tmpdir,
        low_resolution=True,
        reduce_confounds=True,
    )
    assert isinstance(cambridge_bunch, sklearn.utils.Bunch)

    # Check the functional files
    assert isinstance(cambridge_bunch["func"], list)
    assert isinstance(cambridge_bunch["func"][0], tuple)
    assert len(cambridge_bunch["func"]) == 1
    assert len(cambridge_bunch["func"][0]) == 4
    assert all(os.path.isfile(f) for f in cambridge_bunch["func"][0])

    # Check the confounds files
    assert isinstance(cambridge_bunch["confounds"], list)
    assert len(cambridge_bunch["confounds"]) == 1
    assert os.path.isfile(cambridge_bunch["confounds"][0])

    # Check the description
    assert isinstance(cambridge_bunch["description"], str)
