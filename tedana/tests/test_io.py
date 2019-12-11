"""
Tests for tedana.io
"""

import nibabel as nib
import numpy as np
import pytest
import pandas as pd

from tedana import io as me
from tedana.tests.test_utils import fnames, tes

from tedana.tests.utils import get_test_data_path

import os

data_dir = get_test_data_path()


def test_new_nii_like():
    data, ref = me.load_data(fnames, n_echos=len(tes))
    nimg = me.new_nii_like(ref, data)

    assert isinstance(nimg, nib.Nifti1Image)
    assert nimg.shape == (39, 50, 33, 3, 5)


def test_filewrite():
    pass


def test_load_data():
    fimg = [nib.load(f) for f in fnames]
    exp_shape = (64350, 3, 5)

    # list of filepath to images
    d, ref = me.load_data(fnames, n_echos=len(tes))
    assert d.shape == exp_shape
    assert isinstance(ref, nib.Nifti1Image)
    assert np.allclose(ref.get_data(), nib.load(fnames[0]).get_data())

    # list of img_like
    d, ref = me.load_data(fimg, n_echos=len(tes))
    assert d.shape == exp_shape
    assert isinstance(ref, nib.Nifti1Image)
    assert ref == fimg[0]

    # imagine z-cat img
    d, ref = me.load_data(fnames[0], n_echos=3)
    assert d.shape == (21450, 3, 5)
    assert isinstance(ref, nib.Nifti1Image)
    assert ref.shape == (39, 50, 11, 1)

    with pytest.raises(ValueError):
        me.load_data(fnames[0])


# SMOKE TESTS

def test_smoke_split_ts():
    """
    Ensures that split_ts returns output when fed in with random inputs
    Note: classification is ["accepted", "rejected", "ignored"]
    """
    np.random.seed(0)  # seeded because comptable MUST have accepted components
    n_samples = 100
    n_times = 20
    n_components = 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)

    # creating the component table with component as random floats,
    # a "metric," and random classification
    component = np.random.random((n_components))
    metric = np.random.random((n_components))
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    df_data = np.column_stack((component, metric, classification))
    comptable = pd.DataFrame(df_data, columns=['component', 'metric', 'classification'])

    hikts, resid = me.split_ts(data, mmix, mask, comptable)

    assert hikts is not None
    assert resid is not None


def test_smoke_write_split_ts():
    """
    Ensures that write_split_ts writes out the expected files with random input and tear them down
    """
    np.random.seed(0)  # at least one accepted and one rejected, thus all files are generated
    n_samples, n_times, n_components = 64350, 10, 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)
    ref_img = os.path.join(data_dir, 'mask.nii.gz')
    # ref_img has shape of (39, 50, 33) so data is 64350 (39*33*50) x 10
    # creating the component table with component as random floats,
    # a "metric," and random classification
    component = np.random.random((n_components))
    metric = np.random.random((n_components))
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    df_data = np.column_stack((component, metric, classification))
    comptable = pd.DataFrame(df_data, columns=['component', 'metric', 'classification'])

    assert me.write_split_ts(data, mmix, mask, comptable, ref_img) is not None

    # TODO: midk_ts.nii is never generated?
    for filename in ["hik_ts_.nii.gz", "lowk_ts_.nii.gz", "dn_ts_.nii.gz"]:
        # remove all files generated
        try:
            os.remove(filename)
        except OSError:
            print(filename + " not generated")
            pass


def test_smoke_writefeats():
    """
    Ensures that writefeats writes out the expected feature with random
    input, since there is no suffix, remove feats_.nii
    """
    n_samples, n_times, n_components = 64350, 10, 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)
    ref_img = os.path.join(data_dir, 'mask.nii.gz')

    assert me.writefeats(data, mmix, mask, ref_img) is not None

    # this only generates feats_.nii, so delete that
    try:
        os.remove("feats_.nii.gz")
    except OSError:
        print("feats_.nii not generated")
        pass


def test_smoke_filewrite():
    """
    Ensures that filewrite writes out a neuroimage with random input,
    since there is no name, remove the image named .nii
    """
    n_samples, n_times, _ = 64350, 10, 6
    data_1d = np.random.random((n_samples))
    data_2d = np.random.random((n_samples, n_times))
    filename = ""
    ref_img = os.path.join(data_dir, 'mask.nii.gz')

    assert me.filewrite(data_1d, filename, ref_img) is not None
    assert me.filewrite(data_2d, filename, ref_img) is not None

    try:
        os.remove(".nii.gz")
    except OSError:
        print(".nii not generated")


def test_smoke_load_data():
    """
    Ensures that data is loaded when given a random neuroimage
    """
    data = os.path.join(data_dir, 'mask.nii.gz')
    n_echos = 1

    fdata, ref_img = me.load_data(data, n_echos)
    assert fdata is not None
    assert ref_img is not None

# TODO: "BREAK" AND UNIT TESTS
