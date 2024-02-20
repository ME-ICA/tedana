"""Tests for tedana.utils."""

import random
from os.path import dirname
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import pytest

from tedana import io, utils

rs = np.random.RandomState(1234)
datadir = pjoin(dirname(__file__), "data")
fnames = [pjoin(datadir, f"echo{n}.nii.gz") for n in range(1, 4)]
tes = ["14.5", "38.5", "62.5"]


def test_unmask():
    # generate boolean mask + get number of True values
    mask = rs.choice([0, 1], size=(100,)).astype(bool)
    n_data = mask.sum()

    inputs = [
        (rs.rand(n_data, 3), float),  # 2D float
        (rs.rand(n_data, 3, 3), float),  # 3D float
        (rs.randint(10, size=(n_data, 3)), int),  # 2D int
        (rs.randint(10, size=(n_data, 3, 3)), int),  # 3D int
    ]

    for input, dtype in inputs:
        out = utils.unmask(input, mask)
        assert out.shape == (100,) + input.shape[1:]
        assert out.dtype == dtype


def test_dice():
    arr = rs.choice([0, 1], size=(100, 100))
    # identical arrays should have a Dice index of 1
    assert utils.dice(arr, arr) == 1.0
    # inverted arrays should have a Dice index of 0
    assert utils.dice(arr, np.logical_not(arr)) == 0.0
    # float arrays should return same as integer or boolean array
    assert utils.dice(arr * rs.rand(100, 100), arr * rs.rand(100, 100)) == 1.0
    # empty arrays still work
    assert utils.dice(np.array([]), np.array([])) == 0.0
    # different size arrays raise a ValueError
    with pytest.raises(ValueError):
        utils.dice(arr, rs.choice([0, 1], size=(20, 20)))


def test_andb():
    # test with a range of dimensions and ensure output dtype is int
    for ndim in range(1, 5):
        shape = (10,) * ndim
        out = utils.andb([rs.randint(10, size=shape) for f in range(5)])
        assert out.shape == shape
        assert out.dtype == int

    # confirm error raised when dimensions are not the same
    with pytest.raises(ValueError):
        utils.andb([rs.randint(10, size=(10, 10)), rs.randint(10, size=(20, 20))])


def test_reshape_niimg():
    fimg = nib.load(fnames[0])
    exp_shape = (64350, 5)

    # load filepath to image
    assert utils.reshape_niimg(fnames[0]).shape == exp_shape
    # load img_like object
    assert utils.reshape_niimg(fimg).shape == exp_shape
    # load array
    assert utils.reshape_niimg(fimg.get_fdata()).shape == exp_shape


def test_make_adaptive_mask():
    # load data make masks
    data = io.load_data(fnames, n_echos=len(tes))[0]
    mask, masksum = utils.make_adaptive_mask(data, getsum=True, threshold=1)

    # getsum doesn't change mask values
    assert np.allclose(mask, utils.make_adaptive_mask(data))
    # shapes are all the same
    assert mask.shape == masksum.shape == (64350,)
    assert np.allclose(mask, (masksum >= 1).astype(bool))
    # mask has correct # of entries
    assert mask.sum() == 50786
    # masksum has correct values
    vals, counts = np.unique(masksum, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([13564, 3977, 5060, 41749]))

    # test user-defined mask
    # TODO: Add mask file with no bad voxels to test against
    mask, masksum = utils.make_adaptive_mask(
        data, mask=pjoin(datadir, "mask.nii.gz"), getsum=True, threshold=3
    )
    assert np.allclose(mask, (masksum >= 3).astype(bool))


# SMOKE TESTS


def test_smoke_reshape_niimg():
    """Ensure that reshape_niimg returns reasonable objects with random inputs.

    in the correct format.

    Note: reshape_niimg could take in 3D or 4D array.
    """
    data_3d = np.random.random((100, 5, 20))
    data_4d = np.random.random((100, 5, 20, 50))

    assert utils.reshape_niimg(data_3d) is not None
    assert utils.reshape_niimg(data_4d) is not None

    with pytest.raises(TypeError):
        utils.reshape_niimg(5)

    with pytest.raises(ValueError):
        utils.reshape_niimg("/path/to/nonexistent/file")


def test_smoke_make_adaptive_mask():
    """Ensure that make_adaptive_mask returns reasonable objects with random inputs.

    in the correct format.

    Note: make_adaptive_mask has optional paramters - mask and getsum.
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    mask = np.random.randint(2, size=n_samples)

    assert utils.make_adaptive_mask(data) is not None
    assert utils.make_adaptive_mask(data, mask=mask) is not None  # functions with mask
    assert utils.make_adaptive_mask(data, getsum=True) is not None  # functions when getsumis true


def test_smoke_unmask():
    """Ensure that unmask returns reasonable objects with random inputs.

    in the correct format.

    Note: unmask could take in 1D or 2D or 3D arrays.
    """
    data_1d = np.random.random(100)
    data_2d = np.random.random((100, 5))
    data_3d = np.random.random((100, 5, 20))
    mask = np.random.randint(2, size=100)

    assert utils.unmask(data_1d, mask) is not None
    assert utils.unmask(data_2d, mask) is not None
    assert utils.unmask(data_3d, mask) is not None


def test_smoke_dice():
    """Ensure that dice returns reasonable objects with random inputs.

    in the correct format.

    Note: two arrays must be in the same length.
    """
    arr1 = np.random.random(100)
    arr2 = np.random.random(100)

    assert utils.dice(arr1, arr2) is not None


def test_smoke_andb():
    """Ensure that andb returns reasonable objects with random inputs.

    in the correct format.
    """
    arr = np.random.random((100, 10)).tolist()  # 2D list of "arrays"

    assert utils.andb(arr) is not None


def test_smoke_get_spectrum():
    """Ensure that get_spectrum returns reasonable objects with random inputs.

    in the correct format.
    """
    data = np.random.random(100)
    tr = random.random()

    spectrum, freqs = utils.get_spectrum(data, tr)
    assert spectrum is not None
    assert freqs is not None


def test_smoke_threshold_map():
    """Ensure that threshold_map returns reasonable objects with random inputs.

    in the correct format.

    Note: using 3D array as img, some parameters are optional and are all tested.
    """
    img = np.random.random((10, 10, 10))  # 3D array must of of size S
    min_cluster_size = random.randint(1, 100)

    threshold = random.random()
    mask = np.random.randint(2, size=1000)

    assert utils.threshold_map(img, min_cluster_size) is not None

    # test threshold_map with different optional parameters
    assert utils.threshold_map(img, min_cluster_size, threshold=threshold) is not None
    assert utils.threshold_map(img, min_cluster_size, mask=mask) is not None
    assert utils.threshold_map(img, min_cluster_size, binarize=False) is not None
    assert utils.threshold_map(img, min_cluster_size, sided="one") is not None
    assert utils.threshold_map(img, min_cluster_size, sided="bi") is not None


def test_sec2millisec():
    """Ensure that sec2millisec returns 1000x the input values."""
    assert utils.sec2millisec(5) == 5000
    assert utils.sec2millisec(np.array([5])) == np.array([5000])


def test_millisec2sec():
    """Ensure that millisec2sec returns 1/1000x the input values."""
    assert utils.millisec2sec(5000) == 5
    assert utils.millisec2sec(np.array([5000])) == np.array([5])


# TODO: "BREAK" AND UNIT TESTS
