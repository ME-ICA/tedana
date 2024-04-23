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
    """Test tedana.utils.make_adaptive_mask with different methods."""
    # load data make masks
    mask_file = pjoin(datadir, "mask.nii.gz")
    data = io.load_data(fnames, n_echos=len(tes))[0]

    # Add in simulated values
    base_val = np.mean(data[:, 0, :])  # mean value of first echo
    idx = 5457
    # Three good echoes (3)
    data[idx, :, :] = np.array([base_val, base_val - 1, base_val - 2])[:, None]
    # Dropout: good bad good (3)
    # Decay: good good bad (2)
    data[idx + 1, :, :] = np.array([base_val, 1, base_val])[:, None]
    # Dropout: good bad bad (1)
    # Decay: good good bad (2)
    data[idx + 2, :, :] = np.array([base_val, 1, 1])[:, None]
    # Dropout: good good good (3)
    # Decay: good bad bad (1)
    data[idx + 3, :, :] = np.array([base_val, base_val, base_val])[:, None]
    # Dropout: bad bad bad (0)
    # Decay: good good good (3)
    data[idx + 4, :, :] = np.array([1, 0.9, 0.8])[:, None]
    # Base: good good bad (2)
    # Dropout: bad bad bad (0)
    # Decay: good good good (3)
    data[idx + 5, :, :] = np.array([1, 0.9, -1])[:, None]

    # Just dropout method
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        mask=mask_file,
        threshold=1,
        methods=["dropout"],
    )

    assert mask.shape == adaptive_mask.shape == (64350,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 3
    assert adaptive_mask[idx + 2] == 1
    assert adaptive_mask[idx + 3] == 3
    assert adaptive_mask[idx + 4] == 0
    assert adaptive_mask[idx + 5] == 0
    assert mask.sum() == 49374
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([14976, 1817, 4427, 43130]))

    # Just decay method
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        mask=mask_file,
        threshold=1,
        methods=["decay"],
    )

    assert mask.shape == adaptive_mask.shape == (64350,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 2
    assert adaptive_mask[idx + 2] == 2
    assert adaptive_mask[idx + 3] == 1
    assert adaptive_mask[idx + 4] == 3
    assert adaptive_mask[idx + 5] == 2
    assert mask.sum() == 60985  # This method can't flag first echo as bad
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([3365, 4366, 5973, 50646]))

    # Dropout and decay methods combined
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        mask=mask_file,
        threshold=1,
        methods=["dropout", "decay"],
    )

    assert mask.shape == adaptive_mask.shape == (64350,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 2
    assert adaptive_mask[idx + 2] == 1
    assert adaptive_mask[idx + 3] == 1
    assert adaptive_mask[idx + 4] == 0
    assert adaptive_mask[idx + 5] == 0
    assert mask.sum() == 49374
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([14976, 3111, 6248, 40015]))

    # Adding "none" should have no effect
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        mask=mask_file,
        threshold=1,
        methods=["dropout", "decay", "none"],
    )

    assert mask.shape == adaptive_mask.shape == (64350,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 2
    assert adaptive_mask[idx + 2] == 1
    assert adaptive_mask[idx + 3] == 1
    assert adaptive_mask[idx + 4] == 0
    assert adaptive_mask[idx + 5] == 0
    assert mask.sum() == 49374
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([14976, 3111, 6248, 40015]))

    # Just "none"
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        mask=mask_file,
        threshold=1,
        methods=["none"],
    )

    assert mask.shape == adaptive_mask.shape == (64350,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 3
    assert adaptive_mask[idx + 2] == 3
    assert adaptive_mask[idx + 3] == 3
    assert adaptive_mask[idx + 4] == 3
    assert adaptive_mask[idx + 5] == 2
    assert mask.sum() == 60985
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([3365, 1412, 1195, 58378]))


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

    Note: make_adaptive_mask has optional paramters - mask and threshold.
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))
    mask = np.random.randint(2, size=n_samples)

    assert utils.make_adaptive_mask(data, mask=mask, methods=["dropout", "decay"]) is not None


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
