"""
Tests for tedana.utils
"""

from os.path import join as pjoin, dirname

import nibabel as nib
import numpy as np
import pytest

from tedana import (utils, io)

rs = np.random.RandomState(1234)
datadir = pjoin(dirname(__file__), 'data')
fnames = [pjoin(datadir, 'echo{}.nii.gz'.format(n)) for n in range(1, 4)]
tes = ['14.5', '38.5', '62.5']


def test_unmask():
    # generate boolean mask + get number of True values
    mask = rs.choice([0, 1], size=(100,)).astype(bool)
    n_data = mask.sum()

    inputs = [
        (rs.rand(n_data, 3), float),  # 2D float
        (rs.rand(n_data, 3, 3), float),  # 3D float
        (rs.randint(10, size=(n_data, 3)), int),  # 2D int
        (rs.randint(10, size=(n_data, 3, 3)), int)  # 3D int
    ]

    for (input, dtype) in inputs:
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
        utils.andb([rs.randint(10, size=(10, 10)),
                    rs.randint(10, size=(20, 20))])


def test_load_image():
    fimg = nib.load(fnames[0])
    exp_shape = (64350, 5)

    # load filepath to image
    assert utils.load_image(fnames[0]).shape == exp_shape
    # load img_like object
    assert utils.load_image(fimg).shape == exp_shape
    # load array
    assert utils.load_image(fimg.get_data()).shape == exp_shape


def test_make_adaptive_mask():
    # load data make masks
    data = io.load_data(fnames, n_echos=len(tes))[0]
    mask, masksum = utils.make_adaptive_mask(data, getsum=True)

    # getsum doesn't change mask values
    assert np.allclose(mask, utils.make_adaptive_mask(data))
    # shapes are all the same
    assert mask.shape == masksum.shape == (64350,)
    assert np.allclose(mask, (masksum >= 3).astype(bool))
    # mask has correct # of entries
    assert mask.sum() == 41749
    # masksum has correct values
    vals, counts = np.unique(masksum, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([13564,  3977,  5060, 41749]))

    # test user-defined mask
    # TODO: Add mask file with no bad voxels to test against
    mask, masksum = utils.make_adaptive_mask(data, mask=pjoin(datadir,
                                                              'mask.nii.gz'),
                                             getsum=True)
    assert np.allclose(mask, (masksum >= 3).astype(bool))
