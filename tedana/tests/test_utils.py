"""
Tests for tedana.utils
"""

import nibabel as nib
import numpy as np
import pytest
from tedana import utils

rs = np.random.RandomState(1234)


def test_get_dtype():
    # various combinations of input types
    good_inputs = [
        (['echo1.func.gii', 'echo2.func.gii', 'echo3.func.gii'], 'GIFTI'),
        ('echo1.func.gii', 'GIFTI'),
        (['echo1.nii.gz', 'echo2.nii.gz', 'echo3.nii.gz'], 'NIFTI'),
        ('echo1.nii.gz', 'NIFTI'),
        (['echo1.unknown', 'echo2.unknown', 'echo3.unknown'], 'OTHER'),
        ('echo1.unknown', 'OTHER'),
        (nib.Nifti1Image(np.zeros((10,)*3),
                         affine=np.diag(np.ones(4))), 'NIFTI')
    ]

    for (input, expected) in good_inputs:
        assert utils.get_dtype(input) == expected

    with pytest.raises(ValueError):  # mixed arrays don't work
        utils.get_dtype(['echo1.func.gii', 'echo1.nii.gz'])

    with pytest.raises(TypeError):  # non-img_like inputs don't work
        utils.get_dtype(rs.rand(100, 100))


def test_getfbounds():
    good_inputs = range(1, 12)
    bad_inputs = [
        (0, ValueError),
        (12, ValueError),
        (10.5, TypeError)
    ]

    for n_echos in good_inputs:
        utils.getfbounds(n_echos)

    for (n_echos, err) in bad_inputs:
        with pytest.raises(err):
            utils.getfbounds(n_echos)


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


def test_fitgaussian():
    # not sure a good way to test this
    # it's straight out of the scipy cookbook, so hopefully its robust?
    assert utils.fitgaussian(rs.rand(100, 100)).size == 5


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


# should tedana ship with test data for these?
def test_load_image():
    pass


def test_load_data():
    pass


def test_make_adaptive_mask():
    pass


def test_make_min_mask():
    pass


def test_filewrite():
    pass


def test_new_nii_like():
    pass


def test_new_gii_like():
    pass


def test_new_gii_darray():
    pass
