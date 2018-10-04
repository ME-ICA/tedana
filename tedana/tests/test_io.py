"""
Tests for tedana.io
"""

import nibabel as nib

import tedana.io
from tedana import utils
from tedana.tests.test_utils import fnames, tes


def test_new_nii_like():
    data, ref = utils.load_data(fnames, n_echos=len(tes))
    nimg = tedana.io.new_nii_like(ref, data)

    assert isinstance(nimg, nib.Nifti1Image)
    assert nimg.shape == (39, 50, 33, 3, 5)


def test_filewrite():
    pass
