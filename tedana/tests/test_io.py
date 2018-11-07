"""
Tests for tedana.io
"""

import nibabel as nib
import numpy as np
import pytest

from tedana import io as me
from tedana.tests.test_utils import fnames, tes


def test_gen_fname():
    truefile = ('sub-01_task-rest_run-01_space-MNI152NLin2009cAsym_'
                'desc-preproc_desc-thing_bold.nii.gz')
    basefile = ('sub-01_task-rest_run-01_echo-1_space-MNI152NLin2009cAsym_'
                'desc-preproc_bold.nii.gz')
    desc = 'thing'
    ext = '.nii.gz'
    testfile = me.gen_fname(basefile, extension=ext, desc=desc)
    assert testfile == truefile


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
    assert ref.shape == (39, 50, 11)

    with pytest.raises(ValueError):
        me.load_data(fnames[0])
