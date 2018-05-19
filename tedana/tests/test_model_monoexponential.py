"""
Tests for tedana.model.monoexponential
"""
import os.path as op

import numpy as np
import nibabel as nib

from tedana import utils
from tedana.model import monoexponential as me


def test_fit_decay():
    tes = np.array([14.5, 38.5, 62.5])
    true_dir = '/home/neuro/data/TED/'
    in_file = '/home/neuro/data/zcat_ffd.nii.gz'
    data, ref_img = utils.load_data(in_file, n_echos=len(tes))
    mask, mask_sum = utils.make_adaptive_mask(data, minimum=False, getsum=True)
    t2sv, s0v, t2ss, s0vs, t2svG, s0vG = me.fit_decay(data, tes, mask,
                                                      mask_sum)
    true_data = nib.load(op.join(true_dir, 't2sv.nii')).get_data()
    test_data = t2sv.reshape(ref_img.shape[:3] + t2sv.shape[1:])
    assert np.allclose(true_data, test_data)
    true_data = nib.load(op.join(true_dir, 's0v.nii')).get_data()
    test_data = s0v.reshape(ref_img.shape[:3] + s0v.shape[1:])
    assert np.allclose(true_data, test_data)
    true_data = nib.load(op.join(true_dir, 't2ss.nii')).get_data()
    test_data = t2ss.reshape(ref_img.shape[:3] + t2ss.shape[1:])
    assert np.allclose(true_data, test_data)
    true_data = nib.load(op.join(true_dir, 's0vs.nii')).get_data()
    test_data = s0vs.reshape(ref_img.shape[:3] + s0vs.shape[1:])
    assert np.allclose(true_data, test_data)
    true_data = nib.load(op.join(true_dir, 't2svG.nii')).get_data()
    test_data = t2svG.reshape(ref_img.shape[:3] + t2svG.shape[1:])
    assert np.allclose(true_data, test_data)
    true_data = nib.load(op.join(true_dir, 's0vG.nii')).get_data()
    test_data = s0vG.reshape(ref_img.shape[:3] + s0vG.shape[1:])
    assert np.allclose(true_data, test_data)


def test_fit_decay_ts():
    """
    fit_decay_ts should return data in samples x time shape.
    """
    tes = np.array([14.5, 38.5, 62.5])
    true_dir = '/home/neuro/data/TED/'
    in_file = '/home/neuro/data/zcat_ffd.nii.gz'
    data, ref_img = utils.load_data(in_file, n_echos=len(tes))
    mask, mask_sum = utils.make_adaptive_mask(data, minimum=False, getsum=True)
    t2sv, s0v, t2svG, s0vG = me.fit_decay_ts(data, tes, mask, mask_sum)
    assert len(t2sv.shape) == 2
    assert len(s0v.shape) == 2
    assert len(t2svG.shape) == 2
    assert len(s0vG.shape) == 2
