"""
Tests for tedana.
"""

import os.path as op
from pathlib import Path

import numpy as np
import nibabel as nib

from tedana import workflows


def test_basic_tedana():
    """
    A very simple test, to confirm that tedana creates output
    files.
    """
    workflows.tedana_workflow([op.expanduser('~/data/zcat_ffd.nii.gz')],
                              [14.5, 38.5, 62.5])
    assert op.isfile('comp_table.txt')


def compare_nifti(fn, test_dir, res_dir):
    """
    Helper function to compare two niftis
    """
    res_fp = (res_dir/fn).as_posix()
    test_fp = (test_dir/fn).as_posix()

    isfile = op.isfile(res_fp)
    if isfile:
        res_dat = nib.load(res_fp).get_data()
        test_dat = nib.load(test_fp).get_data()
        if res_dat.shape == test_dat.shape:
            passed = np.allclose(res_dat, test_dat)
        else:
            passed = False
    else:
        passed = False
    return isfile, passed


def test_outputs():
    """
    Compare the niftis specified in the below list again
    """

    nifti_test_list = [
        't2sv.nii',
        's0v.nii',
        't2ss.nii',
        's0vs.nii',
        't2svG.nii',
        's0vG.nii',
        'T1gs.nii',
        'tsoc_orig.nii',
        'tsoc_nogs.nii',
        'veins_l0.nii',
        'veins_l1.nii',
        'ts_OC.nii',
        'hik_ts_OC.nii',
        'midk_ts_OC.nii',
        'lowk_ts_OC.nii',
        'dn_ts_OC.nii',
        'betas_OC.nii',
        'betas_hik_OC.nii',
        'feats_OC2.nii',
        'betas_hik_OC_T1c.nii',
        'dn_ts_OC_T1c.nii',
        'hik_ts_OC_T1c.nii',
        'sphis_hik.nii'
    ]
    out1, out2 = [], []
    for fn in nifti_test_list:
        isfile, passed = compare_nifti(fn, Path(op.expanduser('~/data/TED/')),
                                       Path(op.expanduser('~/code/TED.zcat_ffd/')))
        if not isfile:
            out1.append(fn)

        if not passed:
            out2.append(fn)

    out_str = 'File DNE: '+', '.join(out1)+'\nNot passed: '+', '.join(out2)
    assert not out1 and not out2, out_str
