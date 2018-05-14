"""
Tests for tedana.
"""

import os.path
import numpy as np
import nibabel as nb
from pathlib import Path
from tedana.cli import run_tedana
from tedana import workflows


def test_basic_tedana():
    """
    A very simple test, to confirm that tedana creates output
    files.
    """

    parser = run_tedana.get_parser()
    options = parser.parse_args(['-d', '/home/neuro/data/zcat_ffd.nii.gz',
                                 '-e', '14.5', '38.5', '62.5'])
    workflows.tedana(**vars(options))
    assert os.path.isfile('comp_table.txt')


def compare_nifti(fn, test_dir, res_dir):
    """
    Helper function to compare two niftis
    """
    res_fp = (res_dir/fn).as_posix()
    test_fp = (test_dir/fn).as_posix()
    assert np.allclose(nb.load(res_fp).get_data(), nb.load(test_fp).get_data())


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
    for fn in nifti_test_list:
        compare_nifti(fn, Path('/home/neuro/data/TED/'), Path('/home/neuro/code/TED/'))
