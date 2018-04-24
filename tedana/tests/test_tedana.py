"""
Tests for tedana.
"""

import os.path
from tedana.interfaces import tedana
from tedana.cli import run
import nibabel as nb
import numpy as np
from pathlib import Path


def test_basic_tedana():
    """
    A very simple test, to confirm that tedana creates output
    files.
    """

    parser = run.get_parser()
    options = parser.parse_args(['-d', '/home/neuro/data/zcat_ffd.nii.gz',
                                 '-e', '14.5', '38.5', '62.5'])
    tedana.main(options)
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
     '.cc_temp.nii.gz',
     '.fcl_in.nii.gz',
     '.fcl_out.nii.gz',
     '__clin.nii.gz',
     '__clout.nii.gz',
     'betas_hik_OC.nii',
     'betas_hik_OC_T1c.nii',
     'betas_OC.nii',
     'dn_ts_OC.nii',
     'dn_ts_OC_T1c.nii',
     'feats_OC2.nii',
     'hik_ts_OC.nii',
     'hik_ts_OC_T1c.nii',
     'lowk_ts_OC.nii',
     'midk_ts_OC.nii',
     's0v.nii',
     's0vG.nii',
     's0vs.nii',
     'sphis_hik.nii',
     'T1gs.nii',
     't2ss.nii',
     't2sv.nii',
     't2svG.nii',
     'ts_OC.nii',
     'tsoc_nogs.nii',
     'tsoc_orig.nii',
     'veins_l0.nii',
     'veins_l1.nii']
    test_dir = Path('/home/neuro/data/TED/')
    res_dir = Path('/home/neuro/code/TED/')
    for fn in nifti_test_list:
        compare_nifti(fn, test_dir, res_dir)
