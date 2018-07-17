"""
Tests for tedana.
"""

import os.path
import numpy as np
import nibabel as nb
from pathlib import Path
from tedana.cli import run_t2smap
from tedana import workflows


def test_basic_t2smap():
    """
    A very simple test, to confirm that t2smap creates output
    files.
    """
    parser = run_t2smap.get_parser()
    options = parser.parse_args(['-d', '/home/neuro/data/zcat_ffd.nii.gz',
                                 '-e', '14.5', '38.5', '62.5',
                                 '--label', 't2smap0'])
    workflows.t2smap(**vars(options))
    assert os.path.isfile('/home/neuro/code/TED.zcat_ffd.t2smap0/ts_OC.nii')


def test_basic_t2smap2():
    """
    A very simple test, to confirm that t2smap creates output
    files when fitmode is set to ts.
    """
    parser = run_t2smap.get_parser()
    options = parser.parse_args(['-d', '/home/neuro/data/zcat_ffd.nii.gz',
                                 '-e', '14.5', '38.5', '62.5',
                                 '--fitmode', 'ts',
                                 '--label', 't2smap1'])
    workflows.t2smap(**vars(options))
    assert os.path.isfile('/home/neuro/code/TED.zcat_ffd.t2smap0/ts_OC.nii')


def test_basic_t2smap3():
    """
    A very simple test, to confirm that t2smap creates output
    files when combmode is set to 'ste'.
    """
    parser = run_t2smap.get_parser()
    options = parser.parse_args(['-d', '/home/neuro/data/zcat_ffd.nii.gz',
                                 '-e', '14.5', '38.5', '62.5',
                                 '--combmode', 'ste',
                                 '--label', 't2smap2'])
    workflows.t2smap(**vars(options))
    assert os.path.isfile('/home/neuro/code/TED.zcat_ffd.t2smap2/ts_OC.nii')


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
        't2svG.nii',
        's0vG.nii',
        'ts_OC.nii',
    ]
    for fn in nifti_test_list:
        compare_nifti(fn, Path('/home/neuro/data/TED/'),
                      Path('/home/neuro/code/TED.zcat_ffd.t2smap0/'))
