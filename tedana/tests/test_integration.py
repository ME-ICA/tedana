"""
Integration tests for "real" data
"""

import pytest
import os
import re
from tedana.workflows import tedana_workflow


def check_outputs(fname, outpath):
    """
    Checks outputs of integration tests

    Parameters
    ----------
    fname : str
        Path to file with expected outputs
    outpath : str
        Path to output directory generated from integration tests
    """

    # Checks named files
    with open(fname, 'r') as f:
        tocheck = f.read().splitlines()
    for fname in tocheck:
        assert os.path.exists(os.path.join(outpath, fname))

    # Checks for log file
    log_regex = ('^tedana_'
                 '[12][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}:'
                 '[0-9]{2}:[0-9]{2}.txt$')
    logfiles = [out for out in os.listdir(outpath) if re.match(log_regex, out)]
    assert len(logfiles) == 1


def test_integration_five_echo(skip_integration, include_five_echo):
    """
    An integration test of the full tedana workflow using five-echo test data.
    """
    if skip_integration or not include_five_echo:
        pytest.skip('Skipping five-echo integration test')
    out_dir = '/data/five-echo/TED.five-echo'
    out_filename = '/tedana/.circleci/tedana_outputs_verbose.txt'
    tedana_workflow(
        data='/data/five-echo/p06.SBJ01_S09_Task11_e[1,2,3,4,5].sm.nii.gz',
        tes=[15.4, 29.7, 44.0, 58.3, 72.6],
        out_dir=out_dir,
        debug=True, verbose=True)
    check_outputs(out_filename, out_dir)


def test_integration_three_echo(skip_integration):
    """
    An integration test of the full tedana workflow using three-echo test data.
    """
    if skip_integration:
        pytest.skip('Skipping three-echo integration test')
    out_dir = '/data/three-echo/TED.three-echo'
    out_filename='/tedana/.circleci/tedana_outputs.txt'
    tedana_workflow(
        data='/data/three-echo/three_echo_Cornell_zcat.nii.gz',
        tes=[14.5, 38.5, 62.5],
        out_dir=out_dir,
        tedpca='kundu', png=True)
    check_outputs(out_filename, out_dir)
