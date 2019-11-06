import pytest
import os.path as op
import re
from tedana.workflows import tedana_workflow


def check_outputs(fname, outpath):
    """
    Checks outputs of integration tests
    """
    # Checks named files
    with open(fname, 'r') as f:
        tocheck = f.read().splitlines()

    for f in tocheck:
        assert op.exists(f)

    # Checks for log file
    logfiles = []
    log_regex = ('^tedana_'
                 '[12][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}:'
                 '[0-9]{2}:[0-9]{2}.txt$')
    outputs = os.listdir(outpath)
    for out in outputs:
        if re.match(log_regex, out):
            logfiles.append(out)

    assert len(logfiles) == 1

def test_integration_five_echo(skip_integration, include_five_echo):
    """
    An integration test of the full tedana workflow using five-echo test data.
    """
    if skip_integration or not include_five_echo:
        pytest.skip('Skipping five-echo integration')
    outputs_file = '/tedana/.circleci/tedana_outputs_verbose.txt'
    out_dir = '/data/five-echo/TED.five-echo'
    # tedana_workflow(
    #     data='/data/five-echo/p06.SBJ01_S09_Task11_e[1,2,3,4,5].sm.nii.gz',
    #     tes=[15.4, 29.7, 44.0, 58.3, 72.6],
    #     out_dir=out_dir,
    #     debug=True, verbose=True)
    check_outputs(outputs_file, out_dir)


def test_integration_three_echo(skip_integration):
    """
    An integration test of the full tedana workflow using three-echo test data.
    """
    if skip_integration:
        pytest.skip('Skipping three-echo integration.')
    outputs_file = '/tedana/.circleci/tedana_outputs.txt'
    out_dir = '/data/three-echo/TED.three-echo'
    tedana_workflow(
        data='/data/three-echo/three_echo_Cornell_zcat.nii.gz',
        tes=[14.5, 38.5, 62.5],
        out_dir=out_dir,
        tedpca='kundu', png=True)
    check_outputs(outputs_file, out_dir)
