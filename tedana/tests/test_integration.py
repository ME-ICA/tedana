import pytest
from tedana.workflows import tedana_workflow


def test_integration_five_echo(skip_integration, include_five_echo):
    """
    An integration test of the full tedana workflow using five-echo test data.
    """
    if skip_integration or not include_five_echo:
        pytest.skip('Skipping five-echo integration')
    tedana_workflow(
        data='/tmp/data/five-echo/p06.SBJ01_S09_Task11_e[1,2,3,4,5].sm.nii.gz',
        tes=[15.4, 29.7, 44.0, 58.3, 72.6],
        out_dir='/tmp/data/five-echo/TED.five-echo',
        debug=True, verbose=True)

def test_integration_three_echo(skip_integration):
    """
    An integration test of the full tedana workflow using three-echo test data.
    """
    if skip_integration:
        pytest.skip('Skipping three-echo integration.')
    tedana_workflow(
        data='/tmp/data/three-echo/three_echo_Cornell_zcat.nii.gz',
        tes=[14.5, 38.5, 62.5],
        out_dir='/tmp/data/three-echo/TED.three-echo',
        tedpca='kundu', png=True)
