from tedana.workflows import tedana_workflow


def test_integration_five_echo():
    """
    An integration test of the full tedana workflow using five-echo test data.
    """
    tedana_workflow(
        data='/tmp/data/five-echo/p06.SBJ01_S09_Task11_e[1,2,3,4,5].sm.nii.gz',
        tes=[15.4, 29.7, 44.0, 58.3, 72.6],
        out_dir='/tmp/data/five-echo/TED.five-echo',
        debug=True, verbose=True)
