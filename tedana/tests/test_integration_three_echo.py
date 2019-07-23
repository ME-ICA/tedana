from tedana.workflows import tedana_workflow


def test_integration_three_echo():
    """
    An integration test of the full tedana workflow using three-echo test data.
    """
    tedana_workflow(
        data='/tmp/data/three-echo/three_echo_Cornell_zcat.nii.gz',
        tes=[14.5, 38.5, 62.5],
        out_dir='/tmp/data/three-echo/TED.three-echo',
        tedpca='kundu', png=True)
