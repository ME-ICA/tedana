"""
Tests for tedana.model.fit
"""

import os

import numpy as np
import pytest

import tedana.gscontrol as gsc
from tedana.io import OutputGenerator
from tedana.tests.utils import get_test_data_path

data_dir = get_test_data_path()
ref_img = os.path.join(data_dir, "mask.nii.gz")


def test_break_gscontrol_raw():
    """
    Ensure that gscontrol_raw fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_vols = 10000, 4, 100
    catd = np.empty((n_samples, n_echos, n_vols))
    optcom = np.empty((n_samples, n_vols))
    io_generator = OutputGenerator(ref_img)

    catd = np.empty((n_samples + 1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(
            catd=catd, optcom=optcom, n_echos=n_echos, io_generator=io_generator, dtrank=4
        )
    assert str(e_info.value) == (
        "First dimensions of catd ({0}) and optcom ({1}) do not "
        "match".format(catd.shape[0], optcom.shape[0])
    )

    catd = np.empty((n_samples, n_echos + 1, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(
            catd=catd, optcom=optcom, n_echos=n_echos, io_generator=io_generator, dtrank=4
        )
    assert str(e_info.value) == (
        "Second dimension of catd ({0}) does not match "
        "n_echos ({1})".format(catd.shape[1], n_echos)
    )

    catd = np.empty((n_samples, n_echos, n_vols))
    optcom = np.empty((n_samples, n_vols + 1))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(
            catd=catd, optcom=optcom, n_echos=n_echos, io_generator=io_generator, dtrank=4
        )
    assert str(e_info.value) == (
        "Third dimension of catd ({0}) does not match "
        "second dimension of optcom "
        "({1})".format(catd.shape[2], optcom.shape[1])
    )
