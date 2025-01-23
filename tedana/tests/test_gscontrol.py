"""Tests for tedana.model.fit."""

import os

import numpy as np
import pytest

import tedana.gscontrol as gsc
from tedana.io import OutputGenerator
from tedana.tests.utils import get_test_data_path

data_dir = get_test_data_path()
ref_img = os.path.join(data_dir, "mask.nii.gz")


def test_break_gscontrol_raw():
    """Ensure that gscontrol_raw fails when input data do not have the right shapes."""
    n_samples, n_echos, n_vols = 10000, 4, 100
    data_cat = np.empty((n_samples, n_echos, n_vols))
    optcom = np.empty((n_samples, n_vols))
    io_generator = OutputGenerator(ref_img)

    data_cat = np.empty((n_samples + 1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(
            data_cat=data_cat,
            data_optcom=optcom,
            n_echos=n_echos,
            io_generator=io_generator,
            dtrank=4,
        )
    assert str(e_info.value) == (
        f"First dimensions of data_cat ({data_cat.shape[0]}) and data_optcom ({optcom.shape[0]}) "
        "do not match"
    )

    data_cat = np.empty((n_samples, n_echos + 1, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(
            data_cat=data_cat,
            data_optcom=optcom,
            n_echos=n_echos,
            io_generator=io_generator,
            dtrank=4,
        )
    assert str(e_info.value) == (
        f"Second dimension of data_cat ({data_cat.shape[1]}) does not match n_echos ({n_echos})"
    )

    data_cat = np.empty((n_samples, n_echos, n_vols))
    optcom = np.empty((n_samples, n_vols + 1))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(
            data_cat=data_cat,
            data_optcom=optcom,
            n_echos=n_echos,
            io_generator=io_generator,
            dtrank=4,
        )
    assert str(e_info.value) == (
        f"Third dimension of data_cat ({data_cat.shape[2]}) does not match "
        f"second dimension of data_optcom ({optcom.shape[1]})"
    )
