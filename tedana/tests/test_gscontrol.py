"""
Tests for tedana.model.fit
"""

import numpy as np
import pytest

import tedana.gscontrol as gsc


def test_break_gscontrol_raw():
    """
    Ensure that gscontrol_raw fails when input data do not have the right
    shapes.
    """
    n_samples, n_echos, n_vols = 10000, 4, 100
    data_cat = np.empty((n_samples, n_echos, n_vols))
    data_oc = np.empty((n_samples, n_vols))
    ref_img = ''

    data_cat = np.empty((n_samples+1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(data_cat=data_cat, data_oc=data_oc, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('First dimensions of data_cat ({0}) and data_oc ({1}) do not '
                                 'match'.format(data_cat.shape[0], data_oc.shape[0]))

    data_cat = np.empty((n_samples, n_echos+1, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(data_cat=data_cat, data_oc=data_oc, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('Second dimension of data_cat ({0}) does not match '
                                 'n_echos ({1})'.format(data_cat.shape[1], n_echos))

    data_cat = np.empty((n_samples, n_echos, n_vols))
    data_oc = np.empty((n_samples, n_vols+1))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(data_cat=data_cat, data_oc=data_oc, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('Third dimension of data_cat ({0}) does not match '
                                 'second dimension of data_oc '
                                 '({1})'.format(data_cat.shape[2], data_oc.shape[1]))
