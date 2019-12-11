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
    catd = np.empty((n_samples, n_echos, n_vols))
    optcom = np.empty((n_samples, n_vols))
    ref_img = ''

    catd = np.empty((n_samples + 1, n_echos, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(catd=catd, optcom=optcom, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('First dimensions of catd ({0}) and optcom ({1}) do not '
                                 'match'.format(catd.shape[0], optcom.shape[0]))

    catd = np.empty((n_samples, n_echos + 1, n_vols))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(catd=catd, optcom=optcom, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('Second dimension of catd ({0}) does not match '
                                 'n_echos ({1})'.format(catd.shape[1], n_echos))

    catd = np.empty((n_samples, n_echos, n_vols))
    optcom = np.empty((n_samples, n_vols + 1))
    with pytest.raises(ValueError) as e_info:
        gsc.gscontrol_raw(catd=catd, optcom=optcom, n_echos=n_echos,
                          ref_img=ref_img, dtrank=4)
    assert str(e_info.value) == ('Third dimension of catd ({0}) does not match '
                                 'second dimension of optcom '
                                 '({1})'.format(catd.shape[2], optcom.shape[1]))
