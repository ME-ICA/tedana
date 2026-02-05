"""Tests for tedana.utils."""

import random
from os.path import dirname
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import pytest

from tedana import utils

rs = np.random.RandomState(1234)
datadir = pjoin(dirname(__file__), "data")
fnames = [pjoin(datadir, f"echo{n}.nii.gz") for n in range(1, 4)]
tes = ["14.5", "38.5", "62.5"]


def test_unmask():
    # generate boolean mask + get number of True values
    mask = rs.choice([0, 1], size=(100,)).astype(bool)
    n_data = mask.sum()

    inputs = [
        (rs.rand(n_data, 3), float),  # 2D float
        (rs.rand(n_data, 3, 3), float),  # 3D float
        (rs.randint(10, size=(n_data, 3)), int),  # 2D int
        (rs.randint(10, size=(n_data, 3, 3)), int),  # 3D int
    ]

    for input, dtype in inputs:
        out = utils.unmask(input, mask)
        assert out.shape == (100,) + input.shape[1:]
        assert out.dtype == dtype


def test_dice():
    arr = rs.choice([0, 1], size=(100, 100))
    # identical arrays should have a Dice index of 1
    assert utils.dice(arr, arr) == 1.0
    # inverted arrays should have a Dice index of 0
    assert utils.dice(arr, np.logical_not(arr)) == 0.0
    # float arrays should return same as integer or boolean array
    assert utils.dice(arr * rs.rand(100, 100), arr * rs.rand(100, 100)) == 1.0
    # empty arrays still work
    assert utils.dice(np.array([]), np.array([])) == 0.0
    # different size arrays raise a ValueError
    with pytest.raises(ValueError):
        utils.dice(arr, rs.choice([0, 1], size=(20, 20)))


def test_andb():
    # test with a range of dimensions and ensure output dtype is int
    for ndim in range(1, 5):
        shape = (10,) * ndim
        out = utils.andb([rs.randint(10, size=shape) for f in range(5)])
        assert out.shape == shape
        assert out.dtype == int

    # confirm error raised when dimensions are not the same
    with pytest.raises(ValueError):
        utils.andb([rs.randint(10, size=(10, 10)), rs.randint(10, size=(20, 20))])


def test_make_adaptive_mask(caplog):
    """Test tedana.utils.make_adaptive_mask with different methods."""
    from nilearn.masking import apply_mask

    # load data make masks
    mask_file = pjoin(datadir, "mask.nii.gz")
    data = np.stack([apply_mask(f, mask_file).T for f in fnames], axis=1)
    n_voxels_in_mask = data.shape[0]

    # Add in simulated values
    base_val = np.mean(data[:, 0, :])  # mean value of first echo
    idx = 5457
    # Three good echoes (3)
    data[idx, :, :] = np.array([base_val, base_val - 1, base_val - 2])[:, None]
    # Dropout: good bad good (3)
    # Decay: good good bad (2)
    data[idx + 1, :, :] = np.array([base_val, 1, base_val])[:, None]
    # Dropout: good bad bad (1)
    # Decay: good good bad (2)
    data[idx + 2, :, :] = np.array([base_val, 1, 1])[:, None]
    # Dropout: good good good (3)
    # Decay: good bad bad (1)
    data[idx + 3, :, :] = np.array([base_val, base_val, base_val])[:, None]
    # Dropout: bad bad bad (0)
    # Decay: good good good (3)
    data[idx + 4, :, :] = np.array([1, 0.9, 0.8])[:, None]
    # Base: good good bad (2)
    # Dropout: bad bad bad (0)
    # Decay: good good good (3)
    data[idx + 5, :, :] = np.array([1, 0.9, -1])[:, None]

    # Simulating 5 echo data to test the n_independent_echos parameter
    data5 = np.concatenate(
        (
            data,
            0.95 * np.expand_dims(data[:, 2, :], axis=1),
            0.9 * np.expand_dims(data[:, 2, :], axis=1),
        ),
        axis=1,
    )

    # Just dropout method
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        threshold=1,
        methods=["dropout"],
    )

    assert mask.shape == adaptive_mask.shape == (n_voxels_in_mask,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 3
    assert adaptive_mask[idx + 2] == 1
    assert adaptive_mask[idx + 3] == 3
    assert adaptive_mask[idx + 4] == 0
    assert adaptive_mask[idx + 5] == 0
    assert mask.sum() == 49386
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([12568, 1815, 4402, 43169]))
    assert "voxels in user-defined mask do not have good signal" in caplog.text

    # Just decay method
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        threshold=1,
        methods=["decay"],
    )

    assert mask.shape == adaptive_mask.shape == (n_voxels_in_mask,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 2
    assert adaptive_mask[idx + 2] == 2
    assert adaptive_mask[idx + 3] == 1
    assert adaptive_mask[idx + 4] == 3
    assert adaptive_mask[idx + 5] == 2
    assert mask.sum() == 60985  # This method can't flag first echo as bad
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([969, 4366, 5974, 50645]))

    # Dropout and decay methods combined
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        threshold=1,
        methods=["dropout", "decay"],
    )

    assert mask.shape == adaptive_mask.shape == (n_voxels_in_mask,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 2
    assert adaptive_mask[idx + 2] == 1
    assert adaptive_mask[idx + 3] == 1
    assert adaptive_mask[idx + 4] == 0
    assert adaptive_mask[idx + 5] == 0
    assert mask.sum() == 49386
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([12568, 3113, 6233, 40040]))

    # Adding "none" should have no effect
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        threshold=1,
        methods=["dropout", "decay", "none"],
    )

    assert mask.shape == adaptive_mask.shape == (n_voxels_in_mask,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 2
    assert adaptive_mask[idx + 2] == 1
    assert adaptive_mask[idx + 3] == 1
    assert adaptive_mask[idx + 4] == 0
    assert adaptive_mask[idx + 5] == 0
    assert mask.sum() == 49386
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([12568, 3113, 6233, 40040]))

    # Just "none"
    mask, adaptive_mask = utils.make_adaptive_mask(
        data,
        threshold=1,
        methods=["none"],
    )

    assert mask.shape == adaptive_mask.shape == (n_voxels_in_mask,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 3
    assert adaptive_mask[idx + 1] == 3
    assert adaptive_mask[idx + 2] == 3
    assert adaptive_mask[idx + 3] == 3
    assert adaptive_mask[idx + 4] == 3
    assert adaptive_mask[idx + 5] == 2
    assert mask.sum() == 60985
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 3]))
    assert np.allclose(counts, np.array([969, 1412, 1196, 58377]))
    assert "No methods provided for adaptive mask generation." in caplog.text

    # testing n_independent_echos
    # This should match "decay" from above, except all voxels with 3 good echoes should now have 5
    # since two echoes were added that should not have caused more decay
    mask, adaptive_mask = utils.make_adaptive_mask(
        data5,
        threshold=1,
        methods=["decay"],
        n_independent_echos=3,
    )

    assert mask.shape == adaptive_mask.shape == (n_voxels_in_mask,)
    assert np.allclose(mask, (adaptive_mask >= 1).astype(bool))
    assert adaptive_mask[idx] == 5
    assert adaptive_mask[idx + 1] == 2
    assert adaptive_mask[idx + 2] == 2
    assert adaptive_mask[idx + 3] == 1
    assert adaptive_mask[idx + 4] == 5
    assert adaptive_mask[idx + 5] == 2
    assert mask.sum() == 60985  # This method can't flag first echo as bad
    vals, counts = np.unique(adaptive_mask, return_counts=True)
    assert np.allclose(vals, np.array([0, 1, 2, 5]))
    assert np.allclose(counts, np.array([969, 4366, 5974, 50645]))
    # 4366 + 5973 = 10399 (i.e. voxels with 1 or 2 good echoes are flagged here)
    assert (
        "10340 voxels (17.0%) have fewer than 3.0 good voxels. "
        "These voxels will be used in all analyses, "
        "but might not include 3 independent echo measurements."
    ) in caplog.text

    mask, adaptive_mask = utils.make_adaptive_mask(
        data5,
        threshold=1,
        methods=["decay"],
        n_independent_echos=4,
    )

    assert (
        "10340 voxels (17.0%) have fewer than 3.0 good voxels. "
        "The degrees of freedom for fits across echoes will remain 4 even if "
        "there might be fewer independent echo measurements."
    ) in caplog.text


# SMOKE TESTS


def test_smoke_make_adaptive_mask():
    """Ensure that make_adaptive_mask returns reasonable objects with random inputs.

    in the correct format.

    Note: make_adaptive_mask has optional paramters - mask and threshold.
    """
    n_samples = 100
    n_echos = 5
    n_times = 20
    data = np.random.random((n_samples, n_echos, n_times))

    assert utils.make_adaptive_mask(data, methods=["dropout", "decay"]) is not None


def test_smoke_unmask():
    """Ensure that unmask returns reasonable objects with random inputs.

    in the correct format.

    Note: unmask could take in 1D or 2D or 3D arrays.
    """
    mask = np.random.randint(2, size=100)
    n_samples_in_mask = mask.sum()
    data_1d = np.random.random(n_samples_in_mask)
    data_2d = np.random.random((n_samples_in_mask, 5))
    data_3d = np.random.random((n_samples_in_mask, 5, 20))

    assert utils.unmask(data_1d, mask) is not None
    assert utils.unmask(data_2d, mask) is not None
    assert utils.unmask(data_3d, mask) is not None


def test_smoke_dice():
    """Ensure that dice returns reasonable objects with random inputs.

    in the correct format.

    Note: two arrays must be in the same length.
    """
    arr1 = np.random.random(100)
    arr2 = np.random.random(100)

    assert utils.dice(arr1, arr2) is not None


def test_smoke_andb():
    """Ensure that andb returns reasonable objects with random inputs.

    in the correct format.
    """
    arr = np.random.random((100, 10)).tolist()  # 2D list of "arrays"

    assert utils.andb(arr) is not None


def test_smoke_get_spectrum():
    """Ensure that get_spectrum returns reasonable objects with random inputs.

    in the correct format.
    """
    data = np.random.random(100)
    tr = random.random()

    spectrum, freqs = utils.get_spectrum(data, tr)
    assert spectrum is not None
    assert freqs is not None


def test_smoke_threshold_map():
    """Ensure that threshold_map returns reasonable objects with random inputs.

    in the correct format.

    Note: using 3D array as img, some parameters are optional and are all tested.
    """
    img = np.random.random((10, 10, 10))  # 3D array must of of size S
    min_cluster_size = random.randint(1, 100)

    threshold = random.random()

    assert utils.threshold_map(img, min_cluster_size) is not None

    # test threshold_map with different optional parameters
    assert utils.threshold_map(img, min_cluster_size, threshold=threshold) is not None
    assert utils.threshold_map(img, min_cluster_size, binarize=False) is not None
    assert utils.threshold_map(img, min_cluster_size, sided="one") is not None
    assert utils.threshold_map(img, min_cluster_size, sided="bi") is not None


def test_create_legendre_polynomial_basis_set():
    """Checking that accurate Legendre polynomials are created."""

    n_vols = 100
    legendre_arr = utils.create_legendre_polynomial_basis_set(n_vols, dtrank=6)

    # Testing first 6 orders to 6 decimal accuracy using
    #  explicit equations rathern than scipy's lpmv
    legendre_rounded = np.round(legendre_arr, decimals=6)
    bounds = np.linspace(-1, 1, n_vols)
    # order 0 (all 1's)
    assert (legendre_arr[:, 0] == 1).sum() == n_vols
    # order 1 (y=x)
    assert np.abs((legendre_rounded[:, 1] - np.round(bounds, decimals=6))).sum() == 0
    # order 2 (y = 0.5*(3*x^2 - 1))
    tmp_o2 = 0.5 * (3 * bounds**2 - 1)
    assert np.abs((legendre_rounded[:, 2] - np.round(tmp_o2, decimals=6))).sum() == 0
    # order 3 (y = 0.5*(5*x^3 - 3*x))
    tmp_o3 = 0.5 * (5 * bounds**3 - 3 * bounds)
    assert np.abs((legendre_rounded[:, 3] - np.round(tmp_o3, decimals=6))).sum() == 0
    # order 4 (y = 0.125*(35*x^4 - 30*x^2 + 3))
    tmp_o4 = 0.125 * (35 * bounds**4 - 30 * bounds**2 + 3)
    assert np.abs((legendre_rounded[:, 4] - np.round(tmp_o4, decimals=6))).sum() == 0
    # order 5 (y = 0.125*(63*x^5 - 70*x^3 + 15x))
    tmp_o5 = 0.125 * (63 * bounds**5 - 70 * bounds**3 + 15 * bounds)
    assert np.abs((legendre_rounded[:, 5] - np.round(tmp_o5, decimals=6))).sum() == 0


def test_sec2millisec():
    """Ensure that sec2millisec returns 1000x the input values."""
    assert utils.sec2millisec(5) == 5000
    assert utils.sec2millisec(np.array([5])) == np.array([5000])


def test_millisec2sec():
    """Ensure that millisec2sec returns 1/1000x the input values."""
    assert utils.millisec2sec(5000) == 5
    assert utils.millisec2sec(np.array([5000])) == np.array([5])


def test_check_te_values(caplog):
    """Ensure that check_te_values returns the correct values."""
    # Values in seconds (preferred per BIDS) - should be converted to milliseconds
    assert utils.check_te_values([0.015, 0.039, 0.063]) == [15, 39, 63]
    assert utils.check_te_values([0.15, 0.35, 0.55]) == [150, 350, 550]

    # EPTI echo times (48 echoes)
    epti_te_ms = [
        6.70,
        7.63,
        8.56,
        9.49,
        10.42,
        11.35,
        12.28,
        13.21,
        14.14,
        15.07,
        16.00,
        16.93,
        17.86,
        18.79,
        19.72,
        20.65,
        21.58,
        22.51,
        23.44,
        24.37,
        25.30,
        26.23,
        27.16,
        28.09,
        29.02,
        29.95,
        30.88,
        31.81,
        32.74,
        33.67,
        34.60,
        35.53,
        36.46,
        37.39,
        38.32,
        39.25,
        40.18,
        41.11,
        42.04,
        42.97,
        43.90,
        44.83,
        45.76,
        46.69,
        47.62,
        48.55,
        49.48,
        50.41,
    ]
    epti_te_sec = [te / 1000 for te in epti_te_ms]
    assert utils.check_te_values(epti_te_sec) == epti_te_ms

    # Values in milliseconds (deprecated) - should be returned as-is with warning
    assert utils.check_te_values([15, 39, 63]) == [15, 39, 63]
    assert (
        "TE values appear to be in milliseconds. Per BIDS convention, echo times should "
        "be provided in seconds. Support for millisecond TE values is deprecated and will "
        "be removed in a future version. Please provide TE values in seconds."
    ) in caplog.text
    assert utils.check_te_values([2, 3, 4]) == [2, 3, 4]

    # EPTI echo times in milliseconds (deprecated)
    assert utils.check_te_values(epti_te_ms) == epti_te_ms

    # Check that the error is raised when TE values are in mixed units
    with pytest.raises(ValueError):
        utils.check_te_values([0.5, 2, 3])


def test_check_t2s_values(caplog):
    """Ensure that check_t2s_values returns the correct values."""
    # Values in seconds (expected per BIDS) - should be converted to milliseconds
    t2s_sec = np.array([0.015, 0.025, 0.035, 0.045])
    result = utils.check_t2s_values(t2s_sec)
    np.testing.assert_array_equal(result, [15, 25, 35, 45])

    # Values in milliseconds (common mistake) - should be returned as-is with warning
    t2s_ms = np.array([15, 25, 35, 45])
    result = utils.check_t2s_values(t2s_ms)
    np.testing.assert_array_equal(result, [15, 25, 35, 45])
    assert (
        "T2* map median value is 30.00, which suggests values are in "
        "milliseconds rather than seconds. Per BIDS convention, T2* maps should be "
        "in seconds. The map will be used as-is (in milliseconds), but please consider "
        "providing T2* maps in seconds in the future for consistency with BIDS."
    ) in caplog.text

    # Array with zeros (common in T2* maps for masked voxels)
    t2s_with_zeros = np.array([0, 0.020, 0.030, 0, 0.040])
    result = utils.check_t2s_values(t2s_with_zeros)
    np.testing.assert_array_equal(result, [0, 20, 30, 0, 40])

    # All zeros - should return as-is with warning
    t2s_all_zeros = np.array([0, 0, 0, 0])
    result = utils.check_t2s_values(t2s_all_zeros)
    np.testing.assert_array_equal(result, [0, 0, 0, 0])

    # Values that are too large - should raise ValueError
    t2s_invalid = np.array([1500, 2500, 3500])
    with pytest.raises(ValueError):
        utils.check_t2s_values(t2s_invalid)


def test_parse_volume_indices():
    """Ensure that parse_volume_indices returns the correct values."""
    assert utils.parse_volume_indices("0,1,2") == [0, 1, 2]
    assert utils.parse_volume_indices("0:3") == [0, 1, 2]
    assert utils.parse_volume_indices("0:3,5,6,10:12") == [0, 1, 2, 5, 6, 10, 11]

    # Check that errors are raised for invalid inputs
    # Step size is not supported
    with pytest.raises(ValueError, match="Invalid volume indices string"):
        utils.parse_volume_indices("0:10:2")

    with pytest.raises(ValueError, match="Invalid volume indices string"):
        utils.parse_volume_indices("::2")

    # Open-ended range is not supported
    with pytest.raises(ValueError, match="Invalid volume indices string"):
        utils.parse_volume_indices(":5")

    with pytest.raises(ValueError, match="Invalid volume indices string"):
        utils.parse_volume_indices("5:")

    # Negative indices are not supported (in a range)
    with pytest.raises(ValueError, match="Invalid volume indices string"):
        utils.parse_volume_indices("0:-5")

    # Negative indices are not supported (at all)
    with pytest.raises(ValueError, match="Invalid volume indices string"):
        utils.parse_volume_indices("-1")
