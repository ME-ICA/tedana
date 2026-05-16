"""Tests for tedana.utils."""

import random
from os.path import dirname
from os.path import join as pjoin

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


def test_threshold_map_4d_thresholds_per_volume():
    """Ensure threshold_map supports 4D input + per-volume thresholds."""
    img = np.zeros((5, 5, 5, 2), dtype=float)
    img[2, 2, 2, 0] = 0.9
    img[1, 1, 1, 1] = 0.6

    out = utils.threshold_map(img, min_cluster_size=1, threshold=[0.8, 0.7], binarize=True)
    assert out.shape == img.shape
    assert out[2, 2, 2, 0] is True or out[2, 2, 2, 0] is np.True_
    assert out[1, 1, 1, 1] is False or out[1, 1, 1, 1] is np.False_

    # Scalar threshold should be broadcast to all volumes
    out2 = utils.threshold_map(img, min_cluster_size=1, threshold=0.8, binarize=True)
    assert out2.shape == img.shape
    assert out2[2, 2, 2, 0] is True or out2[2, 2, 2, 0] is np.True_
    assert out2[1, 1, 1, 1] is False or out2[1, 1, 1, 1] is np.False_

    # Wrong-length thresholds should error
    with pytest.raises(ValueError):
        utils.threshold_map(img, min_cluster_size=1, threshold=[0.5], binarize=True)


def test_load_mask_user_mask_converted_to_nifti1(tmp_path):
    """`load_mask` should load a user mask and return a NIfTI1 image."""
    import nibabel as nb

    shape = (5, 5, 5)
    affine = np.eye(4)
    ref_img = nb.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)

    mask_arr = np.zeros(shape, dtype=np.uint8)
    mask_arr[2, 2, 2] = 1
    # Use NIfTI2 to exercise conversion to NIfTI1
    mask_img = nb.Nifti2Image(mask_arr, affine)
    mask_path = tmp_path / "mask.nii.gz"
    mask_img.to_filename(mask_path)

    out_mask_img, out_r2s = utils.load_mask(ref_img, mask=str(mask_path), t2smap=None)
    assert isinstance(out_mask_img, nb.Nifti1Image)
    assert out_r2s is None
    assert np.array_equal(np.asanyarray(out_mask_img.dataobj), mask_arr)


def test_load_mask_t2smap_only_builds_mask_and_returns_r2s(tmp_path):
    """`load_mask` with t2smap should build a mask and return R2* (= 1/T2*)."""
    import nibabel as nb

    shape = (3, 3, 3)
    affine = np.eye(4)
    ref_img = nb.Nifti1Image(np.ones(shape, dtype=np.float32), affine)

    t2s_arr = np.zeros(shape, dtype=np.float32)
    t2s_arr[1, 1, 1] = 0.02  # T2* in seconds -> R2* = 50 s^-1
    t2s_img = nb.Nifti1Image(t2s_arr, affine)
    t2s_path = tmp_path / "t2smap.nii.gz"
    t2s_img.to_filename(t2s_path)

    out_mask_img, out_r2s = utils.load_mask(ref_img, mask=None, t2smap=str(t2s_path))

    assert out_mask_img is not None
    assert out_r2s is not None
    assert np.asarray(out_r2s).shape == (1,)
    assert np.allclose(out_r2s[0], 50.0)  # 1/0.02 = 50 s^-1


def test_load_mask_combines_mask_and_t2smap(tmp_path):
    """`load_mask` with mask + t2smap should combine both and return R2*."""
    import nibabel as nb

    shape = (3, 3, 3)
    affine = np.eye(4)
    ref_img = nb.Nifti1Image(np.ones(shape, dtype=np.float32), affine)

    mask_arr = np.zeros(shape, dtype=np.uint8)
    mask_arr[1, 1, 1] = 1
    mask_img = nb.Nifti1Image(mask_arr, affine)
    mask_path = tmp_path / "mask.nii.gz"
    mask_img.to_filename(mask_path)

    t2s_arr = np.zeros(shape, dtype=np.float32)
    t2s_arr[1, 1, 1] = 0.03  # T2* = 30ms -> R2* ~= 33.33 s^-1
    t2s_img = nb.Nifti1Image(t2s_arr, affine)
    t2s_path = tmp_path / "t2smap.nii.gz"
    t2s_img.to_filename(t2s_path)

    out_mask_img, out_r2s = utils.load_mask(ref_img, mask=str(mask_path), t2smap=str(t2s_path))

    assert out_mask_img is not None
    assert np.asarray(out_r2s).shape == (1,)
    assert np.allclose(out_r2s[0], 1 / 0.03)


def test_load_mask_r2smap_only_builds_mask_and_returns_r2s(tmp_path):
    """`load_mask` with r2smap should build a mask and return the R2* values directly."""
    import nibabel as nb

    shape = (3, 3, 3)
    affine = np.eye(4)
    ref_img = nb.Nifti1Image(np.ones(shape, dtype=np.float32), affine)

    r2s_arr = np.zeros(shape, dtype=np.float32)
    r2s_arr[1, 1, 1] = 33.3  # R2* in s^-1
    r2s_img = nb.Nifti1Image(r2s_arr, affine)
    r2s_path = tmp_path / "r2smap.nii.gz"
    r2s_img.to_filename(r2s_path)

    out_mask_img, out_r2s = utils.load_mask(ref_img, mask=None, r2smap=str(r2s_path))

    assert out_mask_img is not None
    assert out_r2s is not None
    assert np.asarray(out_r2s).shape == (1,)
    assert np.allclose(out_r2s[0], 33.3, rtol=1e-4)


def test_load_mask_falls_back_to_compute_epi_mask(monkeypatch):
    """If neither mask nor T2* map are provided, `compute_epi_mask(ref_img)` is used."""
    import nibabel as nb

    shape = (5, 5, 5)
    affine = np.eye(4)
    ref_img = nb.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)

    sentinel_mask = nb.Nifti1Image(np.ones(shape, dtype=np.uint8), affine)
    called = {"n": 0, "arg": None}

    def _fake_compute_epi_mask(arg):
        called["n"] += 1
        called["arg"] = arg
        return sentinel_mask

    # `compute_epi_mask` is imported inside `utils.load_mask`, so patch the module attr.
    import nilearn.masking as nm

    monkeypatch.setattr(nm, "compute_epi_mask", _fake_compute_epi_mask)

    out_mask_img, out_r2s = utils.load_mask(ref_img, mask=None, t2smap=None)
    assert called["n"] == 1
    assert called["arg"] is ref_img
    assert out_mask_img is sentinel_mask
    assert out_r2s is None


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


def test_check_te_values(caplog):
    """Ensure that check_te_values returns values in seconds."""
    # Values in seconds (preferred per BIDS) - should be returned as-is
    assert utils.check_te_values([0.015, 0.039, 0.063]) == [0.015, 0.039, 0.063]
    assert utils.check_te_values([0.15, 0.35, 0.55]) == [0.15, 0.35, 0.55]

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
    # Seconds input returns seconds unchanged
    assert utils.check_te_values(epti_te_sec) == epti_te_sec

    # Values in milliseconds (deprecated) - should be converted to seconds with warning
    np.testing.assert_allclose(utils.check_te_values([15, 39, 63]), [0.015, 0.039, 0.063])
    assert (
        "TE values appear to be in milliseconds. Per BIDS convention, echo times should "
        "be provided in seconds. Support for millisecond TE values is deprecated and will "
        "be removed in a future version. Please provide TE values in seconds."
    ) in caplog.text
    np.testing.assert_allclose(utils.check_te_values([2, 3, 4]), [0.002, 0.003, 0.004])

    # EPTI echo times in milliseconds (deprecated) - should be converted to seconds
    result = utils.check_te_values(epti_te_ms)
    np.testing.assert_allclose(result, epti_te_sec)

    # Check that the error is raised when TE values are in mixed units
    with pytest.raises(ValueError):
        utils.check_te_values([0.5, 2, 3])


def test_check_t2s_values(caplog):
    """Ensure that check_t2s_values returns values in seconds."""
    # Values in seconds (expected per BIDS) - should be returned as-is
    t2s_sec = np.array([0.015, 0.025, 0.035, 0.045])
    result = utils.check_t2s_values(t2s_sec)
    np.testing.assert_array_equal(result, [0.015, 0.025, 0.035, 0.045])

    # Values in milliseconds (common mistake) - should be converted to seconds with warning
    t2s_ms = np.array([15.0, 25.0, 35.0, 45.0])
    result = utils.check_t2s_values(t2s_ms)
    np.testing.assert_allclose(result, [0.015, 0.025, 0.035, 0.045])
    assert "milliseconds rather than seconds" in caplog.text

    # Array with zeros (common in T2* maps for masked voxels) - values in seconds
    t2s_with_zeros = np.array([0, 0.020, 0.030, 0, 0.040])
    result = utils.check_t2s_values(t2s_with_zeros)
    np.testing.assert_array_equal(result, [0, 0.020, 0.030, 0, 0.040])

    # All zeros - should return as-is with warning
    t2s_all_zeros = np.array([0, 0, 0, 0])
    result = utils.check_t2s_values(t2s_all_zeros)
    np.testing.assert_array_equal(result, [0, 0, 0, 0])

    # Values that are too large - should raise ValueError
    t2s_invalid = np.array([1500, 2500, 3500])
    with pytest.raises(ValueError):
        utils.check_t2s_values(t2s_invalid)


def test_check_r2s_values():
    """check_r2s_values should return values in s⁻¹ and raise/warn on bad input."""
    # Valid R2* values (typical 10-100 s⁻¹)
    r2s_valid = np.array([14.3, 25.0, 33.3, 50.0, 66.7])
    result = utils.check_r2s_values(r2s_valid)
    np.testing.assert_array_equal(result, r2s_valid)

    # R2* with zeros (masked voxels) — should be returned unchanged
    r2s_with_zeros = np.array([0.0, 25.0, 33.3, 0.0, 50.0])
    result = utils.check_r2s_values(r2s_with_zeros)
    np.testing.assert_array_equal(result, r2s_with_zeros)

    # All zeros — should return as-is with warning
    r2s_all_zeros = np.array([0.0, 0.0, 0.0])
    result = utils.check_r2s_values(r2s_all_zeros)
    np.testing.assert_array_equal(result, r2s_all_zeros)

    # Values that look like T2* in seconds (too small for R2*) — should raise
    r2s_like_t2s_seconds = np.array([0.015, 0.025, 0.035])
    with pytest.raises(ValueError, match="too small"):
        utils.check_r2s_values(r2s_like_t2s_seconds)

    # Values that are absurdly large — should raise
    r2s_too_large = np.array([2000.0, 3000.0, 5000.0])
    with pytest.raises(ValueError, match="too large"):
        utils.check_r2s_values(r2s_too_large)


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
