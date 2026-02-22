"""Tests for tedana.io."""

import json
import os
from unittest import mock

import nibabel as nb
import numpy as np
import pandas as pd
import pytest
import requests

from tedana import io as me
from tedana.tests.utils import data_for_testing_info, get_test_data_path

data_dir = get_test_data_path()


# SMOKE TESTS


def test_smoke_split_ts():
    """
    Ensures that split_ts returns output when fed in with random inputs.

    Note: classification is ["accepted", "rejected", "ignored"].
    """
    np.random.seed(0)  # seeded because component_table MUST have accepted components
    n_samples = 100
    n_times = 20
    n_components = 6
    mixing = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)
    n_samples_in_mask = mask.sum()
    data = np.random.random((n_samples_in_mask, n_times))

    # creating the component table with component as random floats,
    # a "metric," and random classification
    component = np.random.random(n_components)
    metric = np.random.random(n_components)
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    df_data = np.column_stack((component, metric, classification))
    component_table = pd.DataFrame(df_data, columns=["component", "metric", "classification"])

    hikts, resid = me.split_ts(
        data=data,
        mixing=mixing,
        component_table=component_table,
    )

    assert hikts.shape == (n_samples_in_mask, n_times)
    assert resid.shape == (n_samples_in_mask, n_times)


def test_smoke_write_split_ts():
    """Ensures that write_split_ts writes out the expected files with.

    random input and tear them down.
    """
    np.random.seed(0)  # at least one accepted and one rejected, thus all files are generated

    ref_img = os.path.join(data_dir, "mask.nii.gz")
    ref_img = nb.load(ref_img)
    ref_img_4d = nb.Nifti1Image(ref_img.get_fdata()[..., None], ref_img.affine, ref_img.header)
    ref_img_4d.header.set_zooms(ref_img.header.get_zooms() + (1,))
    n_samples_in_mask = int(ref_img.get_fdata().sum())
    n_times, n_components = 10, 6

    mask = np.random.randint(2, size=n_samples_in_mask).astype(bool)
    data = np.random.random((n_samples_in_mask, n_times))
    mixing = np.random.random((n_times, n_components))

    # ref_img has shape of (39, 50, 33) so data is 64350 (39*33*50) x 10
    # creating the component table with component as random floats,
    # a "metric," and random classification
    io_generator = me.OutputGenerator(ref_img_4d)
    io_generator.register_mask(ref_img)

    component = np.random.random(n_components)
    metric = np.random.random(n_components)
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    df_data = np.column_stack((component, metric, classification))
    component_table = pd.DataFrame(df_data, columns=["component", "metric", "classification"])
    io_generator.verbose = True
    io_generator.register_mask(ref_img)

    me.write_split_ts(
        data=data,
        mixing=mixing,
        mask=mask,
        component_table=component_table,
        io_generator=io_generator,
    )

    # TODO: midk_ts.nii is never generated?
    fn = io_generator.get_name
    split = ("high kappa ts img", "low kappa ts img", "denoised ts img")
    fnames = [fn(f) for f in split]
    for filename in fnames:
        # remove all files generated
        os.remove(filename)

    io_generator.verbose = False

    me.write_split_ts(
        data=data,
        mixing=mixing,
        mask=mask,
        component_table=component_table,
        io_generator=io_generator,
    )

    # TODO: midk_ts.nii is never generated?
    fn = io_generator.get_name
    split = "denoised ts img"
    fname = fn(split)
    # remove all files generated
    os.remove(fname)


def test_load_data_nilearn_multi_echo_fastpath(tmp_path):
    """`load_data_nilearn` should return (Mb, E, T) for multi-echo files."""
    affine = np.eye(4)
    shape3d = (4, 3, 2)
    n_vols = 5
    n_echos = 2

    # Mask with a few voxels enabled
    mask_arr = np.zeros(shape3d, dtype=np.uint8)
    mask_arr[0, 0, 0] = 1
    mask_arr[1, 2, 1] = 1
    mask_arr[3, 1, 0] = 1
    mask_img = nb.Nifti1Image(mask_arr, affine)
    mask_bool = mask_arr.astype(bool)
    n_vox = int(mask_bool.sum())

    echo1 = np.random.RandomState(0).rand(*shape3d, n_vols).astype(np.float64)
    echo2 = np.random.RandomState(1).rand(*shape3d, n_vols).astype(np.float64)
    e1_path = tmp_path / "echo1.nii.gz"
    e2_path = tmp_path / "echo2.nii.gz"
    nb.Nifti1Image(echo1, affine).to_filename(e1_path)
    nb.Nifti1Image(echo2, affine).to_filename(e2_path)

    out = me.load_data_nilearn(
        [str(e1_path), str(e2_path)],
        mask_img=mask_img,
        n_echos=n_echos,
        dtype=np.float32,
    )
    assert out.shape == (n_vox, n_echos, n_vols)
    assert out.dtype == np.float32

    expected = np.stack(
        [echo1.astype(np.float32)[mask_bool], echo2.astype(np.float32)[mask_bool]],
        axis=1,
    )
    assert np.allclose(out, expected)


def test_load_data_nilearn_zcat_fastpath(tmp_path):
    """`load_data_nilearn` should support z-concatenated input (len(data)==1)."""
    affine = np.eye(4)
    x, y, n_z, n_vols, n_echos = 4, 3, 2, 5, 3
    z_cat = n_z * n_echos

    # Mask is defined in the per-echo z-space (x, y, n_z)
    mask_arr = np.zeros((x, y, n_z), dtype=np.uint8)
    mask_arr[0, 0, 0] = 1
    mask_arr[1, 2, 1] = 1
    mask_img = nb.Nifti1Image(mask_arr, affine)
    mask_bool = mask_arr.astype(bool)
    n_vox = int(mask_bool.sum())

    # Build a z-concat 4D image where each echo has a distinct constant value
    arr = np.zeros((x, y, z_cat, n_vols), dtype=np.float32)
    for i_echo in range(n_echos):
        arr[:, :, i_echo * n_z : (i_echo + 1) * n_z, :] = (i_echo + 1) * 10.0

    zcat_path = tmp_path / "zcat.nii.gz"
    nb.Nifti1Image(arr, affine).to_filename(zcat_path)

    out = me.load_data_nilearn(
        [str(zcat_path)],
        mask_img=mask_img,
        n_echos=n_echos,
        dtype=np.float32,
    )
    assert out.shape == (n_vox, n_echos, n_vols)

    expected = []
    for i_echo in range(n_echos):
        echo_arr = arr[:, :, i_echo * n_z : (i_echo + 1) * n_z, :]
        expected.append(echo_arr[mask_bool])
    expected = np.stack(expected, axis=1)
    assert np.allclose(out, expected)


def test_load_data_nilearn_zcat_requires_4d(tmp_path):
    """z-concatenated inputs must be 4D."""
    affine = np.eye(4)
    shape3d = (4, 3, 2)
    mask_img = nb.Nifti1Image(np.ones(shape3d, dtype=np.uint8), affine)

    bad_path = tmp_path / "zcat_bad.nii.gz"
    nb.Nifti1Image(np.zeros(shape3d, dtype=np.float32), affine).to_filename(bad_path)

    with pytest.raises(ValueError, match="Expected 4D z-concatenated image"):
        me.load_data_nilearn([str(bad_path)], mask_img=mask_img, n_echos=2, dtype=np.float32)


def test_load_data_nilearn_zcat_mask_shape_mismatch(tmp_path):
    """z-concatenated inputs should raise if mask doesn't match per-echo slice shape."""
    affine = np.eye(4)
    x, y, n_z, n_vols, n_echos = 4, 3, 2, 5, 3
    z_cat = n_z * n_echos

    # Wrong mask shape (z differs)
    wrong_mask = nb.Nifti1Image(np.ones((x, y, n_z + 1), dtype=np.uint8), affine)

    arr = np.zeros((x, y, z_cat, n_vols), dtype=np.float32)
    zcat_path = tmp_path / "zcat.nii.gz"
    nb.Nifti1Image(arr, affine).to_filename(zcat_path)

    with pytest.raises(ValueError, match="Z-cat echo slice/mask shape mismatch"):
        me.load_data_nilearn(
            [str(zcat_path)],
            mask_img=wrong_mask,
            n_echos=n_echos,
            dtype=np.float32,
        )


def test_load_data_nilearn_multi_echo_mask_shape_mismatch_executes_fastpath_check(
    tmp_path, monkeypatch
):
    """Multi-echo inputs with shape mismatch should hit the fast-path check and then fail."""
    affine = np.eye(4)
    shape3d = (4, 3, 2)
    n_vols, n_echos = 5, 2

    echo = np.random.RandomState(0).rand(*shape3d, n_vols).astype(np.float32)
    e1_path = tmp_path / "echo1.nii.gz"
    e2_path = tmp_path / "echo2.nii.gz"
    nb.Nifti1Image(echo, affine).to_filename(e1_path)
    nb.Nifti1Image(echo, affine).to_filename(e2_path)

    wrong_mask = nb.Nifti1Image(np.ones((4, 3, 3), dtype=np.uint8), affine)

    # Ensure we fail deterministically in the fallback too (avoid nilearn-specific messages).
    def _raise_apply_mask(*args, **kwargs):  # noqa:U100
        raise ValueError("Image/mask shape mismatch")

    monkeypatch.setattr(me.masking, "apply_mask", _raise_apply_mask)

    with pytest.raises(ValueError, match="Image/mask shape mismatch"):
        me.load_data_nilearn([str(e1_path), str(e2_path)], mask_img=wrong_mask, n_echos=n_echos)


def test_load_data_nilearn_multi_echo_requires_4d(tmp_path):
    """Multi-echo inputs must be 4D; this should error even via fallback."""
    affine = np.eye(4)
    shape3d = (4, 3, 2)
    n_echos = 2

    mask_img = nb.Nifti1Image(np.ones(shape3d, dtype=np.uint8), affine)

    # 3D (invalid) images
    arr3d = np.random.RandomState(0).rand(*shape3d).astype(np.float32)
    e1_path = tmp_path / "echo1_3d.nii.gz"
    e2_path = tmp_path / "echo2_3d.nii.gz"
    nb.Nifti1Image(arr3d, affine).to_filename(e1_path)
    nb.Nifti1Image(arr3d, affine).to_filename(e2_path)

    with pytest.raises(ValueError, match="Expected 4D image"):
        me.load_data_nilearn([str(e1_path), str(e2_path)], mask_img=mask_img, n_echos=n_echos)


def test_load_data_nilearn_multi_echo_fallback_path(tmp_path, monkeypatch):
    """Force the nilearn fallback path and verify output matches expected masking."""
    affine = np.eye(4)
    shape3d = (4, 3, 2)
    n_vols, n_echos = 5, 2

    mask_arr = np.zeros(shape3d, dtype=np.uint8)
    mask_arr[0, 0, 0] = 1
    mask_arr[1, 2, 1] = 1
    mask_img = nb.Nifti1Image(mask_arr, affine)
    mask_bool = mask_arr.astype(bool)

    echo1 = np.random.RandomState(0).rand(*shape3d, n_vols).astype(np.float32)
    echo2 = np.random.RandomState(1).rand(*shape3d, n_vols).astype(np.float32)

    # Save as NIfTI2 to also exercise `_convert_to_nifti1(..., dtype=...)` in fallback.
    e1_path = tmp_path / "echo1.nii.gz"
    e2_path = tmp_path / "echo2.nii.gz"
    nb.Nifti2Image(echo1, affine).to_filename(e1_path)
    nb.Nifti2Image(echo2, affine).to_filename(e2_path)

    real_load = me.nb.load
    calls = {"n": 0}

    def _flaky_load(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic load failure to force fallback")
        return real_load(path)

    monkeypatch.setattr(me.nb, "load", _flaky_load)

    out = me.load_data_nilearn(
        [str(e1_path), str(e2_path)],
        mask_img=mask_img,
        n_echos=n_echos,
        dtype=np.float32,
    )
    expected = np.stack([echo1[mask_bool], echo2[mask_bool]], axis=1).astype(np.float32)
    assert out.shape == expected.shape
    assert out.dtype == np.float32
    assert np.allclose(out, expected)


def test_smoke_filewrite():
    """
    Ensures that filewrite fails for no known image type, write a known key.

    in both bids and orig formats.
    """
    ref_img = os.path.join(data_dir, "mask.nii.gz")
    ref_img = nb.load(ref_img)
    n_samples_in_mask = int(ref_img.get_fdata().sum())
    io_generator = me.OutputGenerator(ref_img)
    io_generator.register_mask(ref_img)

    data_1d = np.random.random(n_samples_in_mask)
    with pytest.raises(KeyError):
        io_generator.save_file(data_1d, "")

    for convention in ("bidsv1.5.0", "orig"):
        io_generator.convention = convention
        fname = io_generator.save_file(data_1d, "t2star img")
        assert fname is not None
        try:
            os.remove(fname)
        except OSError:
            print("File not generated!")


# TODO: "BREAK" AND UNIT TESTS


def test_prep_data_for_json():
    """Tests for prep_data_for_json."""
    # Should reject non-dict entities since that is required for saver
    with pytest.raises(TypeError):
        me.prep_data_for_json(1)

    # Should not modify something with no special types
    d = {"mustang": "vroom"}
    new_d = me.prep_data_for_json(d)
    assert new_d == d

    # Should coerce an ndarray into a list
    d = {"number": np.ndarray(1)}
    new_d = me.prep_data_for_json(d)
    assert isinstance(new_d["number"], list)

    # Should work for nested dict
    d = {
        "dictionary": {
            "serializable": "cat",
            "array": np.ndarray([1, 2, 3]),
        }
    }
    new_d = me.prep_data_for_json(d)
    assert isinstance(new_d["dictionary"]["array"], list)


def test_str_to_component_list():
    """Tests for converting a string to a component list."""
    int_list_1 = [1]
    int_list_2 = [1, 4, 5]
    test_list_1 = [str(x) for x in int_list_1]
    test_list_2 = [str(x) for x in int_list_2]
    delims_to_test = (
        "\t",
        "\n",
        " ",
        ",",
    )
    for d in delims_to_test:
        test_data = d.join(test_list_1)
        assert me.str_to_component_list(test_data) == int_list_1
        test_data = d.join(test_list_2)
        assert me.str_to_component_list(test_data) == int_list_2

    # Test that one-line, one-element works
    assert me.str_to_component_list("1\n") == [1]
    # Test that one-line, multi-element works
    assert me.str_to_component_list("1,1\n") == [1, 1]
    # Test that extra delimeter is ignored
    assert me.str_to_component_list("1,1,") == [1, 1]

    with pytest.raises(ValueError, match=r"While parsing component"):
        me.str_to_component_list("1,2\t")


def test_fname_to_component_list():
    test_data = [1, 2, 3]
    temp_csv_fname = os.path.join(data_dir, "test.csv")
    df = pd.DataFrame(data=test_data)
    df.to_csv(path_or_buf=temp_csv_fname)
    result = me.fname_to_component_list(temp_csv_fname)
    os.remove(temp_csv_fname)
    assert result == test_data

    temp_txt_fname = os.path.join(data_dir, "test.txt")
    with open(temp_txt_fname, "w") as fp:
        fp.write("1,1,")
    result = me.fname_to_component_list(temp_txt_fname)
    os.remove(temp_txt_fname)
    assert result == [1, 1]


def test_fname_to_component_list_empty_file():
    """Test for testing empty files in fname_to_component_list function."""
    temp_csv_fname = os.path.join(data_dir, "test.csv")
    with open(temp_csv_fname, "w"):
        pass
    result = me.fname_to_component_list(temp_csv_fname)
    os.remove(temp_csv_fname)

    temp_txt_fname = os.path.join(data_dir, "test.txt")
    with open(temp_txt_fname, "w"):
        pass
    result = me.fname_to_component_list(temp_txt_fname)
    os.remove(temp_txt_fname)

    assert result == []


def test_custom_encoder():
    """Test the encoder we use for JSON incompatibilities."""
    # np int64
    test_data = {"data": np.int64(4)}
    encoded = json.dumps(test_data, cls=me.CustomEncoder)
    decoded = json.loads(encoded)
    assert test_data == decoded

    # np array
    test_data = {"data": np.asarray([1, 2, 3])}
    encoded = json.dumps(test_data, cls=me.CustomEncoder)
    decoded = json.loads(encoded)
    assert np.array_equal(test_data["data"], decoded["data"])

    # set should become list
    test_data = {"data": {"cat", "dog", "fish"}}
    encoded = json.dumps(test_data, cls=me.CustomEncoder)
    decoded = json.loads(encoded)
    assert list(test_data["data"]) == decoded["data"]

    # no special cases should use standard encoder
    test_data = {"pet": "dog"}
    encoded = json.dumps(test_data, cls=me.CustomEncoder)
    decoded = json.loads(encoded)
    assert test_data == decoded


@mock.patch("tedana.io.requests.get")
@mock.patch("tedana.io.op.isfile")
def test_download_json_file_not_found(mock_isfile, mock_requests_get):
    """Test case when file doesn't exist locally or on figshare."""
    mock_isfile.return_value = False

    mock_response = mock.Mock()
    mock_response.raise_for_status = mock.Mock()
    mock_response.json.return_value = {"files": [{"name": "tree.json", "download_url": "url.com"}]}
    mock_requests_get.return_value = mock_response

    result = me.download_json("non_existent_tree", "some_dir")

    assert result is None
    mock_response.raise_for_status.assert_called_once()


@mock.patch("tedana.io.requests.get")
@mock.patch("tedana.io.op.isfile")
def test_download_json_skips_if_exists(mock_isfile, mock_requests_get):
    """Test case when file already exists locally."""
    mock_isfile.return_value = True

    result = me.download_json("my_tree", "some_dir")

    assert result == "some_dir/my_tree.json"
    mock_requests_get.assert_not_called()


@mock.patch("tedana.io.requests.get")
def test_download_json_doesnt_connect_to_url(mock_requests_get, caplog: pytest.LogCaptureFixture):
    """Tests that the correct log message appears when URL not connected."""
    mock_requests_get.side_effect = requests.exceptions.ConnectionError(
        "Simulated connection error"
    )

    result = me.download_json("tedana_orig", "./")
    assert result is None
    assert "Cannot connect to figshare" in caplog.text


@mock.patch("tedana.io.requests.get")
@mock.patch("tedana.io.op.isfile")
def test_download_json_file_is_downloaded(mock_isfile, mock_requests_get):
    """Test json is downloaded if it exists on figshare."""
    mock_isfile.return_value = False

    metadata_response = mock.Mock()
    metadata_response.raise_for_status = mock.Mock()
    metadata_response.json.return_value = {
        "files": [{"name": "tree.json", "download_url": "url.com"}]
    }

    download_response = mock.Mock()
    download_response.raise_for_status = mock.Mock()
    file_content = {"sample": "data"}
    download_response.content = json.dumps(file_content).encode("utf-8")

    mock_requests_get.side_effect = [metadata_response, download_response]

    out_dir = data_for_testing_info("path")
    result = me.download_json("tree.json", out_dir)

    assert result == f"{out_dir}/tree.json"
    assert os.path.exists(result)

    with open(result, "r") as f:
        content = json.load(f)
    assert content == file_content
    os.remove(result)


def test_add_dict_to_file():
    """Test add_dict_to_file method for merging dictionaries into JSON files."""
    ref_img = os.path.join(data_dir, "mask.nii.gz")
    io_generator = me.OutputGenerator(ref_img, overwrite=True)

    # Test 1: Create a new file when none exists
    initial_data = {"key1": "value1", "key2": 42}
    fname = io_generator.add_dict_to_file(initial_data, "ICA metrics json")
    assert os.path.exists(fname)

    with open(fname, "r") as f:
        saved_data = json.load(f)
    assert saved_data == initial_data

    # Test 2: Merge new data into existing file
    additional_data = {"key3": "value3", "key4": [1, 2, 3]}
    fname = io_generator.add_dict_to_file(additional_data, "ICA metrics json")

    with open(fname, "r") as f:
        merged_data = json.load(f)
    assert "key1" in merged_data
    assert "key3" in merged_data
    assert merged_data["key1"] == "value1"
    assert merged_data["key3"] == "value3"
    assert merged_data["key4"] == [1, 2, 3]

    # Test 3: Update existing keys
    update_data = {"key1": "updated_value"}
    fname = io_generator.add_dict_to_file(update_data, "ICA metrics json")

    with open(fname, "r") as f:
        updated_data = json.load(f)
    assert updated_data["key1"] == "updated_value"
    assert updated_data["key3"] == "value3"  # Other keys preserved

    # Cleanup
    os.remove(fname)
