"""Tests for tedana.io."""

import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from tedana import io as me
from tedana.tests.test_utils import fnames, tes
from tedana.tests.utils import get_test_data_path

data_dir = get_test_data_path()


def test_new_nii_like():
    data, ref = me.load_data(fnames, n_echos=len(tes))
    nimg = me.new_nii_like(ref, data)

    assert isinstance(nimg, nib.Nifti1Image)
    assert nimg.shape == (39, 50, 33, 3, 5)


def test_load_data():
    fimg = [nib.load(f) for f in fnames]
    exp_shape = (64350, 3, 5)

    # list of filepath to images
    d, ref = me.load_data(fnames, n_echos=len(tes))
    assert d.shape == exp_shape
    assert isinstance(ref, nib.Nifti1Image)
    assert np.allclose(ref.get_fdata(), nib.load(fnames[0]).get_fdata())

    # list of filepath to images *without n_echos*
    d, ref = me.load_data(fnames)
    assert d.shape == exp_shape
    assert isinstance(ref, nib.Nifti1Image)
    assert np.allclose(ref.get_fdata(), nib.load(fnames[0]).get_fdata())

    # list of img_like
    d, ref = me.load_data(fimg, n_echos=len(tes))
    assert d.shape == exp_shape
    assert isinstance(ref, nib.Nifti1Image)
    assert ref == fimg[0]

    # list of img_like *without n_echos*
    d, ref = me.load_data(fimg)
    assert d.shape == exp_shape
    assert isinstance(ref, nib.Nifti1Image)
    assert ref == fimg[0]

    # bad entry
    fimg_with_bad_item = fimg[:]
    fimg_with_bad_item[-1] = 5
    with pytest.raises(TypeError):
        d, ref = me.load_data(fimg_with_bad_item)

    # unsupported tuple of img_like
    fimg_tuple = tuple(fimg)
    with pytest.raises(TypeError):
        d, ref = me.load_data(fimg_tuple, n_echos=len(tes))

    # tuple of img_like *without n_echos*
    with pytest.raises(TypeError):
        d, ref = me.load_data(fimg_tuple)

    # two echos should raise value error
    with pytest.raises(ValueError):
        me.load_data(fnames[:2])

    # imagine z-cat img
    d, ref = me.load_data(fnames[0], n_echos=3)
    assert d.shape == (21450, 3, 5)
    assert isinstance(ref, nib.Nifti1Image)
    assert ref.shape == (39, 50, 11, 1)

    # z-cat without n_echos should raise an error
    with pytest.raises(ValueError):
        me.load_data(fnames[0])

    # imagine z-cat img in list
    d, ref = me.load_data(fnames[:1], n_echos=3)
    assert d.shape == (21450, 3, 5)
    assert isinstance(ref, nib.Nifti1Image)
    assert ref.shape == (39, 50, 11, 1)

    # z-cat in list without n_echos should raise an error
    with pytest.raises(ValueError):
        me.load_data(fnames[:1])


# SMOKE TESTS


def test_smoke_split_ts():
    """
    Ensures that split_ts returns output when fed in with random inputs.

    Note: classification is ["accepted", "rejected", "ignored"].
    """
    np.random.seed(0)  # seeded because comptable MUST have accepted components
    n_samples = 100
    n_times = 20
    n_components = 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)

    # creating the component table with component as random floats,
    # a "metric," and random classification
    component = np.random.random(n_components)
    metric = np.random.random(n_components)
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    df_data = np.column_stack((component, metric, classification))
    comptable = pd.DataFrame(df_data, columns=["component", "metric", "classification"])

    hikts, resid = me.split_ts(data, mmix, mask, comptable)

    assert hikts is not None
    assert resid is not None


def test_smoke_write_split_ts():
    """Ensures that write_split_ts writes out the expected files with.

    random input and tear them down.
    """
    np.random.seed(0)  # at least one accepted and one rejected, thus all files are generated
    n_samples, n_times, n_components = 64350, 10, 6
    data = np.random.random((n_samples, n_times))
    mmix = np.random.random((n_times, n_components))
    mask = np.random.randint(2, size=n_samples)
    ref_img = os.path.join(data_dir, "mask.nii.gz")
    # ref_img has shape of (39, 50, 33) so data is 64350 (39*33*50) x 10
    # creating the component table with component as random floats,
    # a "metric," and random classification
    io_generator = me.OutputGenerator(ref_img)
    component = np.random.random(n_components)
    metric = np.random.random(n_components)
    classification = np.random.choice(["accepted", "rejected", "ignored"], n_components)
    df_data = np.column_stack((component, metric, classification))
    comptable = pd.DataFrame(df_data, columns=["component", "metric", "classification"])
    io_generator.verbose = True

    me.write_split_ts(data, mmix, mask, comptable, io_generator)

    # TODO: midk_ts.nii is never generated?
    fn = io_generator.get_name
    split = ("high kappa ts img", "low kappa ts img", "denoised ts img")
    fnames = [fn(f) for f in split]
    for filename in fnames:
        # remove all files generated
        os.remove(filename)

    io_generator.verbose = False

    me.write_split_ts(data, mmix, mask, comptable, io_generator)

    # TODO: midk_ts.nii is never generated?
    fn = io_generator.get_name
    split = "denoised ts img"
    fname = fn(split)
    # remove all files generated
    os.remove(fname)


def test_smoke_filewrite():
    """
    Ensures that filewrite fails for no known image type, write a known key.

    in both bids and orig formats.
    """
    n_samples, _, _ = 64350, 10, 6
    data_1d = np.random.random(n_samples)
    ref_img = os.path.join(data_dir, "mask.nii.gz")
    io_generator = me.OutputGenerator(ref_img)

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


def test_smoke_load_data():
    """Ensures that data is loaded when given a random neuroimage."""
    data = os.path.join(data_dir, "mask.nii.gz")
    n_echos = 1

    fdata, ref_img = me.load_data(data, n_echos)
    assert fdata is not None
    assert ref_img is not None


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
    """Test for testing empty files in fname_to_component_list function"""
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
