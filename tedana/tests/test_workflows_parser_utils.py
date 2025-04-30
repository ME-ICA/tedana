"""Test workflow parser utility functions."""

import argparse
import os.path as op

import pandas as pd
import pytest

from tedana.tests.utils import data_for_testing_info
from tedana.workflows.parser_utils import (
    check_tedpca_value,
    parse_manual_list_int,
    parse_manual_list_str,
)


def test_check_tedpca_value():
    """Test the check_tedpca_value function."""
    with pytest.raises(argparse.ArgumentTypeError):
        check_tedpca_value("hello", is_parser=True)

    with pytest.raises(ValueError):
        check_tedpca_value("hello", is_parser=False)

    with pytest.raises(argparse.ArgumentTypeError):
        check_tedpca_value("1.0", is_parser=True)

    with pytest.raises(argparse.ArgumentTypeError):
        check_tedpca_value("1.", is_parser=True)

    with pytest.raises(ValueError):
        check_tedpca_value(1.0, is_parser=False)

    with pytest.raises(argparse.ArgumentTypeError):
        check_tedpca_value("0.0", is_parser=True)

    with pytest.raises(ValueError):
        check_tedpca_value(0.0, is_parser=False)

    with pytest.raises(ValueError):
        check_tedpca_value(-1, is_parser=False)

    with pytest.raises(ValueError):
        check_tedpca_value(0, is_parser=False)

    assert check_tedpca_value(0.95, is_parser=False) == 0.95
    assert check_tedpca_value("0.95") == 0.95
    assert check_tedpca_value("mdl") == "mdl"
    assert check_tedpca_value(1, is_parser=False) == 1.0
    assert check_tedpca_value("1") == 1.0
    assert check_tedpca_value(52, is_parser=False) == 52


def test_parse_manual_list_int():
    """Test the parse_manual_list_int function for all accepted and error generating inputs."""

    tmp = parse_manual_list_int("5")
    assert tmp == [5]

    tmp = parse_manual_list_int(["5", "  6  "])
    assert tmp == [5, 6]

    tmp = parse_manual_list_int("5,  6  ")
    assert tmp == [5, 6]

    tmp = parse_manual_list_int([5, 6])
    assert tmp == [5, 6]

    tmp = parse_manual_list_int([])
    assert tmp == []

    tmp = parse_manual_list_int(None)
    assert tmp == []

    to_accept = [i for i in range(3)]
    test_data_path = data_for_testing_info("path")
    acc_df = pd.DataFrame(data=to_accept, columns=["Components"])
    acc_csv_fname = op.join(test_data_path, "accept.csv")
    acc_df.to_csv(acc_csv_fname)
    tmp = parse_manual_list_int(acc_csv_fname)
    assert tmp == to_accept

    with pytest.raises(
        ValueError,
        match=r"parse_manual_list_int expected a list of integers",
    ):
        tmp = parse_manual_list_int("5, 6.5")

    with pytest.raises(
        ValueError,
        match=r"parse_manual_list_int expected a list of integers",
    ):
        tmp = parse_manual_list_int([5, 6.5])


def test_parse_manual_list_str():
    """Test the parse_manual_list_str function for all accepted and error generating inputs."""

    tmp = parse_manual_list_str("tag 1 ")
    assert tmp == "tag 1"

    tmp = parse_manual_list_str(["tag 1 "])
    assert tmp == "tag 1"

    tmp = parse_manual_list_str("tag 1 ,  tag 2 ")
    assert tmp == "tag 1,tag 2"

    tmp = parse_manual_list_str(["tag 1 ,  tag 2 "])
    assert tmp == "tag 1,tag 2"

    tmp = parse_manual_list_str(["tag 1 ", " tag 2 "])
    assert tmp == "tag 1,tag 2"

    with pytest.raises(
        ValueError,
        match=r"parse_manual_list_str includes a comma in a list of multiple strings. ",
    ):
        tmp = parse_manual_list_str(["tag 1 ", " tag 2 , tag 3 "])

    with pytest.raises(
        ValueError,
        match=r"parse_manual_list_str expected a string or a list of strings, ",
    ):
        tmp = parse_manual_list_str(1)

    with pytest.raises(
        ValueError,
        match=r"parse_manual_list_str expected a string or a list of strings, ",
    ):
        tmp = parse_manual_list_str([1])

    with pytest.raises(
        ValueError,
        match=r"parse_manual_list_str expected a string or a list of strings, ",
    ):
        tmp = parse_manual_list_str(["tag 1", 1])
