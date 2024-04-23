"""Test workflow parser utility functions."""

import argparse

import pytest

from tedana.workflows.parser_utils import check_tedpca_value


def test_check_tedpca_value():
    """Test the check_tedpca_value function."""
    with pytest.raises(argparse.ArgumentTypeError):
        check_tedpca_value("hello", is_parser=True)

    with pytest.raises(ValueError):
        check_tedpca_value("hello", is_parser=False)

    with pytest.raises(argparse.ArgumentTypeError):
        check_tedpca_value(1.5, is_parser=True)

    with pytest.raises(ValueError):
        check_tedpca_value(1.5, is_parser=False)

    with pytest.raises(ValueError):
        check_tedpca_value(-1, is_parser=False)

    assert check_tedpca_value(0.95) == 0.95
    assert check_tedpca_value("0.95") == 0.95
    assert check_tedpca_value("mdl") == "mdl"
    assert check_tedpca_value(52) == 52
