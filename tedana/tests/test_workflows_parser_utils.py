"""Test workflow parser utility functions."""
import argparse
import pytest

from tedana.workflows.parser_utils import string_or_float


def test_string_or_float():
    """Test the string_or_float function."""
    with pytest.raises(argparse.ArgumentTypeError):
        string_or_float("hello")

    with pytest.raises(argparse.ArgumentTypeError):
        string_or_float(1.5)

    assert string_or_float(0.95) == 0.95
    assert string_or_float("mdl") == "mdl"
