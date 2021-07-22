"""
Functions for parsers.
"""
import os.path as op

import argparse


def check_tedpca_value(string, is_parser=True):
    """Check if argument is a float in range 0-1 or one of a list of strings."""
    valid_options = ("mdl", "aic", "kic", "kundu", "kundu-stabilize")
    if string in valid_options:
        return string

    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        floatarg = float(string)
    except ValueError:
        msg = "Argument to tedpca must be a float or one of: {}".format(
            ", ".join(valid_options)
        )
        raise error(msg)

    if not (0 <= floatarg <= 1):
        raise error("Float argument to tedpca must be between 0 and 1.")
    return floatarg


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error('The file {0} does not exist!'.format(arg))

    return arg
