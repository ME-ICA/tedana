"""Functions for parsers."""

import argparse
import os.path as op


def check_tedpca_value(string, is_parser=True):
    """
    Check tedpca argument.

    Check if argument is a float in range (0,1), an int greater than 1 or one of a
    list of strings.
    """
    valid_options = ("mdl", "aic", "kic", "kundu", "kundu-stabilize")
    if string in valid_options:
        return string

    error = argparse.ArgumentTypeError if is_parser else ValueError
    try:
        floatarg = float(string)
    except ValueError:
        msg = f"Argument to tedpca must be a number or one of: {', '.join(valid_options)}"
        raise error(msg)

    if floatarg != int(floatarg):
        if not (0 < floatarg < 1):
            raise error("Float argument to tedpca must be between 0 and 1.")
        return floatarg
    else:
        intarg = int(floatarg)
        if floatarg < 1:
            raise error("Int argument must be greater than 1")
        return intarg


def is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if not op.isfile(arg) and arg is not None:
        parser.error(f"The file {arg} does not exist!")

    return arg
