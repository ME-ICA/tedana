"""Functions for parsers."""

import argparse
import os.path as op


def check_tedpca_value(value, is_parser=True):
    """
    Check tedpca argument.

    Check if argument is a float in range (0,1], a positive integer,
    or one of a list of strings.
    """
    valid_options = ("mdl", "aic", "kic", "kundu", "kundu-stabilize")
    if value in valid_options:
        return value

    error = argparse.ArgumentTypeError if is_parser else ValueError
    dashes = "--" if is_parser else ""

    def check_float(floatvalue):
        assert isinstance(floatvalue, float)
        if not (0.0 < floatvalue <= 1.0):
            msg = f"Floating-point argument to {dashes}tedpca must be in the range (0.0, 1.0]."
            raise error(msg)
        return floatvalue

    def check_int(intvalue):
        assert isinstance(intvalue, int)
        if intvalue < 1:
            msg = f"Integer argument to {dashes}tedpca must be positive"
            raise error(msg)
        return intvalue

    if isinstance(value, float):
        return check_float(value)
    if isinstance(value, int):
        return check_int(value)

    try:
        floatvalue = float(value)
    except ValueError:
        msg = f"Argument to {dashes}tedpca must be a number or one of: {', '.join(valid_options)}"
        raise error(msg)

    try:
        intvalue = int(value)
    except ValueError:
        return check_float(floatvalue)
    return check_int(intvalue)


def is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if not op.isfile(arg) and arg is not None:
        parser.error(f"The file {arg} does not exist!")

    return arg
