"""
Functions for parsers.
"""
import os.path as op
import logging
from numbers import Number

import argparse


def check_tedpca_value(string, is_parser=True):
    """Check if argument is a float in range 0-1 or one of a list of strings."""
    valid_options = ("mdl", "aic", "kic", "kundu", "kundu-stabilize")
    msg = None
    if string not in valid_options:
        if not isinstance(string, Number):
            msg = "Argument must be a float or one of: {}".format(
                ", ".join(valid_options)
            )
        elif not (0 <= float(string) <= 1):
            msg = "Argument must be between 0 and 1."
        else:
            string = float(string)

    if msg:
        if is_parser:
            raise argparse.ArgumentTypeError(msg)
        else:
            raise ValueError(msg)

    return string


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error('The file {0} does not exist!'.format(arg))

    return arg


class ContextFilter(logging.Filter):
    """
    A filter to allow specific logging handlers to ignore specific loggers.
    We use this to prevent our report-generation and reference-compiling
    loggers from printing to the general log file or to stdout.
    """
    NAMES = ['REPORT', 'REFERENCES']

    def filter(self, record):
        if not any([n in record.name for n in self.NAMES]):
            return True
