"""
Functions for parsers.
"""
import os.path as op
import logging

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
