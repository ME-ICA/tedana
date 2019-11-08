"""
Functions for parsers.
"""
import os.path as op
import logging


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
