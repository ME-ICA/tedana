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
    This is a filter which injects contextual information into the log.

    Rather than use actual contextual information, we just use random
    data in this demo.
    """
    NAMES = ['REPORT', 'REFERENCES']

    def filter(self, record):
        if not any([n in record.name for n in self.NAMES]):
            return True
