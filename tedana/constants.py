"""
===========================================
constants module (:mod: `tedana.constants`)
===========================================

.. currentmodule:: tedana.io

The constants module defines constants for use in the `tedana` package.
There are only variable definitions here, and no functions.

Input and Output
----------------
allowed_conventions
    Defines the keys present in each of the "table" variables for i/o.
    Each element represents a naming convention.
bids
    A constant defining the string value of the current BIDS version
img_table
    A table of images that may be written. Images that are split by echo
    end in the word "split" and are formats rather than complete strings.
json_table
    A table of JSON files that may be written.
tsv_table
    A table of TSV files that may be written.


Notes
-----
For input and output constants ending in "table," the first key is the
type of file to be written (for example, 't2star map'). The second key
indicates the naming convention to be used (for example, 'orig'). If an
invalid type of file or convention is used, an ambiguous KeyError will
occur.
"""

from pathlib import Path
import os.path as op

allowed_conventions = ('orig', 'bidsv1.5.0')

bids = 'bidsv1.5.0'

config_path = op.join(str(Path(__file__).parent.absolute()), 'config')

img_table_file = op.join(config_path, 'img_table.json')
json_table_file = op.join(config_path, 'json_table.json')
tsv_table_file = op.join(config_path, 'tsv_table.json')
