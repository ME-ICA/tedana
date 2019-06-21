"""
Tests for tedana.stats.getfbounds
"""

import numpy as np
import pytest

from tedana.stats import getfbounds


def test_getfbounds():
    good_inputs = range(1, 12)

    for n_echos in good_inputs:
        getfbounds(n_echos)
