"""
Tests for tedana.stats.getfbounds
"""

import numpy as np
import pytest
import random

from tedana.stats import getfbounds


def test_getfbounds():
    good_inputs = range(1, 12)

    for n_echos in good_inputs:
        getfbounds(n_echos)


# SMOKE TEST 

def test_smoke_getfbounds():
    """ 
    Ensures that getfbounds returns outputs when fed in a random number of echos
    """
    n_echos = random.randint(3, 10) # At least two echos!
    f05, f025, f01 = getfbounds(n_echos)
    
    assert f05 is not None
    assert f025 is not None
    assert f01 is not None
