"""
Tests for tedana.selection
"""

import numpy as np
import pandas as pd

from tedana import selection


def test_manual_selection():
    """
    Check that manual_selection runs correctly for different combinations of
    accepted and rejected components.
    """
    comptable = pd.DataFrame(index=np.arange(100))
    comptable = selection.manual_selection(comptable, acc=[1, 3, 5])
    assert comptable.loc[comptable.classification == 'accepted'].shape[0] == 3
    assert (comptable.loc[comptable.classification == 'rejected'].shape[0] ==
            (comptable.shape[0] - 3))

    comptable = selection.manual_selection(comptable, rej=[1, 3, 5])
    assert comptable.loc[comptable.classification == 'rejected'].shape[0] == 3
    assert (comptable.loc[comptable.classification == 'accepted'].shape[0] ==
            (comptable.shape[0] - 3))

    comptable = selection.manual_selection(comptable, acc=[0, 2, 4],
                                           rej=[1, 3, 5])
    assert comptable.loc[comptable.classification == 'accepted'].shape[0] == 3
    assert comptable.loc[comptable.classification == 'rejected'].shape[0] == 3
    assert (comptable.loc[comptable.classification == 'ignored'].shape[0] ==
            comptable.shape[0] - 6)
