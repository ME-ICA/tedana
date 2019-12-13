"""
Tests for tedana.model.fit
"""

import numpy as np
import pandas as pd

from tedana.metrics import kundu_fit


def test_smoke_kundu_metrics():
    """
    Smoke test for kundu metrics function. Just make sure that kundu_metrics
    runs without breaking when fed random data in the right formats.
    """
    n_comps = 100
    n_voxels = 10000
    comptable = pd.DataFrame(columns=['kappa', 'rho', 'variance explained',
                                      'normalized variance explained'],
                             data=np.random.random((100, 4)),
                             index=np.arange(100))
    metric_maps = {}
    metric_maps['Z_maps'] = np.random.random((n_voxels, n_comps))
    metric_maps['Z_clmaps'] = np.random.randint(low=0, high=2,
                                                size=(n_voxels, n_comps))
    metric_maps['F_R2_maps'] = np.random.random((n_voxels, n_comps))
    metric_maps['F_S0_clmaps'] = np.random.randint(low=0, high=2,
                                                   size=(n_voxels, n_comps))
    metric_maps['F_R2_clmaps'] = np.random.randint(low=0, high=2,
                                                   size=(n_voxels, n_comps))
    metric_maps['Br_S0_clmaps'] = np.random.randint(low=0, high=2,
                                                    size=(n_voxels, n_comps))
    metric_maps['Br_R2_clmaps'] = np.random.randint(low=0, high=2,
                                                    size=(n_voxels, n_comps))

    comptable = kundu_fit.kundu_metrics(comptable, metric_maps)
    assert comptable is not None
