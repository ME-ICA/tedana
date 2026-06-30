import os.path as op

import nibabel as nb
import numpy as np

from tedana import io, utils
from tedana.metrics import artifact
from tedana.metrics._utils import dependency_resolver


def _grid_mask(shape=(10, 10, 12)):
    aff = np.eye(4)
    mask = np.ones(shape, dtype=np.int16)
    return nb.Nifti1Image(mask, aff), shape


def test_slice_banding_flags_banded_and_spares_smooth():
    mask_img, shape = _grid_mask()
    nx, ny, nz = shape
    n_vox = nx * ny * nz
    # Component 0: alternating-slice banding along z (multiband signature).
    z = np.arange(nz)
    band = ((-1.0) ** z)[None, None, :] * np.ones((nx, ny, 1))
    # Component 1: smooth ramp along z (anatomical gradient, not multiband).
    ramp = (z / nz)[None, None, :] * np.ones((nx, ny, 1))
    # Component 2: spatially random (no slice structure).
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(shape)
    weight_maps = np.stack([band.reshape(-1), ramp.reshape(-1), rand.reshape(-1)], axis=1)
    out = artifact.compute_slice_banding(weight_maps=weight_maps, mask_img=mask_img)
    assert out.shape == (3,)
    assert out[0] > 0.3  # alternating banding -> high
    assert out[1] < 0.1  # smooth ramp -> low (bandMB small)
    assert out[2] < 0.1  # random -> low
    # Scale-invariance.
    out2 = artifact.compute_slice_banding(weight_maps=weight_maps * 7.0, mask_img=mask_img)
    assert np.allclose(out, out2)


def test_spike_flags_transient_and_spares_smooth():
    rng = np.random.default_rng(1)
    n_t = 200
    smooth = np.sin(np.linspace(0, 6 * np.pi, n_t))  # oscillatory signal
    drifty = smooth + np.linspace(-3, 3, n_t)  # signal + linear drift
    spiky = rng.standard_normal(n_t) * 0.1
    spiky[100] += 12.0  # single transient
    mixing = np.stack([smooth, drifty, spiky], axis=1)
    out = artifact.compute_spike(mixing=mixing)
    assert out.shape == (3,)
    assert out[2] > out[0]  # spike >> smooth
    assert out[2] > out[1]  # spike >> drifty
    assert out[1] < 3.0  # linear drift removed -> not flagged


def test_slice_banding_and_spike_are_registered():
    """slice_banding and spike must appear in metrics.json with correct dependencies."""
    cfg = io.load_json(op.join(utils.get_resource_path(), "config", "metrics.json"))
    assert "slice_banding" in cfg["dependencies"]
    assert "spike" in cfg["dependencies"]
    required = dependency_resolver(cfg["dependencies"], ["slice_banding", "spike"], cfg["inputs"])
    assert "map weight" in required  # slice_banding dependency
    assert "mixing" in required  # spike dependency
