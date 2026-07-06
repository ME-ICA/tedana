import os.path as op

import nibabel as nb
import numpy as np
import pandas as pd

from tedana import io, utils
from tedana.metrics import spatial, temporal
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
    out = spatial.compute_slice_banding(weight_maps=weight_maps, mask_img=mask_img)
    assert out.shape == (3,)
    assert out[0] > 0.3  # alternating banding -> high
    assert out[1] < 0.1  # smooth ramp -> low (bandMB small)
    assert out[2] < 0.1  # random -> low
    # Scale-invariance.
    out2 = spatial.compute_slice_banding(weight_maps=weight_maps * 7.0, mask_img=mask_img)
    assert np.allclose(out, out2)


def test_spike_flags_transient_and_spares_smooth():
    rng = np.random.default_rng(1)
    n_t = 200
    smooth = np.sin(np.linspace(0, 6 * np.pi, n_t))  # oscillatory signal
    drifty = smooth + np.linspace(-3, 3, n_t)  # signal + linear drift
    spiky = rng.standard_normal(n_t) * 0.1
    spiky[100] += 12.0  # single transient
    mixing = np.stack([smooth, drifty, spiky], axis=1)
    out = temporal.compute_spike(mixing=mixing)
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


def test_candidate_group_spacings_uses_mb_when_divisible():
    assert spatial._candidate_group_spacings(12, 3) == [4]


def test_candidate_group_spacings_scans_divisors_without_mb():
    # divisors g in [2, n//2]: for 12 -> 2,3,4,6
    assert spatial._candidate_group_spacings(12, None) == [2, 3, 4, 6]


def test_aliasing_stat_high_for_grouped_identical_slices():
    rng = np.random.default_rng(0)
    n, g = 12, 4
    base = {p: rng.standard_normal(25) for p in range(g)}
    planes = []
    for s in range(n):
        v = base[s % g].copy()
        planes.append((v, np.ones(25, dtype=bool)))
    corr = spatial._pairwise_slice_corr(planes)
    grouped = spatial._aliasing_stat(corr, np.arange(n), g)
    shuffled = spatial._aliasing_stat(corr, rng.permutation(n), g)
    assert grouped > 0.9  # partners identical -> ~1
    assert grouped > shuffled


def test_periodicity_stat_high_for_periodic_profile():
    n, g = 12, 4
    profile = np.tile([3.0, 0.0, 0.0, 0.0], n // g)  # period 4 -> freq 1/4
    flat = np.ones(n)
    assert spatial._periodicity_stat(profile, g) > spatial._periodicity_stat(flat, g)
    assert spatial._periodicity_stat(profile, g) > 0.3


def _leakage_volume(shape, n_slices, mb, rng, leak=True):
    """Build a (X,Y,Z=n_slices) squared-weight volume with optional SMS leakage at spacing n_slices//mb."""
    g = n_slices // mb
    nx, ny = shape
    patterns = {p: rng.standard_normal((nx, ny)) ** 2 for p in range(g)}
    amp = {p: 1.0 + p for p in range(g)}  # periodic energy across groups
    vol = np.zeros((nx, ny, n_slices))
    for s in range(n_slices):
        if leak:
            vol[:, :, s] = patterns[s % g] * amp[s % g]
        else:
            vol[:, :, s] = rng.standard_normal((nx, ny)) ** 2
    return vol


def test_slice_leakage_flags_injected_sms_leakage():
    mask_img, shape = _grid_mask((8, 8, 12))
    rng = np.random.default_rng(0)
    leak = _leakage_volume((8, 8), 12, 3, rng, leak=True)
    clean = _leakage_volume((8, 8), 12, 3, np.random.default_rng(1), leak=False)
    maps = np.stack([leak.reshape(-1), clean.reshape(-1)], axis=1)
    out = spatial.compute_slice_leakage(
        weight_maps=maps,
        mask_img=mask_img,
        slice_axis=2,
        mb_factor=3,
        n_permutations=200,
        seed=0,
    )
    assert out["slice_leakage"].shape == (2,)
    assert out["slice_leakage"][0] > 3.0  # injected leakage -> strong
    assert out["slice_leakage"][0] > out["slice_leakage"][1]
    assert out["aliasing_z"][0] > out["aliasing_z"][1]
    assert out["periodicity_z"][0] > out["periodicity_z"][1]


def test_slice_leakage_zero_when_no_multiband():
    mask_img, shape = _grid_mask((6, 6, 12))
    rng = np.random.default_rng(2)
    leak = _leakage_volume((6, 6), 12, 3, rng, leak=True)
    maps = leak.reshape(-1)[:, None]
    out = spatial.compute_slice_leakage(
        weight_maps=maps,
        mask_img=mask_img,
        slice_axis=2,
        mb_factor=1,
        seed=0,
    )
    assert out["slice_leakage"][0] == 0.0
    assert out["aliasing_z"][0] == 0.0
    assert out["periodicity_z"][0] == 0.0


def test_slice_leakage_reproducible_with_seed():
    mask_img, shape = _grid_mask((6, 6, 12))
    rng = np.random.default_rng(3)
    leak = _leakage_volume((6, 6), 12, 3, rng, leak=True)
    maps = leak.reshape(-1)[:, None]
    kw = dict(
        weight_maps=maps, mask_img=mask_img, slice_axis=2, mb_factor=3, n_permutations=100, seed=7
    )
    a = spatial.compute_slice_leakage(**kw)["slice_leakage"]
    b = spatial.compute_slice_leakage(**kw)["slice_leakage"]
    assert np.array_equal(a, b)


def test_slice_leakage_metadata_free_scan_path():
    # No metadata: slice axis inferred from an anisotropic affine (axis 2 = largest
    # spacing), aliasing spacing found by the divisor scan (n_slices=12 -> includes g=4).
    mask_bool = np.ones((8, 8, 12), dtype=np.int16)
    affine = np.diag([2.0, 2.0, 4.0, 1.0])  # z (axis 2) spacing largest
    mask_img = nb.Nifti1Image(mask_bool, affine)
    rng = np.random.default_rng(0)
    leak = _leakage_volume((8, 8), 12, 3, rng, leak=True)
    clean = _leakage_volume((8, 8), 12, 3, np.random.default_rng(1), leak=False)
    maps = np.stack([leak.reshape(-1), clean.reshape(-1)], axis=1)
    out = spatial.compute_slice_leakage(
        weight_maps=maps,
        mask_img=mask_img,
        slice_axis=None,
        mb_factor=None,
        n_permutations=200,
        seed=0,
    )
    assert out["slice_leakage"][0] > out["slice_leakage"][1]
    assert out["aliasing_z"][0] > 2.0  # leakage detected via the scan path


def test_slice_leakage_metrics_registered():
    cfg = io.load_json(op.join(utils.get_resource_path(), "config", "metrics.json"))
    # slice_leakage and each of its two z-scores are independently requestable metrics.
    for name in ("slice_leakage", "slice_leakage_aliasing_z", "slice_leakage_periodicity_z"):
        assert name in cfg["dependencies"]
        required = dependency_resolver(cfg["dependencies"], [name], cfg["inputs"])
        assert "map weight" in required
        assert "mask" in required


def test_slice_leakage_z_scores_have_metadata_descriptions():
    """The aliasing/periodicity z-scores are first-class metrics with output descriptions."""
    from tedana.metrics import collect

    for name in ("slice_leakage", "slice_leakage_aliasing_z", "slice_leakage_periodicity_z"):
        md = collect.get_metadata(pd.DataFrame({name: [0.0, 1.0, 2.0]}))
        assert name in md
        assert md[name]["Description"]
