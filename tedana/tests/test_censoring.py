"""Tests for temporal censoring in tedana."""

import os.path as op

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from nilearn import plotting

from tedana import io, utils
from tedana.decomposition import project_censored_mixing
from tedana.metrics.frequency import calculate_hfc
from tedana.reporting.static_figures import _censored_runs, _mark_censored_volumes
from tedana.tests.utils import get_test_data_path
from tedana.workflows import tedana as tedana_cli
from tedana.workflows.tedana import _check_censoring_options, tedana_workflow


def _censor_mask(n_vols, rng, frac=0.2):
    """Make a sample mask with scattered censored volumes plus one block."""
    sample_mask = np.ones(n_vols, dtype=bool)
    sample_mask[rng.choice(n_vols, int(frac * n_vols / 2), replace=False)] = False
    start = n_vols // 2
    sample_mask[start : start + int(frac * n_vols / 2)] = False
    return sample_mask


def test_lomb_scargle_recovers_peak():
    """The Lomb-Scargle peak matches a sinusoid's frequency despite censoring."""
    rng = np.random.default_rng(0)
    tr, n_vols, freq = 2.0, 200, 0.05
    times = np.arange(n_vols) * tr
    signal = np.sin(2 * np.pi * freq * times) + 3.0  # nonzero mean (floating mean)
    sample_mask = _censor_mask(n_vols, rng, frac=0.25)

    power, freqs = utils.get_spectrum(signal, tr, sample_mask=sample_mask)
    assert freqs[0] > 0
    assert freqs[-1] == pytest.approx(1 / (2 * tr))
    assert abs(freqs[np.argmax(power)] - freq) <= 1 / (n_vols * tr)


def test_get_spectrum_without_censoring_is_fft():
    """An all-True sample mask gives the existing FFT result."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal(50)
    expected = utils.get_spectrum(data, 2.0)
    for sample_mask in (None, np.ones(50, dtype=bool)):
        power, freqs = utils.get_spectrum(data, 2.0, sample_mask=sample_mask)
        np.testing.assert_array_equal(power, expected[0])
        np.testing.assert_array_equal(freqs, expected[1])


def test_lomb_scargle_power_full_or_retained_input():
    """Full-length and retained-only input give the same periodogram."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal((60, 3))
    sample_mask = _censor_mask(60, rng)
    power_full, freqs_full = utils.lomb_scargle_power(data, 1.5, sample_mask)
    power_ret, freqs_ret = utils.lomb_scargle_power(data[sample_mask], 1.5, sample_mask)
    np.testing.assert_allclose(power_full, power_ret)
    np.testing.assert_array_equal(freqs_full, freqs_ret)

    # Censored values never matter
    data_spiked = data.copy()
    data_spiked[~sample_mask] = 1e6
    power_spiked = utils.lomb_scargle_power(data_spiked, 1.5, sample_mask)[0]
    np.testing.assert_allclose(power_spiked, power_full)

    # Constant columns have zero power
    data[:, 1] = 4.0
    assert np.all(utils.lomb_scargle_power(data, 1.5, sample_mask)[0][:, 1] == 0)


def test_lomb_scargle_power_errors():
    """Invalid inputs raise clear errors."""
    sample_mask = np.array([True, False, True, True, False, True])
    with pytest.raises(ValueError, match="rows"):
        utils.lomb_scargle_power(np.ones((5, 1)), 1.0, sample_mask)

    with pytest.raises(ValueError, match="At least 3 retained"):
        utils.lomb_scargle_power(np.ones((2, 1)), 1.0, np.array([True, False, True]))

    with pytest.raises(ValueError, match="positive repetition time"):
        utils.lomb_scargle_power(np.ones((6, 1)), 0.0, sample_mask)

    with pytest.raises(ValueError, match="non-finite"):
        data = np.ones((6, 1))
        data[0] = np.nan
        utils.lomb_scargle_power(data, 1.0, sample_mask)


def test_calculate_hfc_censored():
    """HFC with censoring is low for slow signals, high for fast ones, NaN for constants."""
    rng = np.random.default_rng(3)
    tr, n_vols = 1.0, 300
    times = np.arange(n_vols) * tr
    mixing = np.column_stack(
        [
            np.sin(2 * np.pi * 0.03 * times),
            np.sin(2 * np.pi * 0.45 * times),
            np.full(n_vols, 2.0),
        ]
    )
    sample_mask = _censor_mask(n_vols, rng)

    hfc = calculate_hfc(mixing=mixing[sample_mask], tr=tr, sample_mask=sample_mask)
    assert hfc[0] < 0.1
    assert hfc[1] > 0.85
    assert np.isnan(hfc[2])

    # Full-length input gives the same result as retained-only input
    hfc_full = calculate_hfc(mixing=mixing, tr=tr, sample_mask=sample_mask)
    np.testing.assert_allclose(hfc_full, hfc, equal_nan=True)

    # Without censoring, the FFT path is unchanged
    np.testing.assert_array_equal(
        calculate_hfc(mixing=mixing[:, :2], tr=tr, sample_mask=np.ones(n_vols, dtype=bool)),
        calculate_hfc(mixing=mixing[:, :2], tr=tr),
    )


def _synthetic_data(rng, n_voxels=400, n_comps=5, n_vols=120, noise=0.0):
    """Simulate data = maps @ mixing.T + voxel means (+ noise)."""
    sample_mask = _censor_mask(n_vols, rng)
    mixing = rng.standard_normal((n_vols, n_comps))
    # Center the true time series over the retained volumes, as ICA's z-scoring does
    mixing -= mixing[sample_mask].mean(axis=0)
    maps = rng.standard_normal((n_voxels, n_comps))
    means = rng.uniform(100, 200, size=(n_voxels, 1))
    data = maps @ mixing.T + means + noise * rng.standard_normal((n_voxels, n_vols))
    return data, mixing, maps, sample_mask


def test_project_censored_mixing_noise_free():
    """Retained rows are kept exactly; censored rows are recovered without noise."""
    rng = np.random.default_rng(4)
    data, mixing, _, sample_mask = _synthetic_data(rng)
    mixing_retained = mixing[sample_mask]

    mixing_full, diagnostics = project_censored_mixing(data, mixing_retained, sample_mask)
    assert mixing_full.shape == mixing.shape
    np.testing.assert_array_equal(mixing_full[sample_mask], mixing_retained)
    np.testing.assert_allclose(mixing_full[~sample_mask], mixing[~sample_mask], atol=1e-8)
    assert diagnostics["ProjectionMapsRank"] == mixing.shape[1]
    np.testing.assert_allclose(diagnostics["RetainedProjectionCorrelation"], 1.0, atol=1e-8)


def test_project_censored_mixing_ignores_censored_values_for_maps():
    """Censored values affect only their own projected rows."""
    rng = np.random.default_rng(5)
    data, mixing, _, sample_mask = _synthetic_data(rng, noise=0.5)
    mixing_retained = mixing[sample_mask]
    mixing_full, _ = project_censored_mixing(data, mixing_retained, sample_mask)

    data_spiked = data.copy()
    n_censored = int((~sample_mask).sum())
    data_spiked[:, ~sample_mask] += 1e4 * rng.standard_normal((data.shape[0], n_censored))
    mixing_full_spiked, diagnostics = project_censored_mixing(
        data_spiked,
        mixing_retained,
        sample_mask,
    )
    np.testing.assert_array_equal(mixing_full_spiked[sample_mask], mixing_full[sample_mask])
    assert not np.allclose(mixing_full_spiked[~sample_mask], mixing_full[~sample_mask])
    # Retained-volume diagnostics do not depend on censored values either
    assert diagnostics["RetainedProjectionCorrelation"] == pytest.approx(
        project_censored_mixing(data, mixing_retained, sample_mask)[1][
            "RetainedProjectionCorrelation"
        ]
    )


def test_project_censored_mixing_errors():
    """Non-identifiable or inconsistent inputs raise errors."""
    rng = np.random.default_rng(6)
    data, mixing, _, sample_mask = _synthetic_data(rng)
    mixing_retained = mixing[sample_mask].copy()

    with pytest.raises(ValueError, match="rows"):
        project_censored_mixing(data, mixing_retained[:-1], sample_mask)

    with pytest.raises(ValueError, match="samples x"):
        project_censored_mixing(data[:, :-1], mixing_retained, sample_mask)

    # Duplicate a component, so the maps are rank-deficient
    mixing_dup = np.column_stack([mixing_retained, mixing_retained[:, 0]])
    with pytest.raises(ValueError, match="rank-deficient"):
        project_censored_mixing(data, mixing_dup, sample_mask)

    few = np.zeros(data.shape[1], dtype=bool)
    few[:3] = True
    with pytest.raises(ValueError, match="not enough"):
        project_censored_mixing(data, mixing[few], few)


def test_denoise_ts_censored():
    """Denoising coefficients come from retained volumes; outputs keep every volume."""
    rng = np.random.default_rng(7)
    data, mixing, _, sample_mask = _synthetic_data(rng, noise=1.0)
    mask = np.ones(data.shape[0], dtype=bool)
    mask[:10] = False
    component_table = pd.DataFrame(
        {"classification": ["accepted", "rejected", "accepted", "rejected", "accepted"]}
    )

    dnts, hikts, lowkts = io.denoise_ts(data, mixing, mask, component_table, sample_mask)
    assert dnts.shape == hikts.shape == lowkts.shape == data.shape
    np.testing.assert_allclose(dnts[mask], data[mask] - lowkts[mask])

    # Changing only censored volumes leaves every retained output unchanged
    data_spiked = data.copy()
    data_spiked[:, ~sample_mask] += 1e4
    dnts_s, hikts_s, lowkts_s = io.denoise_ts(
        data_spiked,
        mixing,
        mask,
        component_table,
        sample_mask,
    )
    np.testing.assert_allclose(hikts_s, hikts)
    np.testing.assert_allclose(lowkts_s, lowkts)
    np.testing.assert_allclose(dnts_s[:, sample_mask], dnts[:, sample_mask])

    # Without censoring, the existing behavior is unchanged
    expected = io.denoise_ts(data, mixing, mask, component_table)
    for got, exp in zip(io.denoise_ts(data, mixing, mask, component_table, None), expected):
        np.testing.assert_array_equal(got, exp)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tedpca": "aic", "gscontrol": [None], "external_regressors": None}, "--tedpca aic"),
        ({"tedpca": "mdl", "gscontrol": [None], "external_regressors": None}, "--tedpca mdl"),
        ({"tedpca": "kundu", "gscontrol": ["gsr"], "external_regressors": None}, "gsr"),
        ({"tedpca": 0.95, "gscontrol": ["mir"], "external_regressors": None}, "mir"),
        ({"tedpca": 10, "gscontrol": [None], "external_regressors": "x.tsv"}, "External"),
    ],
)
def test_check_censoring_options(kwargs, match):
    """Options that cannot yet honor censoring are rejected."""
    with pytest.raises(ValueError, match=match):
        _check_censoring_options(**kwargs)


def test_check_censoring_options_allowed():
    """Supported options pass."""
    for tedpca in ("kundu", "kundu-stabilize", 0.95, 20):
        _check_censoring_options(tedpca=tedpca, gscontrol=[None], external_regressors=None)


def test_tedana_censoring_rejects_default_tedpca(tmp_path):
    """tedana --exclude fails early with the default (moving-average) PCA."""
    data_dir = get_test_data_path()
    data = [op.join(data_dir, f"echo{echo}.nii.gz") for echo in range(1, 4)]
    with pytest.raises(ValueError, match="--tedpca aic cannot be used with censoring"):
        tedana_workflow(data, [14.5, 38.5, 62.5], out_dir=str(tmp_path), exclude="2")

    # Malformed exclude strings fail before anything else
    with pytest.raises(ValueError, match="is empty"):
        tedana_workflow(data, [14.5, 38.5, 62.5], out_dir=str(tmp_path), exclude="3:2")


def test_tedana_cli_exclude_option():
    """The tedana CLI exposes --exclude."""
    parser = tedana_cli._get_parser()
    assert parser._option_string_actions["--exclude"].default is None


def test_censored_runs():
    """Runs of censored volumes are reported as (start, stop) pairs, stop exclusive."""
    sample_mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert _censored_runs(sample_mask) == [(0, 1), (3, 5), (6, 7)]
    assert _censored_runs(np.ones(5, dtype=bool)) == []


@pytest.mark.parametrize("n_vols", [20, 1000])
def test_mark_censored_volumes_on_carpet(n_vols):
    """Censor markers line up with the carpet's columns, including when nilearn downsamples."""
    rng = np.random.default_rng(0)
    img = nb.Nifti1Image(rng.standard_normal((3, 3, 3, n_vols)).astype(np.float32), np.eye(4))
    img.header.set_zooms((1, 1, 1, 2.0))
    mask_img = nb.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4))
    sample_mask = np.ones(n_vols, dtype=bool)
    sample_mask[[2, 3, 10]] = False

    fig, ax = plt.subplots()
    plotting.plot_carpet(
        img,
        mask_img,
        figure=fig,
        axes=ax,
        standardize="zscore_sample",
        title="Data",
    )
    carpet_ax = [axis for axis in fig.axes if axis.images][-1]
    n_patches = len(carpet_ax.patches)
    _mark_censored_volumes(fig, sample_mask)

    new_patches = carpet_ax.patches[n_patches:]
    factor = n_vols // carpet_ax.images[-1].get_array().shape[1]
    assert factor == (1 if n_vols == 20 else 2)
    assert [(patch.get_x(), patch.get_width()) for patch in new_patches] == [
        (2 / factor - 0.5, 2 / factor),
        (10 / factor - 0.5, 1 / factor),
    ]
    assert "3 of" in carpet_ax.get_title()

    # No censoring draws nothing
    n_patches = len(carpet_ax.patches)
    _mark_censored_volumes(fig, None)
    assert len(carpet_ax.patches) == n_patches
    plt.close(fig)
