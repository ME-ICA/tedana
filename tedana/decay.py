"""Functions to estimate S0 and T2* from multi-echo data."""

import logging
import os
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy import stats
from tqdm.auto import tqdm

from tedana import utils

LGR = logging.getLogger("GENERAL")
RepLGR = logging.getLogger("REPORT")

# Brain T2* at typical field strengths is well under a second. This generous
# ceiling leaves real estimates untouched while preventing the nonlinear fits
# from driving R2* toward 0, which would send T2* = 1/R2* to absurd (and, once
# cast to float32, infinite) values.
MAX_PHYSIOLOGICAL_T2STAR = 5.0  # seconds
MIN_R2STAR = 1.0 / MAX_PHYSIOLOGICAL_T2STAR  # s^-1

# A complex fit can fall into a degenerate valley where R2* grows until S0 fits
# only the first echo. Signal that decays by more than exp(-20) across the echo
# train is not measurable, so such fits are treated as failures.
MAX_DECAY_ACROSS_ECHOES = 20.0


def _frequency_half_range(echo_times):
    """Return half the frequency alias period (in Hz) for a set of echo times.

    Phase is only observed modulo 2*pi at each echo, so frequencies differing by
    ``1 / dTE`` explain evenly spaced echoes equally well. Complex fits are
    confined to ``initial estimate +/- 1 / (2 * dTE)``, using the shortest echo
    spacing, which keeps the frequency from drifting to an alias or diverging.
    """
    spacing = np.diff(np.unique(np.asarray(echo_times, dtype=float)))
    return 0.5 / spacing.min()


def _is_degenerate_decay(r2star, echo_times):
    """Flag R2* values that decay by more than ``MAX_DECAY_ACROSS_ECHOES`` across the echoes."""
    echo_span = np.max(echo_times) - np.min(echo_times)
    return r2star * echo_span >= MAX_DECAY_ACROSS_ECHOES


def _fit_voxels_cuda(
    data_2d,
    echo_times_1d,
    voxel_idx,
    s0_initial,
    t2s_initial,
    n_iterations=25,
    batch_size=32768,
):
    """Fit independent monoexponential models with batched CUDA Gauss--Newton.

    Voxels that do not meet a strict convergence check are returned as failures
    and are subsequently handled by the existing SciPy implementation.

    Parameters
    ----------
    data_2d : (N, M) :obj:`numpy.ndarray`
        Multi-echo measurements, with ``N`` observations per voxel and ``M``
        voxels in the current adaptive-mask group.
    echo_times_1d : (N,) :obj:`numpy.ndarray`
        Echo time associated with each observation, in seconds.
    voxel_idx : (V,) :obj:`numpy.ndarray`
        Indices of the ``V`` voxels in ``data_2d`` to fit.
    s0_initial : (M,) :obj:`numpy.ndarray`
        Initial S0 estimates for the adaptive-mask group.
    t2s_initial : (M,) :obj:`numpy.ndarray`
        Initial T2* estimates, in seconds, for the adaptive-mask group.
    n_iterations : :obj:`int`, optional
        Number of damped Gauss--Newton iterations. Default is 25.
    batch_size : :obj:`int`, optional
        Maximum number of voxels transferred to and processed on the GPU in
        one batch. Default is 32768.

    Returns
    -------
    fitted_s0 : (V,) :obj:`numpy.ndarray`
        Estimated S0 values. Entries for failed voxels are zero.
    fitted_t2s : (V,) :obj:`numpy.ndarray`
        Estimated T2* values in seconds. Entries for failed voxels are zero.
    failures : (V,) :obj:`numpy.ndarray`
        Boolean array indicating voxels that did not converge or had invalid
        inputs and must be refit on the CPU.
    s0_var : (V,) :obj:`numpy.ndarray`
        Estimated variance of each fitted S0 value. Entries for failed voxels
        are zero.
    t2s_var : (V,) :obj:`numpy.ndarray`
        Estimated variance of each fitted T2* value. Entries for failed voxels
        are zero.
    s0_t2s_covar : (V,) :obj:`numpy.ndarray`
        Estimated covariance between S0 and T2*. Entries for failed voxels are
        zero.

    Raises
    ------
    ImportError
        If CuPy is not installed.

    Notes
    -----
    The function synchronizes the default CUDA stream before returning so that
    callers can accurately measure elapsed execution time. It assumes CUDA
    availability has already been validated by :func:`tedana.utils._resolve_backend`.
    """
    import cupy as cp

    n_voxels = len(voxel_idx)
    fitted_s0 = np.zeros(n_voxels)
    fitted_t2s = np.zeros(n_voxels)
    failures = np.ones(n_voxels, dtype=bool)
    s0_var = np.zeros(n_voxels)
    t2s_var = np.zeros(n_voxels)
    s0_t2s_covar = np.zeros(n_voxels)
    n_observations = data_2d.shape[0]
    eps = np.finfo(np.float64).eps
    echo_times_gpu = cp.asarray(echo_times_1d, dtype=cp.float64)

    for first in range(0, n_voxels, batch_size):
        last = min(first + batch_size, n_voxels)
        batch_voxels = voxel_idx[first:last]
        observations_cpu = np.ascontiguousarray(data_2d[:, batch_voxels].T, dtype=np.float64)
        lower_s0_cpu = observations_cpu.min(axis=1)
        s0_cpu = s0_initial[batch_voxels]
        t2s_cpu = t2s_initial[batch_voxels]
        initial_valid = (
            np.isfinite(observations_cpu).all(axis=1)
            & np.isfinite(s0_cpu)
            & np.isfinite(t2s_cpu)
            & (s0_cpu >= lower_s0_cpu)
            & (t2s_cpu > 0)
        )
        if not initial_valid.any():
            continue

        local_idx = np.flatnonzero(initial_valid)
        observations = cp.asarray(observations_cpu[local_idx])
        lower_s0 = cp.asarray(lower_s0_cpu[local_idx])
        s0 = cp.asarray(s0_cpu[local_idx])
        t2s = cp.asarray(t2s_cpu[local_idx])
        relative_step = cp.full(len(local_idx), cp.inf, dtype=cp.float64)
        determinant = cp.zeros(len(local_idx), dtype=cp.float64)

        for _ in range(n_iterations):
            exponential = cp.exp(-echo_times_gpu[None, :] / t2s[:, None])
            residual = s0[:, None] * exponential - observations
            jacobian_s0 = exponential
            jacobian_t2s = s0[:, None] * exponential * echo_times_gpu[None, :] / t2s[:, None] ** 2
            a00 = cp.sum(jacobian_s0 * jacobian_s0, axis=1)
            a01 = cp.sum(jacobian_s0 * jacobian_t2s, axis=1)
            a11 = cp.sum(jacobian_t2s * jacobian_t2s, axis=1)
            b0 = -cp.sum(jacobian_s0 * residual, axis=1)
            b1 = -cp.sum(jacobian_t2s * residual, axis=1)
            # Unitless diagonal damping stabilizes the two-parameter normal equation.
            a00 *= 1.0 + 1e-8
            a11 *= 1.0 + 1e-8
            determinant = a00 * a11 - a01 * a01
            delta_s0 = (a11 * b0 - a01 * b1) / determinant
            delta_t2s = (a00 * b1 - a01 * b0) / determinant
            s0 = cp.maximum(s0 + delta_s0, lower_s0)
            t2s = cp.maximum(t2s + delta_t2s, eps)
            relative_step = cp.maximum(
                cp.abs(delta_s0) / (cp.abs(s0) + 1.0), cp.abs(delta_t2s) / t2s
            )

        exponential = cp.exp(-echo_times_gpu[None, :] / t2s[:, None])
        residual = s0[:, None] * exponential - observations
        jacobian_s0 = exponential
        jacobian_t2s = s0[:, None] * exponential * echo_times_gpu[None, :] / t2s[:, None] ** 2
        a00 = cp.sum(jacobian_s0 * jacobian_s0, axis=1)
        a01 = cp.sum(jacobian_s0 * jacobian_t2s, axis=1)
        a11 = cp.sum(jacobian_t2s * jacobian_t2s, axis=1)
        determinant = a00 * a11 - a01 * a01
        residual_variance = cp.sum(residual * residual, axis=1) / (n_observations - 2)
        fitted_s0_batch = cp.asnumpy(s0)
        fitted_t2s_batch = cp.asnumpy(t2s)
        determinant_batch = cp.asnumpy(determinant)
        relative_step_batch = cp.asnumpy(relative_step)
        s0_var_batch = cp.asnumpy(a11 / determinant * residual_variance)
        t2s_var_batch = cp.asnumpy(a00 / determinant * residual_variance)
        s0_t2s_covar_batch = cp.asnumpy(-a01 / determinant * residual_variance)
        converged = (
            np.isfinite(fitted_s0_batch)
            & np.isfinite(fitted_t2s_batch)
            & np.isfinite(determinant_batch)
            & (determinant_batch > 0)
            & np.isfinite(relative_step_batch)
            & (relative_step_batch < 1e-8)
            & (n_observations > 2)
        )
        output_idx = first + local_idx[converged]
        fitted_s0[output_idx] = fitted_s0_batch[converged]
        fitted_t2s[output_idx] = fitted_t2s_batch[converged]
        s0_var[output_idx] = s0_var_batch[converged]
        t2s_var[output_idx] = t2s_var_batch[converged]
        s0_t2s_covar[output_idx] = s0_t2s_covar_batch[converged]
        failures[output_idx] = False

    cp.cuda.Stream.null.synchronize()
    return fitted_s0, fitted_t2s, failures, s0_var, t2s_var, s0_t2s_covar


def _apply_t2s_floor(t2s, echo_times):
    """Apply a floor to T2* values to prevent zero division errors during optimal combination.

    Parameters
    ----------
    t2s : (S [x T]) array_like
        T2* estimates.
    echo_times : (E,) array_like
        Echo times in seconds.

    Returns
    -------
    t2s_corrected : (S,) array_like
        T2* estimates with very small, positive values replaced with a floor value.
    """
    t2s_corrected = t2s.copy()

    if t2s.ndim == 2:
        for i_vol in range(t2s.shape[1]):
            t2s_corrected[:, i_vol] = _apply_t2s_floor(t2s[:, i_vol], echo_times)

        return t2s_corrected

    echo_times = np.asarray(echo_times)
    if echo_times.ndim == 1:
        echo_times = echo_times[:, None]

    eps = np.finfo(dtype=t2s.dtype).eps  # smallest value for datatype
    nonzerovox = t2s != 0
    # Exclude values where t2s is 0 when dividing by t2s.
    # These voxels are also excluded from bad_voxel_idx
    temp_arr = np.zeros((len(echo_times), len(t2s)))
    temp_arr[:, nonzerovox] = np.exp(-echo_times / t2s[nonzerovox])  # (E x V) array
    bad_voxel_idx = np.any(temp_arr == 0, axis=0) & (t2s != 0)
    n_bad_voxels = np.sum(bad_voxel_idx)
    if n_bad_voxels > 0:
        n_voxels = temp_arr.size
        floor_percent = 100 * n_bad_voxels / n_voxels
        LGR.debug(
            f"T2* values for {n_bad_voxels}/{n_voxels} voxels ({floor_percent:.2f}%) have been "
            "identified as close to zero and have been adjusted"
        )
    t2s_corrected[bad_voxel_idx] = np.min(-echo_times) / np.log(eps)
    return t2s_corrected


def monoexponential(tes, s0, t2star):
    """Specify a monoexponential model for use with scipy curve fitting.

    Parameters
    ----------
    tes : (E,) :obj:`list`
        Echo times
    s0 : :obj:`float`
        Initial signal parameter
    t2star : :obj:`float`
        T2* parameter

    Returns
    -------
    : obj:`float`
        Predicted signal
    """
    return s0 * np.exp(-tes / t2star)


def _fit_single_voxel(voxel, echo_times_1d, data_column, s0_init, t2s_init, bounds):
    """Fit monoexponential model for a single voxel.

    Parameters
    ----------
    voxel : int
        Voxel index
    echo_times_1d : (E*T,) array_like
        Echo times repeated for each timepoint
    data_column : (E*T,) array_like
        Data for this voxel across all echoes and timepoints
    s0_init : float
        Initial S0 estimate
    t2s_init : float
        Initial T2* estimate
    bounds : tuple
        Bounds for curve_fit

    Returns
    -------
    result : tuple or None
        If successful: (voxel, s0, t2s, False, s0_var, t2s_var, s0t2s_covar)
        If failed: (voxel, None, None, True, None, None, None)
    """
    try:
        popt, cov = scipy.optimize.curve_fit(
            monoexponential,
            echo_times_1d,
            data_column,
            p0=(s0_init, t2s_init),
            bounds=bounds,
        )
        return (voxel, popt[0], popt[1], False, cov[0, 0], cov[1, 1], cov[0, 1])
    except (RuntimeError, ValueError):
        return (voxel, None, None, True, None, None, None)


def complex_decay_model(echo_times, s0, r2star, frequency_hz=0.0, modulation=1.0):
    """Evaluate a single-pool complex R2* decay model.

    The model is ``S(TE) = S0 * exp((-R2* + 1j*2*pi*f) * TE) * modulation``.
    ``s0`` is complex and absorbs the initial phase. ``modulation`` is a
    forward-compatibility hook for a future Dahnke correction; pass 1 for none.

    Parameters
    ----------
    echo_times : (E,) array_like
        Echo times in seconds.
    s0 : complex or array_like
        Complex signal at TE=0.
    r2star : float or array_like
        R2* decay rate in s^-1.
    frequency_hz : float or array_like, optional
        Off-resonance frequency in Hz. Default is 0.0.
    modulation : complex or array_like, optional
        Through-slice modulation term. Default is 1.0 (no modulation).

    Returns
    -------
    signal : :obj:`numpy.ndarray`
        Complex predicted signal, broadcast over the inputs.
    """
    echo_times = np.asarray(echo_times, dtype=float)
    decay = -np.asarray(r2star) + 1j * 2.0 * np.pi * np.asarray(frequency_hz)
    return np.asarray(s0) * np.exp(decay * echo_times) * np.asarray(modulation)


def _initial_complex_decay_params(signal, echo_times):
    """Derive initial ``[log|S0|, R2*, frequency_hz, phase0]`` for one echo train.

    Parameters
    ----------
    signal : (E,) array_like
        Complex echo train for one sample.
    echo_times : (E,) array_like
        Echo times in seconds.

    Returns
    -------
    params : (4,) :obj:`numpy.ndarray`
        Initial estimates from a log-linear magnitude fit (slope -> R2*,
        intercept -> log|S0|) and an unwrapped-phase linear fit
        (slope -> frequency_hz, intercept -> phase0).
    """
    signal = np.asarray(signal)
    echo_times = np.asarray(echo_times, dtype=float)
    amplitude = np.maximum(np.abs(signal), np.finfo(float).tiny)
    if echo_times.size > 1:
        slope, intercept = np.polyfit(echo_times, np.log(amplitude), 1)
        r2star = max(0.0, -float(slope))
        phase = np.unwrap(np.angle(signal))
        phase_slope, phase_intercept = np.polyfit(echo_times, phase, 1)
        frequency_hz = float(phase_slope / (2.0 * np.pi))
        phase0 = float(phase_intercept)
    else:
        intercept = float(np.log(amplitude[0]))
        r2star = 0.0
        frequency_hz = 0.0
        phase0 = float(np.angle(signal[0]))
    return np.array([float(intercept), r2star, frequency_hz, phase0], dtype=float)


def _fit_complex_decay_1d(signal, echo_times, *, lower_bounds, upper_bounds, max_nfev=None):
    """Fit one complex echo train with nonlinear least squares.

    Parameters
    ----------
    signal : (E,) array_like
        Complex-valued data for one sample.
    echo_times : (E,) array_like
        Echo times in seconds.
    lower_bounds, upper_bounds : (4,) array_like
        Bounds for ``[log|S0|, R2*, frequency_hz, phase0]``. The frequency
        bounds are further limited to one alias period around the initial
        estimate (see :func:`_frequency_half_range`).
    max_nfev : int or None, optional
        Maximum function evaluations for :func:`scipy.optimize.least_squares`.

    Returns
    -------
    result : dict or None
        Dict with ``s0`` (complex), ``r2star``, ``frequency_hz``, ``phase0``,
        ``cost``, ``nfev``, ``success``. ``None`` only when fewer than 2 finite
        echoes are available. On optimizer error, or when the fit is degenerate
        (see :func:`_is_degenerate_decay`), falls back to the initial estimate
        with ``success=False``.
    """
    signal = np.asarray(signal)
    echo_times = np.asarray(echo_times, dtype=float)
    valid = np.isfinite(signal.real) & np.isfinite(signal.imag)
    if int(valid.sum()) < 2:
        return None

    y_valid = signal[valid]
    te_valid = echo_times[valid]
    x0 = _initial_complex_decay_params(y_valid, te_valid)
    x0 = np.minimum(np.maximum(x0, lower_bounds), upper_bounds)
    lower_bounds = np.array(lower_bounds, dtype=float)
    upper_bounds = np.array(upper_bounds, dtype=float)
    if np.unique(te_valid).size > 1:
        half_range = _frequency_half_range(te_valid)
        lower_bounds[2] = max(lower_bounds[2], x0[2] - half_range)
        upper_bounds[2] = min(upper_bounds[2], x0[2] + half_range)

    def residuals(params):
        log_s0_abs, r2, freq, phi0 = params
        with np.errstate(over="ignore", invalid="ignore"):
            pred = complex_decay_model(te_valid, np.exp(log_s0_abs) * np.exp(1j * phi0), r2, freq)
            residual = pred - y_valid
        return np.concatenate([residual.real, residual.imag])

    def initial_estimate():
        log_s0_abs, r2star, frequency_hz, phase0 = x0
        return {
            "s0": np.exp(log_s0_abs) * np.exp(1j * phase0),
            "r2star": r2star,
            "frequency_hz": frequency_hz,
            "phase0": phase0,
            "cost": np.nan,
            "nfev": 0,
            "success": False,
        }

    try:
        result = scipy.optimize.least_squares(
            residuals, x0, bounds=(lower_bounds, upper_bounds), max_nfev=max_nfev
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return initial_estimate()

    log_s0_abs, r2star, frequency_hz, phase0 = result.x
    if _is_degenerate_decay(r2star, te_valid):
        return initial_estimate()

    return {
        "s0": np.exp(log_s0_abs) * np.exp(1j * phase0),
        "r2star": r2star,
        "frequency_hz": frequency_hz,
        "phase0": phase0,
        "cost": result.cost,
        "nfev": result.nfev,
        "success": result.success,
    }


def _project_complex_s0(xp, signal, echo_times, r2star, frequency_hz):
    """Solve per-group complex S0 for fixed R2*/frequency and return the residuals.

    Parameters
    ----------
    xp : module
        Array module (:mod:`numpy` or :mod:`cupy`).
    signal : (B, G, E) complex array
        Batched complex data with ``G`` groups of ``E`` echoes per problem.
    echo_times : (E,) array
        Echo times in seconds.
    r2star, frequency_hz : (B,) array
        Shared R2* (s^-1) and off-resonance frequency (Hz) for each problem.

    Returns
    -------
    basis : (B, E) complex array
        Decay basis ``exp((-R2* + 1j*2*pi*f) * TE)``.
    basis_norm : (B,) array
        Squared norm of ``basis``.
    s0 : (B, G) complex array
        Least-squares complex S0 for each group.
    residual : (B, G, E) complex array
        ``s0 * basis - signal``.
    cost : (B,) array
        Half the sum of squared residual magnitudes.
    """
    basis = xp.exp((-r2star[:, None] + 2j * np.pi * frequency_hz[:, None]) * echo_times[None, :])
    basis_norm = xp.sum(basis.real**2 + basis.imag**2, axis=1)
    s0 = xp.sum(xp.conj(basis)[:, None, :] * signal, axis=2) / basis_norm[:, None]
    residual = s0[:, :, None] * basis[:, None, :] - signal
    cost = 0.5 * xp.sum(residual.real**2 + residual.imag**2, axis=(1, 2))
    return basis, basis_norm, s0, residual, cost


def _gauss_newton_complex(xp, signal, echo_times, basis, basis_norm, s0, residual):
    """Build the 2x2 Gauss--Newton system for the variable-projection functional.

    The residual ``P y - y`` (with ``P`` the projection onto the decay basis)
    is differentiated exactly with respect to R2* and frequency, including the
    dependence of the projected S0 on those parameters.

    Returns
    -------
    a00, a01, a11 : (B,) array
        Entries of ``J^T J`` for ``[R2*, frequency_hz]``.
    g0, g1 : (B,) array
        Entries of ``J^T r``.
    """
    d_basis_r2star = -echo_times[None, :] * basis
    d_basis_freq = 2j * np.pi * echo_times[None, :] * basis
    jacobians = []
    for d_basis in (d_basis_r2star, d_basis_freq):
        d_norm = 2.0 * xp.sum((xp.conj(basis) * d_basis).real, axis=1)
        d_proj = xp.sum(xp.conj(d_basis)[:, None, :] * signal, axis=2)
        coefficient = (d_proj - s0 * d_norm[:, None]) / basis_norm[:, None]
        jacobians.append(
            d_basis[:, None, :] * s0[:, :, None] + basis[:, None, :] * coefficient[:, :, None]
        )
    jac_r2star, jac_freq = jacobians

    def _inner(left, right):
        return xp.sum((xp.conj(left) * right).real, axis=(1, 2))

    return (
        _inner(jac_r2star, jac_r2star),
        _inner(jac_r2star, jac_freq),
        _inner(jac_freq, jac_freq),
        _inner(jac_r2star, residual),
        _inner(jac_freq, residual),
    )


def _fit_complex_groups_batched(
    xp,
    signal,
    echo_times,
    fit_groups=None,
    min_r2star=MIN_R2STAR,
    n_iterations=100,
    tolerance=1e-7,
):
    """Fit shared R2*/frequency with per-group complex S0 for a batch of problems.

    This is a vectorized Levenberg--Marquardt solver for the variable-projection
    form of the complex monoexponential model. Each of the ``B`` problems has
    ``G`` groups of ``E`` echoes that share R2* and frequency, while each group
    has its own complex S0, which is solved analytically. It covers every
    complex fit mode: ``fitmode="all"`` and ``"ts"`` are problems with a single
    group, and ``fitmode="varys0"`` uses one group per volume. The solver is
    written against an array module so that the same code runs on the CPU
    (NumPy) and on CUDA devices (CuPy).

    Parameters
    ----------
    xp : module
        Array module (:mod:`numpy` or :mod:`cupy`).
    signal : (B, G, E) complex array
        Finite complex data, already on the device associated with ``xp``.
    echo_times : (E,) :obj:`numpy.ndarray`
        Echo times in seconds, on the host. Must not all be equal.
    fit_groups : (G,) bool array or None, optional
        Groups used to estimate R2*/frequency. S0 is solved for all groups.
        None uses every group.
    min_r2star : float, optional
        Lower bound on R2* in s^-1. Default is ``MIN_R2STAR``.
    n_iterations : int, optional
        Maximum number of Levenberg--Marquardt iterations. Default is 100.
    tolerance : float, optional
        Relative size of the undamped Gauss--Newton step below which a problem
        is considered converged. Default is 1e-7.

    Returns
    -------
    r2star, frequency_hz : (B,) array
        Estimated R2* (s^-1) and frequency (Hz).
    s0 : (B, G) complex array
        Complex S0 for every group, including those excluded by ``fit_groups``.
    converged : (B,) bool array
        Whether each problem reached a stationary point of the bounded problem
        without becoming degenerate (see :func:`_is_degenerate_decay`).

    Notes
    -----
    R2* is bounded below by ``min_r2star`` and frequency is confined to one
    alias period around its initial estimate (see :func:`_frequency_half_range`),
    matching the SciPy implementation.
    """
    echo_times = np.asarray(echo_times, dtype=float)
    half_range = _frequency_half_range(echo_times)
    echo_span = echo_times.max() - echo_times.min()
    signal = xp.asarray(signal, dtype=xp.complex128)
    echo_times = xp.asarray(echo_times, dtype=xp.float64)
    fit_signal = signal if fit_groups is None else signal[:, xp.asarray(fit_groups)]

    # Initial values mirror _initial_complex_decay_params: per-group log-linear
    # magnitude and unwrapped-phase fits, summarized by the median across groups.
    centered_te = echo_times - xp.mean(echo_times)
    te_ss = xp.sum(centered_te**2)
    log_amplitude = xp.log(xp.maximum(xp.abs(fit_signal), np.finfo(float).tiny))
    phase = xp.unwrap(xp.angle(fit_signal), axis=-1)
    r2star_groups = xp.maximum(-xp.sum(centered_te * log_amplitude, axis=-1) / te_ss, 0.0)
    freq_groups = xp.sum(centered_te * phase, axis=-1) / te_ss / (2.0 * np.pi)
    r2star = xp.maximum(xp.median(r2star_groups, axis=1), min_r2star)
    frequency_hz = xp.median(freq_groups, axis=1)
    min_freq = frequency_hz - half_range
    max_freq = frequency_hz + half_range

    damping = xp.full(r2star.shape, 1e-3)
    converged = xp.zeros(r2star.shape, dtype=bool)
    basis, basis_norm, s0, residual, cost = _project_complex_s0(
        xp, fit_signal, echo_times, r2star, frequency_hz
    )
    for i_iter in range(n_iterations):
        a00, a01, a11, g0, g1 = _gauss_newton_complex(
            xp, fit_signal, echo_times, basis, basis_norm, s0, residual
        )

        # A parameter is held at its bound when the undamped step points outward;
        # the other parameter then takes a one-dimensional step.
        determinant = a00 * a11 - a01 * a01
        step_r2star = -(a11 * g0 - a01 * g1) / determinant
        step_freq = -(a00 * g1 - a01 * g0) / determinant
        hold_r2star = (r2star <= min_r2star) & (step_r2star < 0)
        hold_freq = ((frequency_hz <= min_freq) & (step_freq < 0)) | (
            (frequency_hz >= max_freq) & (step_freq > 0)
        )

        # The projected undamped step measures stationarity.
        proj_r2star = xp.where(hold_r2star, 0.0, xp.where(hold_freq, -g0 / a00, step_r2star))
        proj_freq = xp.where(hold_freq, 0.0, xp.where(hold_r2star, -g1 / a11, step_freq))
        relative_step = xp.maximum(
            xp.abs(proj_r2star) / xp.maximum(r2star, 1.0),
            xp.abs(proj_freq) / (xp.abs(frequency_hz) + 1.0),
        )
        converged = xp.isfinite(relative_step) & (relative_step < tolerance)
        if i_iter % 5 == 4 and bool(xp.all(converged)):
            break

        # Levenberg--Marquardt step with Marquardt (diagonal) scaling, projected
        # the same way and clipped to the bounds.
        d00 = a00 * (1.0 + damping)
        d11 = a11 * (1.0 + damping)
        determinant = d00 * d11 - a01 * a01
        lm_r2star = -(d11 * g0 - a01 * g1) / determinant
        lm_freq = -(d00 * g1 - a01 * g0) / determinant
        lm_r2star = xp.where(hold_r2star, 0.0, xp.where(hold_freq, -g0 / d00, lm_r2star))
        lm_freq = xp.where(hold_freq, 0.0, xp.where(hold_r2star, -g1 / d11, lm_freq))
        new_r2star = xp.maximum(r2star + lm_r2star, min_r2star)
        new_freq = xp.clip(frequency_hz + lm_freq, min_freq, max_freq)
        new_basis, new_norm, new_s0, new_residual, new_cost = _project_complex_s0(
            xp, fit_signal, echo_times, new_r2star, new_freq
        )
        accept = (
            xp.isfinite(new_cost)
            & xp.isfinite(new_r2star)
            & xp.isfinite(new_freq)
            & (new_cost <= cost)
            & ~converged
        )
        r2star = xp.where(accept, new_r2star, r2star)
        frequency_hz = xp.where(accept, new_freq, frequency_hz)
        basis = xp.where(accept[:, None], new_basis, basis)
        basis_norm = xp.where(accept, new_norm, basis_norm)
        s0 = xp.where(accept[:, None], new_s0, s0)
        residual = xp.where(accept[:, None, None], new_residual, residual)
        cost = xp.where(accept, new_cost, cost)
        damping = xp.where(
            accept, xp.maximum(damping * 0.1, 1e-12), xp.minimum(damping * 10.0, 1e12)
        )

    basis, basis_norm, s0, _, _ = _project_complex_s0(xp, signal, echo_times, r2star, frequency_hz)
    converged &= xp.isfinite(r2star) & xp.isfinite(frequency_hz) & (basis_norm > 0)
    converged &= xp.all(xp.isfinite(s0), axis=1)
    # Degenerate fits are handed to the SciPy implementation, which flags them
    # as failures if it reaches the same solution.
    converged &= r2star * echo_span < MAX_DECAY_ACROSS_ECHOES
    return r2star, frequency_hz, s0, converged


def _fit_complex_groups_cuda(signal, echo_times, fit_groups=None, batch_size=None):
    """Run :func:`_fit_complex_groups_batched` on a CUDA device in batches.

    Parameters
    ----------
    signal : (B, G, E) complex :obj:`numpy.ndarray`
        Complex data for ``B`` problems.
    echo_times : (E,) :obj:`numpy.ndarray`
        Echo times in seconds.
    fit_groups : (G,) bool :obj:`numpy.ndarray` or None, optional
        Groups used to estimate R2*/frequency.
    batch_size : int or None, optional
        Maximum number of problems transferred to the GPU at once. Batches are
        always limited to at most 2**22 complex samples, which bounds device
        memory use to roughly 1-2 GB. None (default) applies only that limit.
        The solver's run time is dominated by kernel launches, so larger
        batches are much faster.

    Returns
    -------
    r2star, frequency_hz : (B,) :obj:`numpy.ndarray`
        Estimated R2* and frequency. Zero for failed problems.
    s0 : (B, G) complex :obj:`numpy.ndarray`
        Complex S0. Zero for failed problems.
    failures : (B,) bool :obj:`numpy.ndarray`
        Problems that had non-finite inputs or did not converge, and must be
        refit on the CPU.
    """
    import cupy as cp

    n_problems, n_groups, n_echoes = signal.shape
    r2star = np.zeros(n_problems)
    frequency_hz = np.zeros(n_problems)
    s0 = np.zeros((n_problems, n_groups), dtype=np.complex128)
    failures = np.ones(n_problems, dtype=bool)
    max_batch_size = max(1, 2**22 // (n_groups * n_echoes))
    batch_size = max_batch_size if batch_size is None else min(batch_size, max_batch_size)

    for first in range(0, n_problems, batch_size):
        batch = signal[first : first + batch_size]
        valid = np.flatnonzero(np.isfinite(batch).all(axis=(1, 2)))
        if valid.size == 0:
            continue

        r2_gpu, freq_gpu, s0_gpu, converged_gpu = _fit_complex_groups_batched(
            cp, cp.asarray(batch[valid]), echo_times, fit_groups=fit_groups
        )
        converged = cp.asnumpy(converged_gpu)
        output_idx = first + valid[converged]
        r2star[output_idx] = cp.asnumpy(r2_gpu)[converged]
        frequency_hz[output_idx] = cp.asnumpy(freq_gpu)[converged]
        s0[output_idx] = cp.asnumpy(s0_gpu)[converged]
        failures[output_idx] = False

    cp.cuda.Stream.null.synchronize()
    return r2star, frequency_hz, s0, failures


def _cuda_batch_size():
    """Read an optional CUDA batch size cap from ``TEDANA_CUDA_BATCH_SIZE``."""
    if "TEDANA_CUDA_BATCH_SIZE" not in os.environ:
        return None
    batch_size = int(os.environ["TEDANA_CUDA_BATCH_SIZE"])
    if batch_size < 1:
        raise ValueError("TEDANA_CUDA_BATCH_SIZE must be a positive integer")
    return batch_size


def fit_complex_monoexponential(
    data_cat, echo_times, adaptive_mask, report=True, n_threads=1, backend="cpu"
):
    """Fit a complex monoexponential decay model per voxel across all timepoints.

    Echoes and timepoints are flattened into a single fit per voxel (the
    complex analog of the ``fittype="curvefit"``/``fitmode="all"`` scheme),
    using the adaptive mask to choose how many echoes each voxel uses.

    Parameters
    ----------
    data_cat : (Md x E x T) :obj:`numpy.ndarray`
        Complex multi-echo data. Md is samples in the denoising mask.
    echo_times : (E,) array_like
        Echo times in seconds.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Number of good echoes per voxel. See ``make_adaptive_mask``.
    report : bool, optional
        Whether to log a description of this step. Default is True.
    n_threads : int, optional
        Number of threads. If None or <= 0, uses all CPU cores.
    backend : {"cpu", "cuda", "auto"}, optional
        Compute backend. ``cuda`` fits voxels in batches on the GPU and refits
        any that do not converge with SciPy. Default is ``cpu``.

    Returns
    -------
    result : dict
        ``(Md,)`` arrays ``t2s``, ``s0`` (magnitude), ``r2star``,
        ``frequency_hz``, ``phase0``, and boolean ``failures``.
    """
    backend = utils._resolve_backend(backend)
    if backend == "cuda":
        cuda_batch_size = _cuda_batch_size()
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    if report:
        RepLGR.info(
            "A complex monoexponential model was fit to the magnitude and "
            "phase data at each voxel using nonlinear least squares, jointly "
            "estimating T2*, S0, off-resonance frequency, and initial phase."
        )
    echo_times = np.asarray(echo_times, dtype=float)
    n_samp, _, n_vols = data_cat.shape

    echos_to_run = np.unique(adaptive_mask)
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    # R2* is bounded below by MIN_R2STAR so a (near-)flat decay cannot drive
    # T2* = 1/R2* toward infinity.
    lower_bounds = np.array([-np.inf, MIN_R2STAR, -np.inf, -np.inf])
    upper_bounds = np.array([np.inf, np.inf, np.inf, np.inf])

    r2star = np.zeros(n_samp)
    s0_mag = np.zeros(n_samp)
    frequency_hz = np.zeros(n_samp)
    phase0 = np.zeros(n_samp)
    failures = np.zeros(n_samp, dtype=bool)

    for echo_num in echos_to_run:
        if echo_num == 2:
            voxel_idx = np.where(adaptive_mask <= echo_num)[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        data_2d = data_cat[:, :echo_num, :].reshape(n_samp, -1)
        echo_times_1d = np.repeat(echo_times[:echo_num], n_vols)

        if backend == "cuda":
            r2_gpu, freq_gpu, s0_gpu, gpu_failures = _fit_complex_groups_cuda(
                data_2d[voxel_idx, None, :], echo_times_1d, batch_size=cuda_batch_size
            )
            gpu_voxels = voxel_idx[~gpu_failures]
            r2star[gpu_voxels] = r2_gpu[~gpu_failures]
            frequency_hz[gpu_voxels] = freq_gpu[~gpu_failures]
            s0_mag[gpu_voxels] = np.abs(s0_gpu[~gpu_failures, 0])
            phase0[gpu_voxels] = np.angle(s0_gpu[~gpu_failures, 0])
            voxel_idx = voxel_idx[gpu_failures]
            if voxel_idx.size:
                LGR.debug("Refitting %d CUDA non-convergent voxel(s) with SciPy.", voxel_idx.size)

        results = Parallel(n_jobs=n_threads)(
            delayed(_fit_complex_decay_1d)(
                data_2d[voxel],
                echo_times_1d,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            for voxel in tqdm(
                voxel_idx, desc=f"{echo_num}-echo complex NLLS", disable=backend == "cuda"
            )
        )

        for voxel, res in zip(voxel_idx, results):
            if res is None:
                failures[voxel] = True
                continue
            r2star[voxel] = res["r2star"]
            s0_mag[voxel] = np.abs(res["s0"])
            frequency_hz[voxel] = res["frequency_hz"]
            phase0[voxel] = res["phase0"]
            failures[voxel] = not res["success"]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t2s = np.where(r2star > 0, 1.0 / r2star, np.inf)

    return {
        "t2s": t2s,
        "s0": s0_mag,
        "r2star": r2star,
        "frequency_hz": frequency_hz,
        "phase0": phase0,
        "failures": failures,
    }


def _solve_complex_s0(signal, echo_times, r2star, frequency_hz):
    """Analytically solve per-volume complex S0 for fixed R2*/frequency.

    Parameters
    ----------
    signal : (T, E) array_like
        Complex data for one voxel; volumes on axis 0, echoes on axis 1.
    echo_times : (E,) array_like
        Echo times in seconds.
    r2star : float
        Shared R2* in s^-1.
    frequency_hz : float
        Shared off-resonance frequency in Hz.

    Returns
    -------
    s0 : (T,) :obj:`numpy.ndarray`
        Complex S0 per volume. NaN for volumes with fewer than 2 finite echoes.
    """
    signal = np.asarray(signal)
    echo_times = np.asarray(echo_times, dtype=float)
    n_vols = signal.shape[0]
    with np.errstate(over="ignore", invalid="ignore"):
        decay = np.exp((-r2star + 1j * 2.0 * np.pi * frequency_hz) * echo_times)
    s0 = np.full(n_vols, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.isfinite(signal.real) & np.isfinite(signal.imag)
    for vol in range(n_vols):
        echo_mask = valid[vol]
        if int(echo_mask.sum()) < 2:
            continue
        basis = decay[echo_mask]
        denom = np.sum(np.abs(basis) ** 2)
        if denom <= 0 or not np.isfinite(denom):
            continue
        s0[vol] = np.sum(np.conj(basis) * signal[vol, echo_mask]) / denom
    return s0


def _fit_complex_decay_joint(
    signal,
    echo_times,
    *,
    min_r2star=0.0,
    max_r2star=np.inf,
    max_frequency_hz=np.inf,
    max_nfev=None,
):
    """Fit one voxel with shared R2*/frequency and per-volume complex S0.

    Per-volume complex S0 is linear given fixed R2*/frequency and is solved
    analytically inside the residual, keeping the optimizer to two parameters.

    Parameters
    ----------
    signal : (T, E) array_like
        Complex data for one voxel; volumes on axis 0, echoes on axis 1.
    echo_times : (E,) array_like
        Echo times in seconds.
    min_r2star : float, optional
        Lower bound for R2* in s^-1. Default is 0.0.
    max_r2star : float, optional
        Upper bound for R2* in s^-1. Default is inf.
    max_frequency_hz : float, optional
        Symmetric bound for off-resonance in Hz. Default is inf. Frequency is
        also limited to one alias period around the initial estimate (see
        :func:`_frequency_half_range`).
    max_nfev : int or None, optional
        Maximum function evaluations.

    Returns
    -------
    result : dict or None
        Scalar ``r2star``, ``frequency_hz``, ``cost``, ``nfev``, ``success``
        and ``(T,)`` arrays ``s0`` (complex) and ``phase0``. ``None`` if no
        volume has >= 2 finite echoes or optimization fails. If the fit is
        degenerate (see :func:`_is_degenerate_decay`), the initial estimate is
        returned with ``success=False``.
    """
    signal = np.asarray(signal)
    echo_times = np.asarray(echo_times, dtype=float)
    valid = np.isfinite(signal.real) & np.isfinite(signal.imag)
    valid_vols = np.flatnonzero(valid.sum(axis=1) >= 2)
    if valid_vols.size == 0:
        return None

    initial = np.asarray(
        [
            _initial_complex_decay_params(signal[vol, valid[vol]], echo_times[valid[vol]])
            for vol in valid_vols
        ]
    )
    lower_bounds = np.array([min_r2star, -max_frequency_hz])
    upper_bounds = np.array([max_r2star, max_frequency_hz])
    x0 = np.array([float(np.nanmedian(initial[:, 1])), float(np.nanmedian(initial[:, 2]))])
    x0 = np.minimum(np.maximum(x0, lower_bounds), upper_bounds)
    half_range = _frequency_half_range(echo_times)
    lower_bounds[1] = max(lower_bounds[1], x0[1] - half_range)
    upper_bounds[1] = min(upper_bounds[1], x0[1] + half_range)

    def residuals(params):
        r2, freq = params
        with np.errstate(over="ignore", invalid="ignore"):
            s0_est = _solve_complex_s0(signal, echo_times, r2, freq)
            parts = []
            for vol in valid_vols:
                if not np.isfinite(s0_est[vol]):
                    continue
                echo_mask = valid[vol]
                pred = complex_decay_model(echo_times[echo_mask], s0_est[vol], r2, freq)
                residual = pred - signal[vol, echo_mask]
                parts.extend([residual.real, residual.imag])
        return np.concatenate(parts)

    try:
        result = scipy.optimize.least_squares(
            residuals, x0, bounds=(lower_bounds, upper_bounds), max_nfev=max_nfev
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None

    r2star, frequency_hz = result.x
    cost, nfev, success = result.cost, result.nfev, result.success
    if _is_degenerate_decay(r2star, echo_times):
        r2star, frequency_hz = x0
        cost, nfev, success = np.nan, 0, False
    s0 = _solve_complex_s0(signal, echo_times, r2star, frequency_hz)
    return {
        "s0": s0,
        "phase0": np.angle(s0),
        "r2star": float(r2star),
        "frequency_hz": float(frequency_hz),
        "cost": cost,
        "nfev": nfev,
        "success": success,
    }


def fit_complex_decay(
    data, phase, tes, adaptive_mask, fitmode, use_volumes=None, n_threads=1, backend="cpu"
):
    """Estimate complex T2*/S0 maps from magnitude and phase data.

    Parameters
    ----------
    data : (Md x E x T) :obj:`numpy.ndarray`
        Magnitude multi-echo data in the denoising mask.
    phase : (Md x E x T) :obj:`numpy.ndarray`
        Phase multi-echo data in radians, same shape as ``data``.
    tes : (E,) array_like
        Echo times in seconds.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Number of good echoes per voxel.
    fitmode : {"all", "ts", "varys0"}
        ``"all"`` fits one model per voxel across all timepoints. ``"ts"`` fits
        per voxel and timepoint. ``"varys0"`` shares R2*/frequency across
        timepoints with per-volume complex S0.
    use_volumes : (T,) :obj:`numpy.ndarray` of bool or None, optional
        For ``"varys0"`` only: volumes used to estimate the shared
        R2*/frequency. Per-volume S0 is computed for all volumes regardless.
        ``None`` uses all volumes.
    n_threads : int, optional
        Number of threads. If None or <= 0, uses all CPU cores.
    backend : {"cpu", "cuda", "auto"}, optional
        Compute backend. Default is ``cpu``.

    Returns
    -------
    result : dict
        For ``"all"``: ``(Md,)`` arrays ``t2s``, ``s0``, ``r2star``,
        ``frequency_hz``, ``phase0``, ``failures``. For ``"ts"``: same keys as
        ``(Md, T)``. For ``"varys0"``: ``t2s``/``r2star``/``frequency_hz``/
        ``failures`` are ``(Md,)`` and ``s0``/``phase0`` are ``(Md, T)``.
    """
    backend = utils._resolve_backend(backend)
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    data = np.asarray(data)
    phase = np.asarray(phase)
    tes = np.asarray(tes, dtype=float)
    complex_data = data * np.exp(1j * phase)
    n_samp, _, n_vols = complex_data.shape

    if fitmode == "all":
        return fit_complex_monoexponential(
            complex_data, tes, adaptive_mask, n_threads=n_threads, backend=backend
        )

    if fitmode == "ts":
        # Every (voxel, volume) pair is an independent single-volume fit, so
        # flatten them into one set of samples and fit them all at once.
        n_echos = complex_data.shape[1]
        res = fit_complex_monoexponential(
            complex_data.transpose(0, 2, 1).reshape(n_samp * n_vols, n_echos)[:, :, None],
            tes,
            np.repeat(adaptive_mask, n_vols),
            n_threads=n_threads,
            backend=backend,
        )
        return {k: v.reshape(n_samp, n_vols) for k, v in res.items()}

    if fitmode == "varys0":
        if use_volumes is None:
            use_volumes = np.ones(n_vols, dtype=bool)
        else:
            use_volumes = np.asarray(use_volumes, dtype=bool)
        return _fit_complex_decay_varys0(
            complex_data, tes, adaptive_mask, use_volumes, n_threads, backend=backend
        )

    raise ValueError(f"Unknown fitmode option: {fitmode}")


def _fit_complex_decay_varys0(
    complex_data, tes, adaptive_mask, use_volumes, n_threads, backend="cpu"
):
    """Joint-fit driver: shared R2*/frequency, per-volume complex S0.

    Parameters
    ----------
    complex_data : (Md x E x T) :obj:`numpy.ndarray`
        Complex multi-echo data.
    tes : (E,) :obj:`numpy.ndarray`
        Echo times in seconds.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Number of good echoes per voxel.
    use_volumes : (T,) :obj:`numpy.ndarray` of bool
        Volumes used for the shared R2*/frequency estimate.
    n_threads : int
        Number of threads.
    backend : {"cpu", "cuda"}, optional
        Compute backend. Default is ``cpu``.

    Returns
    -------
    result : dict
        ``(Md,)`` ``t2s``, ``r2star``, ``frequency_hz``, ``failures`` and
        ``(Md, T)`` ``s0`` (magnitude) and ``phase0``.
    """
    n_samp, _, n_vols = complex_data.shape
    echos_to_run = np.unique(adaptive_mask)
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    r2star = np.zeros(n_samp)
    frequency_hz = np.zeros(n_samp)
    failures = np.zeros(n_samp, dtype=bool)
    s0_mag = np.zeros((n_samp, n_vols))
    phase0 = np.zeros((n_samp, n_vols))

    def _fit_voxel(voxel, echo_num):
        # (T, E) for the shared fit (used volumes only) and all volumes for S0
        signal_all = complex_data[voxel, :echo_num, :].T
        joint = _fit_complex_decay_joint(
            signal_all[use_volumes], tes[:echo_num], min_r2star=MIN_R2STAR
        )
        if joint is None:
            return voxel, None
        s0_all = _solve_complex_s0(
            signal_all, tes[:echo_num], joint["r2star"], joint["frequency_hz"]
        )
        return voxel, {"joint": joint, "s0_all": s0_all}

    for echo_num in echos_to_run:
        if echo_num == 2:
            voxel_idx = np.where(adaptive_mask <= echo_num)[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        if backend == "cuda":
            r2_gpu, freq_gpu, s0_gpu, gpu_failures = _fit_complex_groups_cuda(
                complex_data[voxel_idx, :echo_num, :].transpose(0, 2, 1),
                tes[:echo_num],
                fit_groups=use_volumes,
                batch_size=_cuda_batch_size(),
            )
            gpu_voxels = voxel_idx[~gpu_failures]
            r2star[gpu_voxels] = r2_gpu[~gpu_failures]
            frequency_hz[gpu_voxels] = freq_gpu[~gpu_failures]
            s0_mag[gpu_voxels, :] = np.abs(s0_gpu[~gpu_failures])
            phase0[gpu_voxels, :] = np.angle(s0_gpu[~gpu_failures])
            voxel_idx = voxel_idx[gpu_failures]
            if voxel_idx.size:
                LGR.debug("Refitting %d CUDA non-convergent voxel(s) with SciPy.", voxel_idx.size)

        results = Parallel(n_jobs=n_threads)(
            delayed(_fit_voxel)(voxel, echo_num)
            for voxel in tqdm(
                voxel_idx, desc=f"{echo_num}-echo varys0 NLLS", disable=backend == "cuda"
            )
        )

        for voxel, res in results:
            if res is None:
                failures[voxel] = True
                continue
            joint = res["joint"]
            r2star[voxel] = joint["r2star"]
            frequency_hz[voxel] = joint["frequency_hz"]
            failures[voxel] = not joint["success"]
            s0_all = res["s0_all"]
            s0_mag[voxel, :] = np.abs(s0_all)
            phase0[voxel, :] = np.angle(s0_all)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t2s = np.where(r2star > 0, 1.0 / r2star, np.inf)

    return {
        "t2s": t2s,
        "r2star": r2star,
        "frequency_hz": frequency_hz,
        "s0": s0_mag,
        "phase0": phase0,
        "failures": failures,
    }


def fit_monoexponential(
    data_cat,
    echo_times,
    adaptive_mask,
    report=True,
    n_threads=1,
    backend="cpu",
    t2s_initial=None,
    s0_initial=None,
    voxels_to_refit=None,
):
    """Fit monoexponential decay model with nonlinear curve-fitting.

    Parameters
    ----------
    data_cat : (Md x E x T) :obj:`numpy.ndarray`
        Multi-echo data. Md is samples in denoising mask, E is echoes, and T is timepoints.
    echo_times : (E,) array_like
        Echo times in seconds.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    report : bool, optional
        Whether to log a description of this step or not. Default is True.
    n_threads : int, optional
        Number of threads to use. Default is 1. If None or <= 0, uses the number
        of available CPU cores.
    backend : {"cpu", "cuda", "auto"}, optional
        Compute backend for nonlinear fitting. ``cpu`` uses SciPy, ``cuda``
        requires CuPy and a CUDA device, and ``auto`` falls back to CPU if CUDA
        is unavailable. Default is ``cpu``.
    t2s_initial : (Md,) :obj:`numpy.ndarray` or None, optional
        Initial T2* estimates to use instead of log-linear fit. Default is None.
    s0_initial : (Md,) :obj:`numpy.ndarray` or None, optional
        Initial S0 estimates to use instead of log-linear fit. Default is None.
    voxels_to_refit : (Md,) :obj:`numpy.ndarray` or None, optional
        Boolean array indicating which voxels to refit. If None, all voxels are fitted.
        Voxels not in this mask retain their initial values and are not marked as failures.

    Returns
    -------
    t2s, s0 : (Md,) :obj:`numpy.ndarray`
        T2* and S0 estimate maps.
        These maps include T2*/S0 estimates for all voxels with adaptive mask >= 1.
        For voxels with adaptive mask == 1, the T2*/S0 estimates are from the first two echoes.
        These voxels should be replaced with zeros in the full T2*/S0 maps.
    failures : (Md,) :obj:`numpy.ndarray`
        Boolean array indicating samples that failed to fit the model.
    t2s_var : (Md,) :obj:`numpy.ndarray`
        Variance of the T2* estimates.
    s0_var : (Md,) :obj:`numpy.ndarray`
        Variance of the S0 estimates.
    t2s_s0_covar : (Md,) :obj:`numpy.ndarray`
        Covariance of the T2* and S0 estimates.

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
        parameter.

    Notes
    -----
    This method is slower, but more accurate, than the log-linear approach.
    """
    backend = utils._resolve_backend(backend)
    if backend == "cuda":
        cuda_batch_size = int(os.getenv("TEDANA_CUDA_BATCH_SIZE", "4096"))
        if cuda_batch_size < 1:
            raise ValueError("TEDANA_CUDA_BATCH_SIZE must be a positive integer")
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    if report:
        RepLGR.info(
            "A monoexponential model was fit to the data at each voxel "
            "using nonlinear model fitting in order to estimate T2* and S0 "
            "maps, using T2*/S0 estimates from a log-linear fit as "
            "initial values. For each voxel, the value from the adaptive "
            "mask was used to determine which echoes would be used to "
            "estimate T2* and S0. In cases of model fit failure, T2*/S0 "
            "estimates from the log-linear fit were retained instead."
        )
    n_samp, _, n_vols = data_cat.shape

    # Currently unused
    # fit_data = np.mean(data_cat, axis=2)
    # fit_sigma = np.std(data_cat, axis=2)

    if t2s_initial is not None and s0_initial is not None:
        t2s_init = t2s_initial.copy()
        s0_init = s0_initial.copy()
    else:
        t2s_init, s0_init = fit_loglinear(
            data_cat=data_cat,
            echo_times=echo_times,
            adaptive_mask=adaptive_mask,
            report=False,
        )

    echos_to_run = np.unique(adaptive_mask)
    # When there is one good echo, use two
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    t2s_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    failures_asc_maps = np.zeros([n_samp, len(echos_to_run)], dtype=bool)
    t2s_var_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_var_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    t2s_s0_covar_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    echo_masks = np.zeros([n_samp, len(echos_to_run)], dtype=bool)

    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
            # Use the first two echoes for cases where there are
            # either one or two good echoes
            voxel_idx = np.where(adaptive_mask <= echo_num)[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        # Create echo masks to assign values to limited vs full maps later
        echo_mask = np.squeeze(echo_masks[..., i_echo])
        echo_mask[adaptive_mask == echo_num] = True
        echo_masks[..., i_echo] = echo_mask

        # Filter to only voxels that need refitting, if specified
        if voxels_to_refit is not None:
            voxel_idx = np.intersect1d(voxel_idx, np.where(voxels_to_refit)[0])

        if len(voxel_idx) == 0:
            t2s_asc_maps[:, i_echo] = t2s_init
            s0_asc_maps[:, i_echo] = s0_init
            continue

        data_2d = data_cat[:, :echo_num, :].reshape(len(data_cat), -1).T
        echo_times_1d = np.repeat(echo_times[:echo_num], n_vols)

        # Perform a monoexponential fit using log-linear estimates as initial values.
        if backend == "cuda":
            (
                fitted_s0,
                fitted_t2s,
                failures,
                fitted_s0_var,
                fitted_t2s_var,
                fitted_s0_t2s_covar,
            ) = _fit_voxels_cuda(
                data_2d,
                echo_times_1d,
                voxel_idx,
                s0_init,
                t2s_init,
                batch_size=cuda_batch_size,
            )
            successful_voxels = voxel_idx[~failures]
            s0_init[successful_voxels] = fitted_s0[~failures]
            t2s_init[successful_voxels] = fitted_t2s[~failures]
            s0_var_asc_maps[successful_voxels, i_echo] = fitted_s0_var[~failures]
            t2s_var_asc_maps[successful_voxels, i_echo] = fitted_t2s_var[~failures]
            t2s_s0_covar_asc_maps[successful_voxels, i_echo] = fitted_s0_t2s_covar[~failures]
            fallback_voxels = voxel_idx[failures]
            if len(fallback_voxels):
                LGR.debug(
                    "Refitting %d CUDA non-convergent voxel(s) with SciPy.", len(fallback_voxels)
                )
                results = Parallel(n_jobs=n_threads)(
                    delayed(_fit_single_voxel)(
                        voxel=voxel,
                        echo_times_1d=echo_times_1d,
                        data_column=data_2d[:, voxel],
                        s0_init=s0_init[voxel],
                        t2s_init=t2s_init[voxel],
                        bounds=((np.min(data_2d[:, voxel]), 0), (np.inf, np.inf)),
                    )
                    for voxel in fallback_voxels
                )
            else:
                results = []
            fail_count = 0
        else:
            results = Parallel(n_jobs=n_threads)(
                delayed(_fit_single_voxel)(
                    voxel=voxel,
                    echo_times_1d=echo_times_1d,
                    data_column=data_2d[:, voxel],
                    s0_init=s0_init[voxel],
                    t2s_init=t2s_init[voxel],
                    bounds=((np.min(data_2d[:, voxel]), 0), (np.inf, np.inf)),
                )
                for voxel in tqdm(voxel_idx, desc=f"{echo_num}-echo monoexponential")
            )
            fail_count = 0

        for (
            voxel,
            s0_voxel,
            t2s_voxel,
            failure,
            s0_var_voxel,
            t2s_var_voxel,
            t2s_s0_covar_voxel,
        ) in results:
            if failure:
                failures_asc_maps[voxel, i_echo] = True
                fail_count += 1
            else:
                s0_init[voxel] = s0_voxel
                t2s_init[voxel] = t2s_voxel
                s0_var_asc_maps[voxel, i_echo] = s0_var_voxel
                t2s_var_asc_maps[voxel, i_echo] = t2s_var_voxel
                t2s_s0_covar_asc_maps[voxel, i_echo] = t2s_s0_covar_voxel

        if fail_count:
            fail_percent = 100 * fail_count / len(voxel_idx)
            LGR.debug(
                f"With {echo_num} echoes, monoexponential fit failed on "
                f"{fail_count}/{len(voxel_idx)} ({fail_percent:.2f}%) voxel(s), "
                "used log linear estimate instead"
            )

        t2s_asc_maps[:, i_echo] = t2s_init
        s0_asc_maps[:, i_echo] = s0_init

    # create full T2* and S0 maps
    s0 = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask > 1)
    t2s = utils.unmask(t2s_asc_maps[echo_masks], adaptive_mask > 1)
    failures = utils.unmask(failures_asc_maps[echo_masks], adaptive_mask > 1)
    s0_var = utils.unmask(s0_var_asc_maps[echo_masks], adaptive_mask > 1)
    t2s_var = utils.unmask(t2s_var_asc_maps[echo_masks], adaptive_mask > 1)
    t2s_s0_covar = utils.unmask(t2s_s0_covar_asc_maps[echo_masks], adaptive_mask > 1)

    # Fill voxels with adaptive mask value of 1 with values from first two echoes
    one_echo = adaptive_mask == 1
    s0[one_echo] = s0_asc_maps[one_echo, 0]
    t2s[one_echo] = t2s_asc_maps[one_echo, 0]
    s0_var[one_echo] = s0_var_asc_maps[one_echo, 0]
    t2s_var[one_echo] = t2s_var_asc_maps[one_echo, 0]
    t2s_s0_covar[one_echo] = t2s_s0_covar_asc_maps[one_echo, 0]

    return t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar


def fit_loglinear(data_cat, echo_times, adaptive_mask, report=True):
    """Fit monoexponential decay model with log-linear regression.

    The monoexponential decay function is fitted to all values for a given
    voxel across TRs, per TE, to estimate voxel-wise :math:`S_0` and :math:`T_2^*`.
    At a given voxel, only those echoes with "good signal", as indicated by the
    value of the voxel in the adaptive mask, are used.
    Therefore, for a voxel with an adaptive mask value of five, the first five
    echoes would be used to estimate T2* and S0.

    Parameters
    ----------
    data_cat : (Md x E x T) :obj:`numpy.ndarray`
        Multi-echo data. Md is samples in denoising mask, E is echoes, and T is timepoints.
    echo_times : (E,) array_like
        Echo times in seconds.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    report : :obj:`bool`, optional
        Whether to log a description of this step or not. Default is True.

    Returns
    -------
    t2s, s0 : (Md,) :obj:`numpy.ndarray`
        "Full" T2* and S0 maps without floors or ceilings applied.
        This includes T2* and S0 estimates for all voxels with adaptive mask >= 1.
        Voxels with adaptive mask == 1 have T2* and S0 estimates from the first two echoes.

    Notes
    -----
    The approach used in this function involves transforming the raw signal values
    (:math:`log(|data| + 1)`) and then fitting a line to the transformed data using
    ordinary least squares.
    This results in two parameter estimates: one for the slope  and one for the intercept.
    The slope estimate is inverted (i.e., 1 / slope) to get  :math:`T_2^*`,
    while the intercept estimate is exponentiated (i.e., e^intercept) to get :math:`S_0`.

    This method is faster, but less accurate, than the nonlinear approach.
    """
    if report:
        RepLGR.info(
            "A monoexponential model was fit to the data at each voxel "
            "using log-linear regression in order to estimate T2* and S0 "
            "maps. For each voxel, the value from the adaptive mask was "
            "used to determine which echoes would be used to estimate T2* "
            "and S0."
        )
    n_samp, _, n_vols = data_cat.shape

    echos_to_run = np.unique(adaptive_mask)
    # When there is one good echo, use two
    if 1 in echos_to_run:
        echos_to_run = np.sort(np.unique(np.append(echos_to_run, 2)))
    echos_to_run = echos_to_run[echos_to_run >= 2]

    t2s_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    s0_asc_maps = np.zeros([n_samp, len(echos_to_run)])
    echo_masks = np.zeros([n_samp, len(echos_to_run)], dtype=bool)

    for i_echo, echo_num in enumerate(echos_to_run):
        if echo_num == 2:
            # Use the first two echoes for cases where there are
            # either one or two good echoes
            voxel_idx = np.where(np.logical_and(adaptive_mask > 0, adaptive_mask <= echo_num))[0]
        else:
            voxel_idx = np.where(adaptive_mask == echo_num)[0]

        # Create echo masks to assign values to limited vs full maps later
        echo_mask = np.squeeze(echo_masks[..., i_echo])
        echo_mask[adaptive_mask == echo_num] = True
        echo_masks[..., i_echo] = echo_mask

        # perform log linear fit of echo times against MR signal
        # make DV matrix: samples x (time series * echos)
        data_2d = data_cat[voxel_idx, :echo_num, :].reshape(len(voxel_idx), -1).T
        log_data = np.log(np.abs(data_2d) + 1)

        # make IV matrix: intercept/TEs x (time series * echos)
        x = np.column_stack([np.ones(echo_num), [-te for te in echo_times[:echo_num]]])
        iv_arr = np.repeat(x, n_vols, axis=0)

        # Log-linear fit
        betas = np.linalg.lstsq(iv_arr, log_data, rcond=None)[0]
        t2s = 1.0 / betas[1, :].T
        s0 = np.exp(betas[0, :]).T

        t2s_asc_maps[voxel_idx, i_echo] = t2s
        s0_asc_maps[voxel_idx, i_echo] = s0

    # create full T2* and S0 maps with S0 estimation errors
    t2s = utils.unmask(t2s_asc_maps[echo_masks], adaptive_mask > 1)
    s0 = utils.unmask(s0_asc_maps[echo_masks], adaptive_mask > 1)
    t2s[adaptive_mask == 1] = t2s_asc_maps[adaptive_mask == 1, 0]
    s0[adaptive_mask == 1] = s0_asc_maps[adaptive_mask == 1, 0]

    return t2s, s0


def fit_decay(
    data,
    tes,
    adaptive_mask,
    fittype,
    report=True,
    n_threads=1,
    backend="cpu",
    t2s_initial=None,
    s0_initial=None,
    voxels_to_refit=None,
):
    """Fit voxel-wise monoexponential decay models to ``data``.

    Parameters
    ----------
    data : (Md x E [x T]) array_like
        Multi-echo data array, where `M` is samples in denoising mask, `E` is echos,
        and `T` is time.
    tes : (E,) :obj:`list`
        Echo times in seconds.
    adaptive_mask : (Md,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    fittype : {loglin, curvefit}
        The type of model fit to use
    report : bool, optional
        Whether to log a description of this step or not. Default is True.
    n_threads : int, optional
        Number of threads to use. Default is 1. If None or <= 0, uses the number
        of available CPU cores.
    backend : {"cpu", "cuda", "auto"}, optional
        Compute backend for curve fitting. Default is ``cpu``.
    t2s_initial : (Md,) :obj:`numpy.ndarray` or None, optional
        Initial T2* estimates for curvefit. If None, log-linear estimates are used.
    s0_initial : (Md,) :obj:`numpy.ndarray` or None, optional
        Initial S0 estimates for curvefit. If None, log-linear estimates are used.
    voxels_to_refit : (Md,) :obj:`numpy.ndarray` or None, optional
        Boolean array indicating which voxels to refit. If None, all voxels are fitted.

    Returns
    -------
    t2s : (Md,) :obj:`numpy.ndarray`
        "Full" T2* map without floors or ceilings applied.
        This includes T2* estimates for all voxels with adaptive mask >= 1.
        Voxels with adaptive mask == 1 have T2* estimates from the first two echoes.
    s0 : (Md,) :obj:`numpy.ndarray`
        "Full" S0 map without floors or ceilings applied.
        This includes S0 estimates for all voxels with adaptive mask >= 1.
        Voxels with adaptive mask == 1 have S0 estimates from the first two echoes.
    failures : (Md,) :obj:`numpy.ndarray` or None
        Boolean array indicating samples that failed to fit the model.
        None if fittype is not "curvefit".
    t2s_var : (Md,) :obj:`numpy.ndarray` or None
        Variance of the T2* estimates.
        None if fittype is not "curvefit".
    s0_var : (Md,) :obj:`numpy.ndarray` or None
        Variance of the S0 estimates.
        None if fittype is not "curvefit".
    t2s_s0_covar : (Md,) :obj:`numpy.ndarray` or None
        Covariance of the T2* and S0 estimates.
        None if fittype is not "curvefit".

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
                                              parameter.
    """
    backend = utils._resolve_backend(backend)
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    if data.shape[1] != len(tes):
        raise ValueError(
            f"Second dimension of data ({data.shape[1]}) does not match number "
            f"of echoes provided (tes; {len(tes)})"
        )
    elif not (data.shape[0] == adaptive_mask.shape[0]):
        raise ValueError(
            f"First dimensions (number of samples) of data ({data.shape[0]}), "
            f"and adaptive_mask ({adaptive_mask.shape[0]}) do not match"
        )

    if data.ndim == 2:
        data = data[:, :, None]

    if fittype == "loglin":
        failures, t2s_var, s0_var, t2s_s0_covar = None, None, None, None
        t2s, s0 = fit_loglinear(
            data_cat=data,
            echo_times=tes,
            adaptive_mask=adaptive_mask,
            report=report,
        )
    elif fittype == "curvefit":
        t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar = fit_monoexponential(
            data_cat=data,
            echo_times=tes,
            adaptive_mask=adaptive_mask,
            report=report,
            n_threads=n_threads,
            backend=backend,
            t2s_initial=t2s_initial,
            s0_initial=s0_initial,
            voxels_to_refit=voxels_to_refit,
        )
    else:
        raise ValueError(f"Unknown fittype option: {fittype}")

    return t2s, s0, failures, t2s_var, s0_var, t2s_s0_covar


def fit_decay_ts(
    data,
    tes,
    adaptive_mask,
    fittype,
    n_threads=1,
    backend="cpu",
    t2s_initial=None,
    s0_initial=None,
    voxels_to_refit=None,
):
    """Fit voxel- and timepoint-wise monoexponential decay models to ``data``.

    Parameters
    ----------
    data : (Md x E x T) array_like
        Multi-echo data array, where `Md` is samples in denoising mask, `E` is echos,
        and `T` is time.
    tes : (E,) :obj:`list`
        Echo times in seconds
    adaptive_mask : (Md,) array_like
        Array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    fittype : :obj: `str`
        The type of model fit to use
    n_threads : int, optional
        Number of threads to use. Default is 1. If None or <= 0, uses the number
        of available CPU cores.
    backend : {"cpu", "cuda", "auto"}, optional
        Compute backend for curve fitting. Default is ``cpu``.
    t2s_initial : (Md x T) :obj:`numpy.ndarray` or None, optional
        Initial T2* estimates for curvefit. If None, log-linear estimates are used.
    s0_initial : (Md x T) :obj:`numpy.ndarray` or None, optional
        Initial S0 estimates for curvefit. If None, log-linear estimates are used.
    voxels_to_refit : (Md x T) :obj:`numpy.ndarray` or None, optional
        Boolean array indicating which voxels to refit per timepoint.
        If None, all voxels are fitted.

    Returns
    -------
    t2s : (Md x T) :obj:`numpy.ndarray`
        Limited T2* map. The limited map only keeps the T2* values for data
        where there are at least two echos with good signal.
    s0 : (Md x T) :obj:`numpy.ndarray`
        Limited S0 map.  The limited map only keeps the S0 values for data
        where there are at least two echos with good signal.
    failures : (Md x T) :obj:`numpy.ndarray` or None
        Boolean array indicating samples that failed to fit the model.
        None if fittype is not "curvefit".
    t2s_var : (Md x T) :obj:`numpy.ndarray` or None
        Variance of the T2* estimates.
        None if fittype is not "curvefit".
    s0_var : (Md x T) :obj:`numpy.ndarray` or None
        Variance of the S0 estimates.
        None if fittype is not "curvefit".
    t2s_s0_covar : (Md x T) :obj:`numpy.ndarray` or None
        Covariance of the T2* and S0 estimates.
        None if fittype is not "curvefit".

    See Also
    --------
    :func:`tedana.utils.make_adaptive_mask` : The function used to create the ``adaptive_mask``
        parameter.
    """
    backend = utils._resolve_backend(backend)
    if n_threads is None or n_threads <= 0:
        n_threads = os.cpu_count() or 1
    n_samples, _, n_vols = data.shape
    tes = np.array(tes)

    t2s = np.zeros([n_samples, n_vols])
    s0 = np.zeros([n_samples, n_vols])
    failures, t2s_var, s0_var, t2s_s0_covar = None, None, None, None
    if fittype == "curvefit":
        failures = np.zeros([n_samples, n_vols], dtype=bool)
        t2s_var = np.zeros([n_samples, n_vols])
        s0_var = np.zeros([n_samples, n_vols])
        t2s_s0_covar = np.zeros([n_samples, n_vols])

    report = True
    for vol in range(n_vols):
        t2s_init_vol = t2s_initial[:, vol] if t2s_initial is not None else None
        s0_init_vol = s0_initial[:, vol] if s0_initial is not None else None
        refit_vol = voxels_to_refit[:, vol] if voxels_to_refit is not None else None
        t2s_vol, s0_vol, failures_vol, t2s_var_vol, s0_var_vol, t2s_s0_covar_vol = fit_decay(
            data=data[:, :, vol][:, :, None],
            tes=tes,
            adaptive_mask=adaptive_mask,
            fittype=fittype,
            report=report,
            n_threads=n_threads,
            backend=backend,
            t2s_initial=t2s_init_vol,
            s0_initial=s0_init_vol,
            voxels_to_refit=refit_vol,
        )
        t2s[:, vol] = t2s_vol
        s0[:, vol] = s0_vol
        if fittype == "curvefit":
            failures[:, vol] = failures_vol
            t2s_var[:, vol] = t2s_var_vol
            s0_var[:, vol] = s0_var_vol
            t2s_s0_covar[:, vol] = t2s_s0_covar_vol

        report = False

    return (
        t2s,
        s0,
        failures,
        t2s_var,
        s0_var,
        t2s_s0_covar,
    )


def modify_t2s_s0_maps(t2s, s0, adaptive_mask, tes):
    """Modify T2* and S0 maps to include estimates for voxels with adaptive mask == 1.

    Parameters
    ----------
    t2s : (Md,) :obj:`numpy.ndarray`
        "Full" T2* map.
        This includes T2* estimates for all voxels with adaptive mask >= 1.
    s0 : (Md,) :obj:`numpy.ndarray`
        "Full" S0 map.
        This includes S0 estimates for all voxels with adaptive mask >= 1.
    adaptive_mask : (Md,) :obj:`numpy.ndarray`
        Adaptive mask array where each value indicates the number of echoes with good signal
        for that voxel. This mask may be thresholded; for example, with values
        less than 3 set to 0.
        For more information on thresholding, see `make_adaptive_mask`.
    tes : (E,) :obj:`list`
        Echo times in seconds.

    Returns
    -------
    t2s : (Md,) :obj:`numpy.ndarray`
        "Full" T2* map with floors and ceilings applied.
        This includes T2* estimates for all voxels with adaptive mask >= 1.
    s0 : (Md,) :obj:`numpy.ndarray`
        "Full" S0 map with floors and ceilings applied.
        This includes S0 estimates for all voxels with adaptive mask >= 1.
    t2s_limited : (Md,) :obj:`numpy.ndarray`
        "Limited" T2* map.
        This includes T2* estimates for all voxels with adaptive mask > 1.
        Voxels with adaptive mask == 1 are set to 0.
    s0_limited : (Md,) :obj:`numpy.ndarray`
        "Limited" S0 map.
        This includes S0 estimates for all voxels with adaptive mask > 1.
        Voxels with adaptive mask == 1 are set to 0.

    Notes
    -----
    This function replaces infinite values in the :math:`T_2^*` map with 0.5 s and
    :math:`T_2^*` values less than or equal to zero with 0.001 s.
    Additionally, very small :math:`T_2^*` values above zero are replaced with a floor
    value to prevent zero-division errors later on in the workflow.
    It also replaces NaN, infinite, and float32-unrepresentable values in the :math:`S_0` map
    with 0.
    """
    # Apply floors and ceilings to the T2* and S0 maps
    t2s[np.isinf(t2s)] = 0.5  # why 0.5 s?
    t2s[t2s <= 0] = 0.001  # set negative values to a small positive value
    t2s = _apply_t2s_floor(t2s, tes)
    # Zero out NaN, inf, and values too large to represent in float32
    s0_f32_max = np.finfo(np.float32).max
    s0[~np.isfinite(s0) | (np.abs(s0) > s0_f32_max)] = 0.0

    t2s_limited = t2s.copy()
    s0_limited = s0.copy()
    t2s_limited[adaptive_mask == 1] = 0
    s0_limited[adaptive_mask == 1] = 0

    # set a hard cap for the T2* map
    # anything that is 10x higher than the 99.5 %ile will be reset to 99.5 %ile
    cap_t2s = stats.scoreatpercentile(t2s_limited.flatten(), 99.5, interpolation_method="lower")
    LGR.debug(f"Setting cap on T2* map at {cap_t2s * 10:.5f}")
    # Apply the cap to both the full and limited maps. Without it, blown-up
    # finite estimates from a poorly constrained fit leak into the full map (and
    # overflow to +Inf when saved as float32).
    t2s[t2s > cap_t2s * 10] = cap_t2s
    t2s_limited[t2s_limited > cap_t2s * 10] = cap_t2s

    return t2s, s0, t2s_limited, s0_limited


def rmse_of_fit_decay_ts(
    *,
    data: np.ndarray,
    tes: List[float],
    adaptive_mask: np.ndarray,
    t2s: np.ndarray,
    s0: np.ndarray,
    fitmode: Literal["all", "ts", "varys0"],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate model fit of voxel- and timepoint-wise monoexponential decay models to ``data``.

    Parameters
    ----------
    data : (Mb x E x T) :obj:`numpy.ndarray`
        Multi-echo data array, where `Mb` is samples in base mask, `E` is echos,
        and `T` is time.
    tes : (E,) :obj:`list`
        Echo times.
    adaptive_mask : (Mb,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal for that voxel.
        This mask may be thresholded; for example, with values less than 3 set to 0.
        For more information on thresholding, see :func:`~tedana.utils.make_adaptive_mask`.
    t2s : (Mb [x T]) :obj:`numpy.ndarray`
        Voxel-wise (and possibly volume-wise) T2* estimates from
        :func:`~tedana.decay.fit_decay_ts`.
    s0 : (Mb [x T]) :obj:`numpy.ndarray`
        Voxel-wise (and possibly volume-wise) S0 estimates from :func:`~tedana.decay.fit_decay_ts`.
    fitmode : {"all", "ts", "varys0"}
        Whether the T2* and S0 estimates are volume-wise ("ts", "varys0") or not ("all").

    Returns
    -------
    rmse_map : (Mb,) :obj:`numpy.ndarray`
        Mean root mean squared error of the model fit across all volumes at each voxel.
    rmse_df : :obj:`pandas.DataFrame`
        Each column is the root mean squared error of the model fit at each timepoint.
        Columns are mean, standard deviation, and percentiles across voxels. Column labels are
        "rmse_mean", "rmse_std", "rmse_min", "rmse_percentile02", "rmse_percentile25",
        "rmse_median", "rmse_percentile75", "rmse_percentile98", and "rmse_max"
    """
    n_samples, _, n_vols = data.shape
    tes = np.array(tes)

    rmse = np.full([n_samples, n_vols], np.nan, dtype=np.float32)
    # n_good_echoes interates from 2 through the number of echoes
    #   0 and 1 are excluded because there aren't T2* and S0 estimates
    #   for less than 2 good echoes. 2 echoes will have a bad estimate so consider
    #   how/if we want to distinguish those
    for n_good_echoes in range(2, len(tes) + 1):
        # a boolean mask for voxels with a specific num of good echoes
        use_vox = adaptive_mask == n_good_echoes
        if n_good_echoes == 2:
            # Voxels with a single good echo are fit using the first two echoes
            # (mirroring fit_monoexponential), so include them here rather than
            # leaving their RMSE as NaN (which renders as empty voxels).
            use_vox = use_vox | (adaptive_mask == 1)
        data_echo = data[use_vox, :n_good_echoes, :]
        if fitmode == "all":
            s0_echo = np.tile(s0[use_vox][:, np.newaxis], (1, n_vols))
            t2s_echo = np.tile(t2s[use_vox][:, np.newaxis], (1, n_vols))
        elif fitmode == "ts":
            s0_echo = s0[use_vox, :]
            t2s_echo = t2s[use_vox, :]
        elif fitmode == "varys0":
            s0_echo = s0[use_vox, :]
            t2s_echo = np.tile(t2s[use_vox][:, np.newaxis], (1, n_vols))
        else:
            raise ValueError(f"Unknown fitmode option {fitmode}")

        predicted_data = np.full([use_vox.sum(), n_good_echoes, n_vols], np.nan, dtype=np.float32)
        # Need to loop by echo since monoexponential can take either single vals for s0 and t2star
        #   or a single TE value.
        # We could expand that func, but this is a functional solution
        for echo_num in range(n_good_echoes):
            predicted_data[:, echo_num, :] = monoexponential(
                tes=tes[echo_num],
                s0=s0_echo,
                t2star=t2s_echo,
            )
        rmse[use_vox, :] = np.sqrt(np.mean((data_echo - predicted_data) ** 2, axis=1))

    rmse_sum_map = np.nansum(rmse, axis=1)
    rmse_count_map = np.sum(~np.isnan(rmse), axis=1)
    rmse_map = np.full(rmse_sum_map.shape, np.nan, dtype=rmse.dtype)
    np.divide(rmse_sum_map, rmse_count_map, out=rmse_map, where=rmse_count_map > 0)

    rmse_sum_ts = np.nansum(rmse, axis=0)
    rmse_count_ts = np.sum(~np.isnan(rmse), axis=0)
    rmse_timeseries = np.full(rmse_sum_ts.shape, np.nan, dtype=rmse.dtype)
    np.divide(rmse_sum_ts, rmse_count_ts, out=rmse_timeseries, where=rmse_count_ts > 0)
    rmse_sd_timeseries = np.nanstd(rmse, axis=0)
    rmse_percentiles_timeseries = np.nanpercentile(rmse, [0, 2, 25, 50, 75, 98, 100], axis=0)

    rmse_df = pd.DataFrame(
        columns=[
            "rmse_mean",
            "rmse_std",
            "rmse_min",
            "rmse_percentile02",
            "rmse_percentile25",
            "rmse_median",
            "rmse_percentile75",
            "rmse_percentile98",
            "rmse_max",
        ],
        data=np.column_stack(
            (
                rmse_timeseries,
                rmse_sd_timeseries,
                rmse_percentiles_timeseries.T,
            )
        ),
    )

    return rmse_map, rmse_df


def generate_decay_metrics(
    *,
    t2star,
    s0,
    rmse_map,
    adaptive_mask,
    n_fit_failures=None,
    n_fit_failures_after_interpolation=None,
):
    """Summarize decay-fit quality into a dict for the QC report.

    All array inputs are in base-mask sample space (one entry per base-mask voxel)
    and must be 1D (``(S,)``). This function is only meaningful for scalar,
    per-voxel summaries; time-varying (``fitmode == "ts"``) maps are not supported.

    Parameters
    ----------
    t2star, s0, rmse_map : (S,) array_like
        Full T2*, S0, and fit-RMSE maps in base-mask sample space. Must be 1D.
    adaptive_mask : (S,) array_like
        Integer count of good echoes per base-mask voxel (0 means no good echo).
        Must be 1D.
    n_fit_failures, n_fit_failures_after_interpolation : int or None
        Curve-fit failure counts. ``None`` when curve-fitting was not used, in which
        case the corresponding keys are omitted from the returned dict.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If any of ``adaptive_mask``, ``t2star``, ``s0``, or ``rmse_map`` is not 1D.
        This most often happens when ``fitmode == "ts"`` produces 2D
        (voxels x time) maps, for which a scalar summary is ill-defined.
    """
    adaptive_mask = np.asarray(adaptive_mask)
    for name, arr in (
        ("adaptive_mask", adaptive_mask),
        ("t2star", t2star),
        ("s0", s0),
        ("rmse_map", rmse_map),
    ):
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(
                "generate_decay_metrics expects 1D base-mask-space arrays; "
                f"got shape {arr.shape} for '{name}'."
            )

    fit_mask = adaptive_mask >= 1

    def _finite_mean_median(arr):
        arr = np.asarray(arr, dtype=float)[fit_mask]
        # Exclude zeros: modify_t2s_s0_maps fills NaN S0 values with 0.0, and
        # those placeholders would otherwise bias the mean/median.
        arr = arr[np.isfinite(arr) & (arr != 0)]
        if arr.size == 0:
            return None, None
        return float(np.mean(arr)), float(np.median(arr))

    t2star_mean, t2star_median = _finite_mean_median(t2star)
    s0_mean, s0_median = _finite_mean_median(s0)

    rmse = np.asarray(rmse_map, dtype=float)[fit_mask]
    rmse = rmse[np.isfinite(rmse)]
    rmse_mean = float(np.mean(rmse)) if rmse.size else None
    rmse_median = float(np.median(rmse)) if rmse.size else None

    good_echo_voxel_counts = {
        int(n): int((adaptive_mask == n).sum()) for n in np.unique(adaptive_mask[fit_mask])
    }

    metrics = {
        "t2star_mean": t2star_mean,
        "t2star_median": t2star_median,
        "s0_mean": s0_mean,
        "s0_median": s0_median,
        "rmse_mean": rmse_mean,
        "rmse_median": rmse_median,
        "n_voxels_base_mask": int(adaptive_mask.size),
        "n_voxels_fit_mask": int(fit_mask.sum()),
        "good_echo_voxel_counts": good_echo_voxel_counts,
    }
    if n_fit_failures is not None:
        metrics["n_fit_failures"] = int(n_fit_failures)
    if n_fit_failures_after_interpolation is not None:
        metrics["n_fit_failures_after_interpolation"] = int(n_fit_failures_after_interpolation)

    return metrics


def t2smap_subworkflow(
    data_cat,
    tes,
    fittype,
    fitmode,
    mask_denoise,
    masksum_denoise,
    interpolate_failing_voxels,
    io_generator,
    data_without_excluded_vols=None,
    n_threads=1,
    backend="cpu",
    phase_cat=None,
    use_volumes=None,
):
    """Fit a T2* map to the data.

    Parameters
    ----------
    data_cat : (Mb x E x T) :obj:`numpy.ndarray`
        Multi-echo data array, where `Mb` is samples in base mask, `E` is echos,
        and `T` is time.
    tes : (E,) :obj:`list`
        Echo times.
    fittype : {"curvefit", "loglin", "nlls"}
        The type of model fit to use. ``"nlls"`` fits a complex model and
        requires ``phase_cat``.
    fitmode : {"all", "ts", "varys0"}
        Whether the T2* and S0 estimates are volume-wise ("ts"), not ("all"),
        or share T2* across volumes with volume-wise S0 ("varys0"; nlls only).
    mask_denoise : (Mb,) :obj:`numpy.ndarray`
        Boolean array indicating which voxels to include in the fit.
    masksum_denoise : (Mb,) :obj:`numpy.ndarray`
        Array where each value indicates the number of echoes with good signal for that voxel.
        This mask may be thresholded; for example, with values less than 3 set to 0.
        For more information on thresholding, see :func:`~tedana.utils.make_adaptive_mask`.
    n_threads : int
        Number of threads to use.
    backend : {"cpu", "cuda", "auto"}
        Compute backend for curve fitting.
    interpolate_failing_voxels : bool
        Whether to interpolate failing voxels.
    io_generator : :obj:`tedana.io.IOGenerator`
        The IO generator to use.
    data_without_excluded_vols : (Mb x E x T) :obj:`numpy.ndarray` or None, optional
        Multi-echo data array, where `Mb` is samples in base mask, `E` is echos,
        and `T` is time. If None, the data_cat is used.
    phase_cat : (Mb x E x T) :obj:`numpy.ndarray` or None, optional
        Phase data in radians, matching ``data_cat``. Required for ``fittype="nlls"``.
    use_volumes : (T,) :obj:`numpy.ndarray` of bool or None, optional
        Volumes of ``data_cat`` retained in ``data_without_excluded_vols``.
        Used to select matching phase volumes for ``fittype="nlls"``.
        None means all volumes are used.

    Returns
    -------
    t2s_full : (Mb [x T]) :obj:`numpy.ndarray`
        The full T2* map.
    """
    n_fit_failures = None
    n_fit_failures_after_interpolation = None

    data_for_fit = (
        data_without_excluded_vols if data_without_excluded_vols is not None else data_cat
    )
    data_for_fit = data_for_fit[mask_denoise, ...]
    masksum_masked = masksum_denoise[mask_denoise]
    if fittype == "nlls":
        if phase_cat is None:
            raise ValueError("fittype='nlls' requires phase_cat.")

        if fitmode == "varys0":
            # Per-volume S0 is estimated for every volume, so fit all volumes and
            # let use_volumes select those informing the shared R2*/frequency.
            complex_results = fit_complex_decay(
                data=data_cat[mask_denoise, ...],
                phase=phase_cat[mask_denoise, ...],
                tes=tes,
                adaptive_mask=masksum_masked,
                fitmode=fitmode,
                use_volumes=use_volumes,
                n_threads=n_threads,
                backend=backend,
            )
        else:
            phase_for_fit = phase_cat if use_volumes is None else phase_cat[:, :, use_volumes]
            complex_results = fit_complex_decay(
                data=data_for_fit,
                phase=phase_for_fit[mask_denoise, ...],
                tes=tes,
                adaptive_mask=masksum_masked,
                fitmode=fitmode,
                n_threads=n_threads,
                backend=backend,
            )
        t2s_full = complex_results["t2s"]
        s0_full = complex_results["s0"]
        failures = complex_results["failures"]
        t2s_var = s0_var = t2s_s0_covar = None
        n_fit_failures = int(failures.sum())

        io_generator.save_file(failures.astype(np.uint8), "fit failures img", mask=mask_denoise)
        io_generator.save_file(complex_results["frequency_hz"], "frequency img", mask=mask_denoise)
        io_generator.save_file(
            {
                "Units": "Hz",
                "Description": (
                    "Off-resonance frequency (B0 field map) estimated by the complex "
                    "monoexponential (nlls) T2*/S0 fit. The frequency is only determined to "
                    f"within +/- {_frequency_half_range(tes):.1f} Hz (half the alias period "
                    "of the echo spacing) of its initial estimate, is not spatially unwrapped, "
                    "and includes chemical shift and any echo-time-dependent phase errors."
                ),
            },
            "frequency json",
        )
        io_generator.save_file(complex_results["phase0"], "phase0 img", mask=mask_denoise)
        io_generator.save_file(
            {
                "Units": "rad",
                "Description": (
                    "Phase of the complex S0 (signal extrapolated to TE = 0) from the complex "
                    "monoexponential (nlls) T2*/S0 fit. The magnitude of S0 is in the S0 map."
                ),
            },
            "phase0 json",
        )

        if interpolate_failing_voxels and failures.any():
            LGR.info("Interpolating T2*/S0 estimates for %d failing fit(s).", n_fit_failures)
            phys_coords = utils.mask_to_phys_coords(io_generator.mask, mask_denoise)
            t2s_full = utils.interpolate_masked_values(
                t2s_full,
                failures,
                io_generator.mask,
                mask_denoise,
                phys_coords=phys_coords,
            )
            # In varys0 mode, S0 is (Md x T) while failures are per voxel.
            s0_failures = (
                np.broadcast_to(failures[:, None], s0_full.shape)
                if s0_full.ndim > failures.ndim
                else failures
            )
            s0_full = utils.interpolate_masked_values(
                s0_full,
                s0_failures,
                io_generator.mask,
                mask_denoise,
                phys_coords=phys_coords,
            )
        elif interpolate_failing_voxels:
            LGR.info("No nlls failures found; skipping interpolation.")
    else:
        decay_function = fit_decay if fitmode == "all" else fit_decay_ts
        t2s_full, s0_full, failures, t2s_var, s0_var, t2s_s0_covar = decay_function(
            data=data_for_fit,
            tes=tes,
            adaptive_mask=masksum_masked,
            fittype=fittype,
            n_threads=n_threads,
            backend=backend,
        )

    if fittype == "curvefit":
        # Track the first-pass failures so the saved failures map can encode
        # 1 = failed the first curvefit pass, 2 = failed both passes.
        first_pass_failures = failures.copy()

        if interpolate_failing_voxels and failures.any():
            first_failures = failures.copy()
            n_first_failures = first_failures.sum()

            # Save first-pass variance/covariance for non-refitted voxels
            first_t2s_var = t2s_var.copy()
            first_s0_var = s0_var.copy()
            first_t2s_s0_covar = t2s_s0_covar.copy()

            # Stage 1: Interpolate failing voxels
            LGR.info(
                "Interpolating T2*/S0 estimates for %d failing voxel(s) (stage 1).",
                n_first_failures,
            )
            phys_coords = utils.mask_to_phys_coords(io_generator.mask, mask_denoise)
            t2s_interp = utils.interpolate_masked_values(
                t2s_full,
                first_failures,
                io_generator.mask,
                mask_denoise,
                phys_coords=phys_coords,
            )
            s0_interp = utils.interpolate_masked_values(
                s0_full,
                first_failures,
                io_generator.mask,
                mask_denoise,
                phys_coords=phys_coords,
            )

            # Stage 2: Re-run curvefit on failing voxels using interp. values as initial values
            LGR.info(
                "Re-running curvefit on %d failing voxel(s) using interpolated initial values.",
                n_first_failures,
            )
            (
                t2s_full,
                s0_full,
                failures,
                t2s_var,
                s0_var,
                t2s_s0_covar,
            ) = decay_function(
                data=data_for_fit,
                tes=tes,
                adaptive_mask=masksum_masked,
                fittype=fittype,
                n_threads=n_threads,
                backend=backend,
                t2s_initial=t2s_interp,
                s0_initial=s0_interp,
                voxels_to_refit=first_failures,
            )

            # Merge variance/covariance: keep first-pass values for voxels that were not refitted
            t2s_var[~first_failures] = first_t2s_var[~first_failures]
            s0_var[~first_failures] = first_s0_var[~first_failures]
            t2s_s0_covar[~first_failures] = first_t2s_s0_covar[~first_failures]
            del first_t2s_var, first_s0_var, first_t2s_s0_covar

            n_second_failures = failures.sum()
            if n_second_failures == 0:
                LGR.info(
                    "All %d previously failing voxel(s) converged with interpolated initial "
                    "values.",
                    n_first_failures,
                )
            elif n_second_failures == n_first_failures:
                LGR.warning(
                    "All %d voxel(s) that failed the first curvefit also failed with interpolated "
                    "initial values. Using interpolated initial values did not help.",
                    n_first_failures,
                )
            else:
                LGR.info(
                    "%d of %d previously failing voxel(s) converged with interpolated initial "
                    "values; %d voxel(s) still failing.",
                    n_first_failures - n_second_failures,
                    n_first_failures,
                    n_second_failures,
                )

            # Stage 3: Interpolate any remaining failures
            if failures.any():
                LGR.info(
                    "Interpolating T2*/S0 estimates for %d remaining failing voxel(s) (stage 2).",
                    n_second_failures,
                )
                t2s_full = utils.interpolate_masked_values(
                    t2s_full,
                    failures,
                    io_generator.mask,
                    mask_denoise,
                    phys_coords=phys_coords,
                )
                s0_full = utils.interpolate_masked_values(
                    s0_full,
                    failures,
                    io_generator.mask,
                    mask_denoise,
                    phys_coords=phys_coords,
                )
        elif interpolate_failing_voxels:
            LGR.info("No curvefit failures found; skipping interpolation.")

        # Save the failures map. Voxels that failed the first curvefit pass are
        # encoded as 1; those that also failed the second (interpolation-seeded)
        # pass are encoded as 2. When interpolation did not run, ``failures``
        # still holds the first-pass result, so only add it when a second pass
        # actually occurred.
        failures_map = first_pass_failures.astype(np.uint8)
        if interpolate_failing_voxels and first_pass_failures.any():
            failures_map += failures.astype(np.uint8)

        n_fit_failures = int(first_pass_failures.sum())
        if interpolate_failing_voxels and first_pass_failures.any():
            n_fit_failures_after_interpolation = int(failures.sum())

        io_generator.save_file(failures_map, "fit failures img", mask=mask_denoise)

        if io_generator.verbose:
            # Save fit-quality maps after the two-stage refit so that voxels
            # which converged in the second stage carry their real variance/
            # covariance values instead of the first-pass zeros.
            io_generator.save_file(t2s_var, "t2star variance img", mask=mask_denoise)
            io_generator.save_file(s0_var, "s0 variance img", mask=mask_denoise)
            io_generator.save_file(t2s_s0_covar, "t2star-s0 covariance img", mask=mask_denoise)

    del data_without_excluded_vols

    # Delete unused variables
    del failures, t2s_var, s0_var, t2s_s0_covar

    t2s_full, s0_full, t2s_limited, s0_limited = modify_t2s_s0_maps(
        t2s=t2s_full,
        s0=s0_full,
        adaptive_mask=masksum_masked,
        tes=tes,
    )
    del masksum_masked

    t2s_full = utils.unmask(t2s_full, mask_denoise)
    s0_full = utils.unmask(s0_full, mask_denoise)

    io_generator.save_file(t2s_full, "t2star img")
    io_generator.save_file(s0_full, "s0 img")
    if io_generator.verbose:
        t2s_limited = utils.unmask(t2s_limited, mask_denoise)
        s0_limited = utils.unmask(s0_limited, mask_denoise)
        io_generator.save_file(s0_limited, "limited s0 img")
        io_generator.save_file(t2s_limited, "limited t2star img")

    del s0_limited, t2s_limited

    LGR.info("Calculating model fit quality metrics")
    # Use the full T2*/S0 maps (the same maps used for optimal combination) so
    # that voxels with a single good echo also receive a fit-error estimate
    # instead of NaN.
    rmse_map, rmse_df = rmse_of_fit_decay_ts(
        data=data_cat,
        tes=tes,
        adaptive_mask=masksum_denoise,
        t2s=t2s_full,
        s0=s0_full,
        fitmode=fitmode,
    )
    io_generator.save_file(rmse_map, "rmse img")
    io_generator.save_file(rmse_df, "confounds tsv")

    if fitmode == "all":
        # In fitmode == "ts", the T2*/S0/RMSE maps are time-varying (Mb x T), so a
        # single scalar summary per voxel is ill-defined and decay metrics are
        # simply not written.
        decay_metrics = generate_decay_metrics(
            t2star=t2s_full,
            s0=s0_full,
            rmse_map=rmse_map,
            adaptive_mask=masksum_denoise,
            n_fit_failures=n_fit_failures,
            n_fit_failures_after_interpolation=n_fit_failures_after_interpolation,
        )
        io_generator.save_file(decay_metrics, "decay metrics json")

    return t2s_full
