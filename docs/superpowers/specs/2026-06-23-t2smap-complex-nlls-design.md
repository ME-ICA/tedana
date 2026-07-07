# Design: Complex NLLS T2*/S0 fitting for t2smap

Date: 2026-06-23

## Summary

Add a complex non-linear least-squares (NLLS) decay fit to the `t2smap`
workflow, ported from the `dahnke` subpackage's `fit_complex_decay_nlls`
(in `complex-tedana/dahnke/src/dahnke/correction.py`). The complex model
jointly estimates `R2*`, complex `S0`, off-resonance `frequency_hz`, and
initial phase `phase0` from complex data `magnitude · exp(i·phase)` per voxel
using `scipy.optimize.least_squares`.

Three user-facing additions:

- A new `--fittype nlls` option.
- A new `--phase` argument supplying per-echo phase NIfTI files (radians).
- A new `--fitmode varys0` option (nlls-only) for the shared-R2*/frequency,
  per-volume-complex-S0 joint fit.

The Dahnke through-slice modulation / gradient / slice-profile machinery is
**not** ported in this project; modulation is implicitly 1. Only the core
complex decay estimator is ported. Dahnke correction is intended as a future
addition that will build on this complex NLLS foundation — it is deferred, not
abandoned (see "Future work" below).

## Scope decisions (confirmed)

1. **Core NLLS only** — port the per-voxel complex monoexponential fit; drop
   the modulation/gradient/slice-profile code.
2. **Extra outputs** — write `frequency_hz`, `phase0`, and fit-quality
   (success/cost/nfev) maps.
3. **Fitmodes** — `nlls` supports the standard fitmodes, plus a new `varys0`
   joint-fit mode (see below).
4. `varys0` is **nlls-only** (error if combined with `loglin`/`curvefit`).
5. `varys0` outputs **3D** R2*/T2*/frequency maps and **4D** S0-magnitude
   (and 4D phase0) timeseries.

## The model

`complex_decay_model(te, s0, r2star, frequency_hz)`:

```
signal(te) = s0 * exp((-r2star + i·2π·frequency_hz) · te)
```

where `s0` is complex. Fit parameters are
`[log(|S0|), R2*, frequency_hz, phase0]`, with `s0 = exp(log|S0|) · exp(i·phase0)`.
Residuals are the concatenation of real and imaginary parts of
`(predicted - observed)`. Initial parameters come from a log-linear fit of the
magnitude (slope → R2*, intercept → log|S0|) and an unwrapped-phase linear fit
(slope → frequency_hz, intercept → phase0), ported from
`_initial_complex_decay_params`.

## CLI changes (`tedana/workflows/t2smap.py`)

- `--fittype` choices → `["loglin", "curvefit", "nlls"]`; help text updated to
  describe `nlls` and that it requires `--phase`.
- `--fitmode` choices → `["all", "ts", "varys0"]`; help text updated.
- New `--phase` argument: `nargs="+"`, `metavar="FILE"`, validated via
  `is_valid_file`, in the data/masking group, default `None`. Help: per-echo
  phase NIfTI files in radians, in ascending echo order, required for `nlls`.
- `t2smap_workflow` signature gains `phase=None`.

### Validation (in `t2smap_workflow`)

- `fittype == "nlls"` requires `phase is not None` → else `ValueError`.
- `phase is not None` requires `len(phase) == len(data)` and matching shapes
  after loading → else `ValueError`.
- `phase is not None` with `fittype != "nlls"` → `ValueError` (phase only used
  by nlls).
- `fitmode == "varys0"` requires `fittype == "nlls"` → else `ValueError`.
- Existing constraint retained: excluding volumes is unsupported with `ts`.
  `varys0` **permits** `exclude`: because R2*/frequency are shared across
  volumes, the goal is an accurate shared T2* uncontaminated by excluded
  volumes, while per-volume S0 is secondary. Excluded volumes are dropped from
  the shared R2*/frequency estimation only.

## Data flow in the workflow

1. Magnitude data loaded as today → `data_cat` (samples × echos × time).
2. When `phase` is provided, phase loaded identically
   (`io.load_data_nilearn`), with the same dummy-scan trimming and `exclude`
   handling, producing `phase_cat` of the same shape. Shape mismatch → error.
3. **Adaptive mask, `make_optcom`, and RMSE continue to use magnitude
   `data_cat` only** — unchanged behavior.
4. The T2*/S0 estimation step receives the complex data. The masked magnitude
   and masked phase are passed to the decay functions; the complex array
   `magnitude · exp(i·phase)` is formed inside `decay.py`.
5. The written `S0` map is `np.abs(complex S0)`. Real `T2*` flows through
   `modify_t2s_s0_maps` / floor logic unchanged.

### Dispatch

`fit_decay` is also called by the main `tedana` workflow (`workflows/tedana.py`)
and unpacks exactly six return values, so its signature must **not** change.
Therefore the complex path uses a **dedicated entry point** rather than
extending `fit_decay`:

```
if fittype == "nlls":
    complex_results = decay.fit_complex_decay(
        data=...,            # masked magnitude (Md x E x T)
        phase=...,           # masked phase, radians (Md x E x T)
        tes=tes,
        adaptive_mask=...,
        fitmode=fitmode,     # "all", "ts", or "varys0"
        use_volumes=...,     # boolean (T,) volume mask; None == all volumes
        n_threads=n_threads,
    )
    # dict with keys: t2s, s0, frequency_hz, phase0, failures, cost, nfev
else:
    decay_function = decay.fit_decay if fitmode == "all" else decay.fit_decay_ts
    t2s_full, s0_full, failures, t2s_var, s0_var, t2s_s0_covar = decay_function(...)
```

`fit_decay`/`fit_decay_ts` are left untouched (loglin/curvefit only).
`fit_complex_decay` is the single new public dispatcher for nlls; it routes to
the per-mode helpers below.

## New code in `tedana/decay.py`

Ported from `dahnke`, stripped of modulation:

- `complex_decay_model(echo_times, s0, r2star, frequency_hz)` — the model.
- `_initial_complex_decay_params(signal, echo_times)` — loglin + phase init.
- `_fit_complex_decay_1d(signal, echo_times, bounds, max_nfev)` — per-sample
  fit returning `s0, r2star, frequency_hz, phase0, cost, nfev, success`.
  Returns failure sentinel when fewer than 2 finite echoes are available or the
  optimizer raises.
- `fit_complex_monoexponential(data_cat, phase_cat, echo_times, adaptive_mask,
  report, n_threads)` — mirrors `fit_monoexponential`'s adaptive-mask echo-group
  loop (`echos_to_run`, first-`echo_num`-echoes, limited/full assembly via
  `echo_masks`), but calls the complex fitter. Returns `t2s, s0` (magnitude)
  plus `frequency_hz`, `phase0`, and `failures` arrays. Parallelized with
  `joblib` like `fit_monoexponential`.
- `_fit_complex_decay_joint(signal, echo_times, max_r2star, max_frequency_hz,
  max_nfev)` — ported joint fit: shared R2*/frequency, per-volume complex S0
  estimated analytically (linear for fixed R2*/frequency). Returns per-volume
  `s0`/`phase0`, scalar `r2star`/`frequency_hz`, and optimizer diagnostics.
- `fit_complex_decay(data, phase, tes, adaptive_mask, fitmode, use_volumes,
  n_threads)` — the public nlls dispatcher. Forms the complex array and routes:
  - `"all"`  → `fit_complex_monoexponential` (single fit per voxel) → returns a
    dict with `(Md,)` arrays.
  - `"ts"`   → loops volumes calling `fit_complex_monoexponential` per volume
    (mirroring `fit_decay_ts`) → `(Md, T)` arrays.
  - `"varys0"` → `_fit_complex_decay_joint` per voxel over the `use_volumes`
    subset for shared R2*/frequency, then analytic per-volume S0 over **all**
    volumes → `t2s`/`frequency_hz` `(Md,)`, `s0`/`phase0` `(Md, T)`.

`fit_decay`/`fit_decay_ts` are unchanged. The extra `frequency_hz`/`phase0`/
diagnostics are returned only through `fit_complex_decay`'s result dict.

## Downstream shape handling

- `all` (nlls): single complex fit over all (echo, time) samples per voxel,
  one `S0`/`phase0`/`R2*`/`frequency_hz` per voxel → 3D maps. Mirrors
  `curvefit`'s flatten approach.
- `ts`: per-timepoint complex fit → 4D maps, as for curvefit.
- `varys0`: 3D `T2*`/`R2*`/`frequency_hz`, 4D `S0`-magnitude and 4D `phase0`.
  - Shared R2*/frequency are fit from the **non-excluded** volumes only. The
    per-volume complex S0 (linear given fixed R2*/frequency) is then computed
    for **all** retained volumes — including any `exclude`d ones — by analytic
    solve, yielding a full-length 4D S0/phase0 timeseries. To do this,
    `fit_decay_varys0` receives both the full data and a boolean
    `use_volumes` mask (the workflow already computes this for `exclude`).
  - `modify_t2s_s0_maps`: floor the 3D T2*; apply the limited-map masking to
    the 4D S0 by broadcasting the adaptive mask across the time axis.
  - `make_optcom`: uses the 3D T2* for echo weighting (unchanged).
  - `rmse_of_fit_decay_ts`: handle 3D-T2* / 4D-S0 by broadcasting T2* across
    time when reconstructing the per-volume model fit.

## Outputs (`tedana/resources/config/outputs.json`)

New descriptors:

- `frequency img` → orig `frequencyHz`, bids `frequencyHzmap`.
- `phase0 img` → orig `phase0`, bids `phase0map`.
- `fit cost img` → orig `fit_cost`, bids `desc-fitCost_statmap` (verbose only).
- `fit nfev img` → orig `fit_nfev`, bids `desc-fitNfev_statmap` (verbose only).

Fit failures reuse the existing `fit failures img` descriptor (derived from
`success`). `frequency img` and `phase0 img` are written whenever
`fittype == "nlls"`; cost/nfev under `--verbose`.

## Testing (TDD)

Unit tests (`tedana/tests/test_decay.py`):

- `complex_decay_model` returns the analytic complex signal.
- `_fit_complex_decay_1d` recovers known `R2*`, `|S0|`, `frequency_hz`,
  `phase0` from noiseless synthetic complex decay; degrades gracefully with
  noise; returns failure with <2 finite echoes.
- `fit_complex_monoexponential` respects the adaptive-mask echo grouping.
- `_fit_complex_decay_joint` / `fit_decay_varys0` recover shared R2*/frequency
  with per-volume S0.

Integration test (`tedana/tests/test_t2smap.py`): run
`t2smap_workflow(fittype="nlls", phase=...)` on the existing five-echo test
data (synthesizing zero/known phase), asserting the expected output files exist
and have correct shapes for `all`, `ts`, and `varys0`. Validation-error tests
for the constraint matrix above.

## Out of scope (this project)

- Dahnke modulation / gradient / slice-profile correction.
- Adding `nlls`/`--phase` to the main `tedana` workflow (this spec is
  `t2smap`-only).
- A magnitude-only `varys0` analog for loglin/curvefit.

## Future work

- **Dahnke correction.** The full Dahnke through-slice modulation / gradient /
  slice-profile correction (present in the `dahnke` subpackage's
  `fit_complex_decay_nlls`) is a planned future addition. The complex NLLS
  model, per-sample fitter, and joint fit ported here are deliberately designed
  so the modulation term can be reintroduced later (the model already
  multiplies by a `modulation` factor of 1). This project intentionally defers
  it to keep the scope focused; it is not being abandoned.
