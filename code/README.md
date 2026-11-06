# DASC controlled evaluation: code

Code for the controlled empirical study of drift-aware spectral conformal
prediction (DASC) and its baselines on non-exchangeable streams.  One
framework, one point forecaster per dataset shared by every method, one
tuning protocol for every method, every table generated from result files.

## Files

| File | Purpose |
|---|---|
| `dasc_lib.py` | the methods, the spectral features, the weighted quantile, the stream runner |
| `forecasters.py` | causal point forecasters (seasonal naive, rolling ridge, rolling gradient boosting) |
| `datasets.py` | data loaders (`DASC_DATA_ROOT`), per-dataset settings, simulation designs |
| `metrics.py` | coverage, width, interval score, worst-block coverage, local coverage, paired moving-block bootstrap |
| `tuning.py` | hyperparameter grids and the tune-then-freeze protocol |
| `test_causality.py` | perturbation test: every interval at `t` depends on `y[:t]` only |
| `check_reproduce_original.py` | regression anchor against the numbers in the original manuscript |
| `run_real_data.py` | electricity, weather, finance |
| `run_simulations.py` | twelve simulation scenarios, 100 replications, drift-detection check |
| `run_assumption_diagnostic.py` | refutation test of the Lipschitz regime assumption (L_min diagnostic), per-set levels eta |
| `run_assumption_diagnostic_bonf.py` | the same statistic at the per-set level eta / J (Bonferroni over the J clusters); stores the per-pair statistics; the article reports its output |
| `run_timing.py` | per-step computation time of every method |
| `run_all.py` | runs the stages above in order |

In comments, "the original project", "the original manuscript" and "the original drivers" mean the earlier
version of this work (`dasc.py`, `run_first_simulation.py`, `run_real_*.py`), which is not part of this
repository; see `NOTICE.md` in the repository root.  Nothing here imports those scripts;
`check_reproduce_original.py` contains a copy of one of their functions.

## Running

```
set DASC_DATA_ROOT=<folder with uci_household_power, open_meteo_dallas, fred_sp500>
python -X utf8 run_all.py
```

The default data root is `../data/real`.  Results go to `../results/`.  The
derived hourly electricity series is cached in `../results/derived/`.
Python 3.13, numpy 2.4, pandas 2.3, scipy 1.16, scikit-learn 1.9, xgboost 3.3,
pyarrow 25 (exact versions: `requirements.txt` in the repository root).

## Methods (all in `dasc_lib.py`, common interface)

`lo, hi, diag = method.interval(t, pred_t, residuals, features)` then
`method.update(miss)`.  At time `t` a method reads `residuals[i]` for `i < t`,
`features[i]` for `i <= t` (where `features[t]` is computed from
`y[t-l], ..., y[t-1]`), and its own state.

| Name | Method (theory label) | Hyperparameters tuned |
|---|---|---|
| `rolling` | rolling-window split conformal, uniform weights, fixed alpha | pool |
| `aci_clip` | adaptive conformal inference (Gibbs and Candes, 2021), alpha_t projected onto [0.01, 0.35] (ACI-proj) | pool, gamma |
| `aci_noclip` | the same with the free update, no projection (ACI-free) | pool, gamma |
| `pid` | PID-style controller on alpha_t as in the original project code (not the quantile-tracking conformal PID of Angelopoulos et al., 2023), projected onto [0.01, 0.35] | pool, k_p, k_i, k_d |
| `decay` | weighted split conformal with weights decay^(t-i), fixed alpha (Barber, Candes, Ramdas and Tibshirani, 2023, time-decay weights) | pool, decay |
| `spectral_fixed` | Gaussian kernel weights on spectral features, fixed alpha (spectral-only) | pool, h, feature window |
| `spectral_aci` | spectral weights + ACI update, no drift gate (DASC-nogate), projected onto [0.01, 0.35] | pool, h, gamma, feature window |
| `dasc_fixed` | spectral weights + drift score + gate, fixed alpha (DASC-fixed) | pool, h, lambda, m_min, feature window |
| `dasc_clip` | spectral weights + drift score + gate + ACI update projected onto [0.01, 0.35] (the original DASC; DASC-proj) | pool, h, lambda, m_min, feature window |
| `dasc_noclip` | the same with the free update (DASC-free) | pool, h, lambda, m_min, feature window |

Definitions follow `dasc.py` and Sections 4 to 6 of the original manuscript:
banded periodogram over K = 6 bands of the positive Fourier frequencies of a
window of length l ending at t-1, l2-normalised; weights exp(-d^2/h^2);
drift score D_t = ||mean of the last l_r = 64 feature vectors -
kernel-weighted pool mean|| (l_r = 64 in every data set, as in the original
drivers); gate m_t = m_max - (m_max - m_min) min(1, D_t/lambda) with a floor of
20 residuals; weighted quantile inf{q : sum_i w_i 1{R_i <= q} >= 1 - alpha_t};
update alpha_{t+1} = alpha_t + gamma (alpha - miss), gamma = 0.015 for DASC.

Per-step diagnostics logged for every method: `alpha_t`, `alpha_eff`
(equal; no level relaxation is used), `miss`, `q`, `width`, `score`, `dbar`
(weight-averaged spectral distance under the final gated weights),
`dbar_pre` (under the pre-gate weights), `drift` (D_t), `recent_gap`,
`gate_m` (m_t), `pool_size`, `neff`, `pi` (final weight on residuals equal
to q), `wmax`, `eps_005`, `eps_010` (grid term sqrt(log(4K/eta)/(2 n_eff)) +
1/K, K = max(2, ceil(n_eff))), `U` (pre-projection value of the update),
`proj_resid` (alpha_{t+1} - U; zero for the free update), `empty`.

### Conventions that apply to every method

* One quantile operator.  All methods use the weighted quantile above with
  weights normalised over the pool and no point mass on the test point.  For
  uniform weights this is one order statistic below numpy's `'higher'`
  quantile used by the original baseline drivers, and it omits the test-point
  mass of Barber et al. (2023) for the decay-weighted method.  The effect is
  at most one order statistic of the pool.
* Free (unprojected) variants.  The state alpha_t follows the plain update
  without projection.  When alpha_t <= 0 the interval is the whole real line
  (miss 0, infinite width); when alpha_t >= 1 it is the empty set (miss 1,
  width 0, scored as the degenerate interval {pred_t}); otherwise the
  weighted quantile at level 1 - alpha_t.  These are the conventions of the
  revised theory's long-run result.  Steps with an infinite width are counted
  (`n_inf_width`) and excluded from the width and interval-score means; in
  the paired bootstrap such steps are excluded for both methods of the pair
  (`n_masked`).  In the tuning criterion an infinite step makes the
  configuration's development score infinite, so it is not selected.
* Projected variants.  The per-step projection residual r_t = alpha_{t+1} -
  U_t is logged; the summary reports the binding fractions N+/T (cap) and
  N-/T (floor) and the exact identity mean(E_t) - alpha = (alpha_1 -
  alpha_{T+1})/(gamma T) + sum_t r_t/(gamma T) on the evaluation window.
* Non-finite residuals (before a forecaster's first prediction) are excluded
  from the pool.  The original `dasc()` kept them as NaN entries with their
  weights for the first m_max steps.
* Finance (non-negative target): the lower limit is replaced by max(lo, 0)
  for every method; width = hi - max(lo, 0).
* The drift score's recent window has length l_r = 64 in every data set (as
  in the original drivers and the revised theory), not the feature window.
* The original `run_first_simulation.spectral_feature` returned raw,
  un-normalised, zero-padded power for windows of 8 to 12 points (t = 8..11).
  `dasc_lib.spectral_feature` applies the banded, normalised definition to
  every window of at least 8 points.  `check_reproduce_original.py` shows the
  effect of both differences on the published numbers.

## Forecasters (`forecasters.py`, `datasets.py`)

Every method on a dataset uses the same forecast array `pred`; `pred[t]`
depends on `y[:t]` and on calendar information for `t` only.

| Dataset | Forecaster | Definition |
|---|---|---|
| electricity, weather | `naive24` | pred[t] = y[t-24] |
| electricity, weather | `ridge` | ridge (alpha = 1, standardised inputs) on y[t-1], y[t-2], y[t-3], y[t-24], y[t-48], y[t-168] and sin/cos of hour of day and day of week; first fit at t = 900; refit every 168 steps on the most recent 1440 rows with target index < refit time |
| electricity, weather | `xgb` | gradient boosting (xgboost, 200 trees, depth 4, learning rate 0.05, subsample 0.8, seed 0, single-threaded) on the same inputs and refit schedule |
| finance | `naive1` | pred[t] = y[t-1] |
| finance | `ridge` | ridge on y[t-1], y[t-2], y[t-3], y[t-5], y[t-10], y[t-21] and the means of y over the past 5 and 21 days; first fit at t = 126; refit every 21 steps on the most recent 504 rows |
| finance | `xgb` | gradient boosting on the same inputs and schedule |
| simulations | `naive1` | pred[t] = y[t-1] |
| simulations | `ridge` | ridge on y[t-1], ..., y[t-6]; first fit at t = 120; refit every 50 steps on the most recent 300 rows |

Electricity: hourly mean of global active power, first 12,000 non-missing
hourly rows (row-based, as in the original project; two gaps of 62 h and 2 h
lie inside, so the lag-24 forecast is misaligned for 24 rows after the long
gap); true timestamps are kept for the hour-of-day analysis.  Weather: first
24,000 hourly Dallas temperatures (gap-free).  Finance: absolute daily log
returns of the S&P 500 in percentage points from FRED, missing closes dropped
before differencing (2,514 values).

## Tuning protocol (`tuning.py`)

* Criterion: mean interval score at level 0.10 on the development data, the
  same for every method.
* Real data: development prefix = steps with start <= t < 0.25 n; each
  configuration runs once over the whole stream, is scored on the prefix, and
  its evaluation-period metrics come from the same run (the adaptive state
  carries over the boundary).  Start: electricity 1080, weather 1440, finance
  252.
* Simulations: development seeds 1000 to 1009 (full streams), evaluation
  seeds 0 to 99.
* Pool candidates (in every grid): electricity {720, 1440}, weather
  {1080, 2160}, finance {252, 504}, simulations {180, 360}.  Feature windows:
  hourly {168, 336}, finance {32, 64}, simulations {32, 64}.  gamma in
  {0.005, 0.01, 0.015, 0.02, 0.03, 0.05}; h in {0.35, 0.55, 0.8}; decay in
  {0.97, 0.985, 0.99, 0.995, 0.998, 0.999}; lambda in {0.15, 0.45}; m_min in
  {0.25, 0.5} of the pool; PID gains k_p {0.01, 0.025, 0.05}, k_i {0.001,
  0.003}, k_d {0, 0.01}; spectral ACI gamma {0.01, 0.015, 0.03}; DASC gamma
  fixed at 0.015.
* Configurations per method: rolling 2, ACI 12 (each variant), PID 24, decay
  12, spectral fixed 12, spectral ACI 36, DASC 48 (each of the three
  variants); 254 per forecaster.  DASC has the largest budget.
* The grid files in `results/tuning/` list every configuration with its
  development score and its evaluation metrics, so a reader can verify that
  selection used the development score only.

## Metrics (`metrics.py`)

Marginal coverage, mean and median width, mean interval score
(width + (2/alpha) times the distance of the outcome outside the interval),
worst-block coverage (minimum over non-overlapping blocks of 168 h, 63 trading
days or 100 simulation steps) and the share of blocks below 0.85, local
coverage by hour of day and month (hourly), by causal 21-day volatility
tercile with cut points fixed on the development prefix and by year (daily),
by regime (simulations).  Paired differences between methods on real data use
a moving-block bootstrap on the per-step differences with the same blocks for
both methods (2000 resamples, percentile 95% intervals, block length 168 h,
21 trading days).  Simulation uncertainty is the standard error over the 100
replications.

## Simulations (`datasets.py`, `run_simulations.py`)

Twelve scenarios, 100 evaluation replications each.  The original three-phase
stream keeps the code's n = 1400 and breaks at 450 and 900.  Recurring-regime
scenarios (blocks of 150 steps): frequency only, frequency and noise scale,
amplitude only (l2-normalised features are blind to amplitude).  Controls:
stationary AR(1), pure drift, white noise.  The five stress scenarios of the
original paper (n = 1500).  The drift-detection check compares the drift
score D_t with a rolling 50-step coverage deficit on scenarios with a known
break: threshold = 99th percentile before the break, detection within 300
steps, AUC of the 100 post-break values against the 300 pre-break values.

## Outputs (`../results/`)

`real_summary.csv`, `real_local_coverage.csv`, `real_paired_bootstrap.csv`,
`forecaster_accuracy.csv`, `tuning/*_grid.csv`, `tuning/*_chosen.csv`,
`sim_summary.csv`, `sim_regime_summary.csv`, `sim_paired.csv`,
`sim_per_seed.csv`, `drift_detection.csv`, `finance_2022_drift_diagnostic.csv`,
`assumption_diagnostic.csv`, `timing.csv`, `reproduction_check.csv`,
`causality_test_report.csv`, and per-step diagnostics
`diagnostics/<dataset>_<forecaster>_<method>.parquet` (real data: every
step with an `in_eval` flag; simulations: seeds 0 to 2) with the columns
listed under "Per-step diagnostics" above.

## Assumption diagnostic (`run_assumption_diagnostic.py`)

For each real series and forecaster (and four simulated streams), k-means
clusters (20, at least 30 points each) of the spectral features define
feature-measurable index sets; for every pair the two-sample Kolmogorov-Smirnov
distance of the residuals, the mean pairwise feature distance D_AB and the
grid terms eps_A(eta), eps_B(eta) give the diagnostic L_min(delta, eta) =
max over pairs of (KS - delta - eps_A - eps_B)^+ / D_AB, a lower confidence
bound on any admissible Lipschitz constant, reported for delta in {0, 0.05,
0.1} and eta in {0.05, 0.1} together with the number of pairs whose numerator
is positive.  It refutes constants; it does not estimate L.

Level.  Each cluster's concentration event fails with probability at most eta, and the
maximum over pairs uses all J retained clusters at once, so the bound holds with probability
at least 1 - J eta (J = `n_clusters_used`).  `results/assumption_diagnostic.csv` reads eta = 0.05
and 0.1 per cluster, which is a simultaneous level of 1 - J eta (nil for J = 11 to 20).
`run_assumption_diagnostic_bonf.py` recomputes every row at eta / J from the stored per-pair
statistics (`results/assumption_diagnostic_pairs.csv`) and writes
`results/assumption_diagnostic_bonf.csv`, the correction applied to the stored original run (the values the article
reports come from the `_xgb1thread` file described below).
The first eleven columns of that file are the original ones, unchanged; the corrected values are
`L_min_bonf` and `n_pairs_refuting_L0_bonf`, the per-set-level values `*_orig`.  The original
diagnostic run produced its gradient-boosting forecasts with `n_jobs = 4`; the current code uses
`n_jobs = 1`, so a fresh run of the current code differs from the stored file for those rows
(`results/assumption_diagnostic_bonf_xgb1thread.csv` holds the corrected values for the
current-code forecasts).
