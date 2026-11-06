# dasc-controlled-evaluation

Code and saved results for

**Drift-Aware Spectral Conformal Prediction for Non-Exchangeable Streaming Data: A Controlled Evaluation**

by Jeffery Opoku and David Banahene. The repository owner on GitHub is `jopoku16`.

The repository holds the code and the saved results of the study, and nothing else: no article text, no
tables or figures, and no raw data. The tables and figures of the article are produced from the saved results
outside this repository. Everything needed to rerun the experiments is here, and the stored results let you check
parts of the study without rerunning it.

Please read `NOTICE.md`: two earlier repositories contain a version of this work whose reported width reductions
do not hold.

## What is in the repository

```
code/       the experiments, the checks and the code for the exploratory studies (Python)
  theory_checks/   numerical checks of the theory, with the recorded outputs
results/    every saved result of the study
  tuning/        every tuned configuration with its development score (grids) and the chosen ones
  diagnostics/   per-step diagnostics of the runs (parquet, 411 files)
  ext_parts/     per-cell parts that ext_combine.py joins into the ext_*.csv files
  ext_logs/      console logs of the extended-grid runs
requirements.txt   exact package versions
LICENSE, CITATION.cff, NOTICE.md, SHA256SUMS.txt, ALLOWLIST_USED.txt, DATA_SHA256.txt
```

In the comments of the code, "the paper" means the article that this repository accompanies.

`code/README.md` documents the methods, the forecasters, the tuning protocol, the metrics and the simulation
designs. `ALLOWLIST_USED.txt` lists every file in the repository. `SHA256SUMS.txt` holds the SHA-256 checksum of
every file except itself.

## Setup

Python 3.13 with the packages in `requirements.txt` (numpy 2.4, pandas 2.3, scipy 1.16, scikit-learn 1.9,
xgboost 3.3, pyarrow 25):

```
pip install -r requirements.txt
```

All code is run from the `code/` directory. Scripts find the results folder relative to their own location, so the
repository can sit anywhere.

## Data

The raw data are not included. The data directory is given by the environment variable `DASC_DATA_ROOT`. If the
variable is not set, the code looks in `data/real` at the root of this repository, which is also where the download
scripts put the files. To point the code at a data directory elsewhere:

```
export DASC_DATA_ROOT=/path/to/real        # macOS and Linux
set DASC_DATA_ROOT=D:\path\to\real         # Windows, command prompt
$env:DASC_DATA_ROOT = "D:\path\to\real"    # Windows, PowerShell
```

The directory must contain:

```
<DASC_DATA_ROOT>/uci_household_power/household_power_consumption.txt
<DASC_DATA_ROOT>/open_meteo_dallas/open_meteo_dallas_hourly_temperature.csv
<DASC_DATA_ROOT>/fred_sp500/sp500_daily_fred.csv
```

Download the three public data sets with

```
python code/download_real_data.py
python code/download_weather_data.py
python code/download_finance_data.py
```

| Data set | Source | Used as |
|---|---|---|
| Electricity | UCI Machine Learning Repository, Individual Household Electric Power Consumption | hourly mean of global active power, first 12,000 non-missing hourly rows |
| Weather | Open-Meteo historical weather API, hourly 2 m temperature for Dallas, 1 January 2021 to 31 December 2023 | first 24,000 hourly values |
| Finance | FRED, series SP500 (S&P 500 daily index) | absolute daily log returns in percentage points |

Each data set remains under its provider's terms of use; check them before redistributing any of the files.

The three data files used for the stored results are identified by the SHA-256 checksums in `DATA_SHA256.txt`. The
finance file covers 13 June 2016 to 12 June 2026. FRED serves only the most recent ten years of this series, and the other two
sources can also change, so a fresh download will in general differ from these files and results computed from it
will not match the stored ones exactly. To reproduce the stored results exactly, compare your files with
`DATA_SHA256.txt`.

## Run order

Run each command from `code/`, with `DASC_DATA_ROOT` set if the data are not in `data/real`.

Main study:

1. `python -X utf8 test_causality.py` runs the perturbation test: every interval at time `t` must depend on `y[:t]` only.
2. `python -X utf8 check_reproduce_original.py` produces the reproduction anchor against the earlier version.
3. `python -X utf8 run_real_data.py` tunes and evaluates every method on the three real streams.
4. `python -X utf8 run_simulations.py` does the same for the twelve simulation scenarios and runs the drift-detection check.
5. `python -X utf8 run_sim_gamma_sensitivity.py` runs the equal-step-size simulations.
6. `python -X utf8 run_assumption_diagnostic.py` runs the refutation test of the Lipschitz regime assumption.
7. `python -X utf8 run_assumption_diagnostic_bonf.py all` evaluates the same statistic at the per-set level eta / J, where J is the number of clusters used (a Bonferroni correction), from per-pair statistics that it computes with the current single-threaded gradient-boosting forecasts, and writes `assumption_diagnostic_bonf_xgb1thread.csv`. The files `assumption_diagnostic_bonf.csv` and `assumption_diagnostic_pairs.csv` are the same correction applied to the 30 October run of step 6, whose gradient-boosting forecasts used four threads; they differ from the `_xgb1thread` files only in the gradient-boosting rows. The docstring of the script describes the stages (`pairs`, `build`, `onethread`, `all`).
8. `python -X utf8 run_timing.py` measures the cost per step.

`python -X utf8 run_all.py` runs steps 1 to 8 in sequence and prints the time of each stage. In the recorded run
the whole sequence took roughly two hours on eight cores.

Equal-budget study: `python -X utf8 run_dascgamma.py [finance electricity weather] [sim]`.

Further studies, each writing new files only:

| Study | Run | Outputs in `results/` |
|---|---|---|
| Extended step-size and bandwidth grids | `ext_run_real.py`, `ext_run_sim.py`, then `ext_combine.py`; supplements `ext_gamma_top.py`, `ext_sim_gamma_top.py`, `ext_neff_by_h.py`, `ext_seed_check.py` | `ext_*.csv`, `ext_parts/`, `ext_logs/`, `tuning/ext_*`, `diagnostics/ext_*`; design in `ext_design.txt` |
| Block length and bootstrap seed | `blockseed_run.py run`, then `blockseed_summarise.py` | `blockseed_*.csv` |
| Evaluation pitfalls | `pitfalls_run.py real`, `sim`, `reuse`, `check` | `pitfalls_*.csv` |
| Exploratory study 1 (block-calibrated test for the Lipschitz constant) | `lead1_selftest.py`, `lead1_repro_check.py`, `lead1_run.py`, then `lead1_summarize.py` | `lead1_*` |
| Exploratory study 2 (phase and calendar variants) | `lead2_test.py`, `lead2_invariance.py`, `lead2_run.py tune`, then `lead2_run.py evaluate`; protocol in `lead2_protocol.txt` | `lead2_*` |
| Exploratory study 4 (averaged recursion), in this order | `lead4_run_grid.py`, `lead4_checks.py`, `lead4_analyze.py`, `lead4_match.py`, `lead4_link.py`, `lead4_check_rolling.py`, `lead4_scenario.py`, `lead4_substitution.py`, `lead4p_checks.py` | `lead4_*`, `lead4p_*` |
| Finance monthly drift diagnostic | `finance_2022_diagnostic.py` | `finance_2022_drift_diagnostic.csv` |
| Numerical checks of the theory | `python -X utf8 theory_checks/run_all_checks.py` (about ten minutes) | `code/theory_checks/outputs/` |

The docstring at the top of each script lists its arguments and outputs.

Notes:

* All seeds are fixed in the scripts: simulation evaluation seeds 0 to 99, development seeds 1000 to 1009, bootstrap seed 0, gradient-boosting seed 0.
* The studies that use worker pools use at most two processes. The extended-grid, pitfalls, block-length and exploratory runs took from a few minutes to a few hours each; the logs in `results/` record the times.
* Most scripts append to or replace their own rows in the shared result files. Work in a copy if you want to keep the stored results untouched.
* `theory_checks/check_longrun.py` has three parts. Part 1 checks identities on the level paths of the earlier version and needs its result files: pass the root of a checkout of the earlier repository with `--repo`. Without `--repo`, part 1 is skipped. The outputs of the recorded run are included.

## Which script writes which result

| Result files | Written by |
|---|---|
| `causality_test_report.csv` | `test_causality.py` |
| `reproduction_check.csv` | `check_reproduce_original.py` |
| `real_summary.csv`, `real_local_coverage.csv`, `real_paired_bootstrap.csv`, `forecaster_accuracy.csv`, `tuning/<dataset>_<forecaster>_{grid,chosen}.csv`, `diagnostics/<dataset>_<forecaster>_<method>.parquet` | `run_real_data.py` |
| `sim_summary.csv`, `sim_regime_summary.csv`, `sim_paired.csv`, `sim_per_seed.csv`, `drift_detection.csv` | `run_simulations.py` |
| `sim_gamma_sensitivity.csv`, `sim_gamma_sensitivity_per_seed.csv` | `run_sim_gamma_sensitivity.py` |
| `assumption_diagnostic.csv` | `run_assumption_diagnostic.py` |
| `assumption_diagnostic_bonf.csv`, `assumption_diagnostic_pairs.csv`, and both with the suffix `_xgb1thread` | `run_assumption_diagnostic_bonf.py` |
| `timing.csv` | `run_timing.py` |
| `*_dascgamma.csv`, `tuning/*_dascgamma_*.csv`, `diagnostics/*_dascgamma.parquet` | `run_dascgamma.py` |
| `finance_2022_drift_diagnostic.csv` | `finance_2022_diagnostic.py` |
| `ext_*.csv`, `ext_parts/`, `tuning/ext_*` | `ext_*.py` (the combined `ext_*.csv` files come from `ext_combine.py`) |
| `blockseed_*.csv` | `blockseed_run.py`, `blockseed_summarise.py` |
| `pitfalls_*.csv` | `pitfalls_run.py` |
| `lead1_*`, `lead2_*`, `lead4_*`, `lead4p_*` | the script with the same prefix (`lead4p_*` from `lead4p_checks.py`) |

## Per-step diagnostics

`results/diagnostics/` holds one parquet file per run, named `<dataset>_<forecaster>_<method>.parquet` (the
extended-grid and equal-budget runs add the suffixes or prefixes `ext_`, `_ext`, `_wide` and `_dascgamma`). For real
data each file has one row per time step with an `in_eval` flag that marks the evaluation period. For simulations
each file has seeds 0 to 2. The columns are `t`, `lo`, `hi`, `miss`, `width`, `score`, `U`, `proj_resid`, `alpha_t`,
`alpha_eff`, `q`, `neff`, `dbar`, `dbar_pre`, `drift`, `recent_gap`, `pool_size`, `gate_m`, `pi`, `wmax`, `eps_005`,
`eps_010`, `empty`, plus `datetime` and `in_eval` (real data) or `seed` and `regime` (simulations). The files make up
about 276 MB of the repository. `blockseed_run.py` recomputes the paired bootstrap from them.

## How to check the repository

1. Integrity: from the repository root, `sha256sum -c SHA256SUMS.txt` (on Windows, compare `Get-FileHash -Algorithm SHA256` output with the file).
2. Causality: `python -X utf8 test_causality.py` (about five minutes) prints `CAUSALITY TEST PASSED` after 630 interval checks and rewrites `results/causality_test_report.csv`. A new run reproduces every pass flag of the stored report and every interval value except those of the gradient-boosting (`xgb`) forecaster streams, which differ: the stored report was written before the gradient-boosting forecaster was set to a single thread (see the docstring of `run_assumption_diagnostic_bonf.py`).
3. Stored summaries regenerated from stored per-run files: `lead1_summarize.py`, `blockseed_summarise.py`, `ext_combine.py` and `finance_2022_diagnostic.py` rewrite 18 result files (`lead1_cells.csv`, `lead1_criteria.csv`, `blockseed_summary.csv`, 14 `ext_*.csv` summaries and `finance_2022_drift_diagnostic.csv`), and the new files are byte for byte the stored ones. Run them in a copy and compare checksums.
4. A short experiment from the raw data: `python -X utf8 run_real_data.py finance --no-xgb` (about two minutes) recomputes the finance rows for the naive and ridge forecasters of `real_summary.csv`; they match the stored rows to within 1e-15. `python -X utf8 run_sim_gamma_sensitivity.py --seeds 3` (about half a minute) reproduces the stored rows for seeds 0 to 2 exactly. Both overwrite result files, so run them in a copy.
5. Theory checks: `theory_checks/check_drift_lemma.py` and `check_combination.py` reproduce their stored outputs exactly.
6. The corrected assumption diagnostic: `python -X utf8 run_assumption_diagnostic_bonf.py onethread` recomputes `results/assumption_diagnostic_bonf_xgb1thread.csv` from the stored `assumption_diagnostic_pairs_xgb1thread.csv` and `assumption_diagnostic.csv` (the new file is byte for byte the stored one), and `python -X utf8 run_assumption_diagnostic_bonf.py pairs --tag check --datasets sims` writes `results/assumption_diagnostic_pairs.check.csv`, whose 950 rows match the simulated-stream rows of the stored `assumption_diagnostic_pairs.csv` exactly. Delete the second file afterwards.

## Licence and citation

MIT licence, see `LICENSE`. Citation: see `CITATION.cff`.
