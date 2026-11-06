# Numerical checks of the theory

Each script checks one or more statements of the theory section of the associated article and writes its results to `outputs/`. All scripts use fixed seeds and only numpy, scipy, pandas and scikit-learn. Run from this directory with `python -X utf8 <script>`. Total runtime about ten minutes.

| Script | Checks | Output | Runtime |
|---|---|---|---|
| `dasc_causal.py` | reference causal implementation of one DASC step (mirrors `dasc.py` of the earlier project) returning `dbar`, `dbar_pre`, `pi_t` (`mass_q`), `wmax`, `recent_gap`; free and projected updates; grid term | (module) | |
| `check_perstep_bound.py` | Theorem 5 against the exact conditional coverage in six simulated models (A: assumption exact; B1-B4: conditional independence violated; B5: Lipschitz violated); also the old displayed concentration term | `perstep_bound_summary.txt`, `.csv`, `perstep_bound_steps.csv.gz` | 3 min |
| `check_longrun.py` | Theorems 6-7 and Corollary 8: exact identities on the published level paths (`--repo` points to a checkout of the earlier project; without it part 1 is skipped and part 3 reads the weather file under `data/real` of this repository), synthetic streams, the constant-stream counterexample, and a re-run of the weather stream with projected and free updates, as published and re-centred | `longrun_summary.txt`, `longrun_published_paths.csv`, `longrun_synthetic.csv`, `longrun_weather_rerun.csv` | 2 min |
| `check_grid_lemma.py` | Lemma 4 (uniform concentration), continuous and discrete laws, four weight profiles | `grid_lemma_summary.txt`, `.csv` | 3 min |
| `check_mixing_bound.py` | Proposition 13 on m-dependent sequences; the arithmetic of the old and new mixing remarks; the gap-block step of the old proof | `mixing_bound_summary.txt`, `mixing_bound_checks.csv` | 1 min |
| `check_drift_lemma.py` | Lemma 10 and the old remark D_t <= d_bar_t | `drift_lemma_summary.txt`, `drift_lemma_cases.csv.gz` | 1 min |
| `check_assumption_diagnostic.py` | Proposition 11 and the diagnostic L_min against a known constant; the old regression slope | `assumption_diagnostic_summary.txt`, `.csv` | 1 min |
| `check_combination.py` | Proposition 9 with the free update in models A and B1 | `combination_summary.txt`, `combination_check.csv` | 1 min |
| `run_all_checks.py` | runs everything above in order | `_run_all.log` | 10 min |

The `_run_*.log` files are the console logs of the recorded runs.

Notes on the weather re-run: `check_longrun.py` sets the first 24 (undefined) residuals to zero instead of letting them enter the pool as NaN as the original driver did; the as-published configuration then gives coverage 0.9655 and width 6.4086 against the published 0.9656 and 6.4129. The identity checks do not depend on this.
