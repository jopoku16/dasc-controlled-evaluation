"""
run_assumption_diagnostic_bonf.py: the Bonferroni-corrected refutation diagnostic.

Why this file exists.  The pair bound of the article gives, for two index sets A and B,
    KS(Fhat_A, Fhat_B) <= L D_AB + delta + eps_A(eta) + eps_B(eta)
with probability at least 1 - 2 eta, because each set's concentration event (Lemma on the
uniform grid term) fails with probability at most eta.  The diagnostic
    L_min(delta, eta) = max over pairs (KS_AB - delta - eps_A - eps_B)^+ / D_AB
takes a maximum over all pairs of the J retained clusters, so it uses the J set events at
once.  The union bound over the J sets gives L_min(delta, eta) <= L with probability at
least 1 - J eta (J = number of index sets = retained clusters = n_clusters_used).
run_assumption_diagnostic.py reads the bound at eta = 0.05 and 0.10 for each set, so
the simultaneous level it attains is 1 - J eta, which is nil for J = 11 to 20.
The correction: evaluate the same statistic at the per-set level eta / J.  Then the
simultaneous statement has probability at least 1 - eta.

run_assumption_diagnostic.py is left unchanged (it is the record of the original run and
lead1_repro_check.py imports its diagnostic()).  This script does not overwrite
results/assumption_diagnostic.csv.

STORED FILES AND THE GRADIENT-BOOSTING SETTING (n_jobs).  Two variants are stored; they differ only in the boosting rows.
  results/assumption_diagnostic_bonf.csv  (pairs file assumption_diagnostic_pairs.csv)
      Built with `pairs ... --xgb-jobs 4`: boosting with n_jobs = 4, the code default on 30 Oct 2026 when
      run_assumption_diagnostic.py produced results/assumption_diagnostic.csv.  It reproduces that file to 6e-13
      (every count, J and median D identical).  A RECORD of the correction applied to the original forecasts.  Not read
      by any generator.
  results/assumption_diagnostic_bonf_xgb1thread.csv  (pairs file assumption_diagnostic_pairs_xgb1thread.csv)
      Built with the current code: boosting with n_jobs = 1 (forecasters.rolling_xgboost default), the setting of the
      boosting rows reported elsewhere in the paper and of this repository.  THE SOURCE OF THE REPORTED VALUES: the article's tables and numbers are generated outside this repository from this file.  Naive and ridge rows are identical in the two files.
  Rerunning the whole pipeline (run_all.py, which runs run_assumption_diagnostic.py and then `all`) gives the
  n_jobs = 1 variant everywhere.

Stages.
  pairs  The per-pair statistics (cluster sizes, KS distance, mean feature distance D_AB) were
         never stored.  This stage recomputes them for the same data, forecasters, feature
         windows, k-means settings (20 clusters, random_state 42, n_init 10, at least 30
         points per cluster) and simulation seed as the original run, and writes them to a
         per-pair file.  From that file L_min at any level is a group-by.  --xgb-jobs N sets the
         boosting thread count (default 1, the current code).
  build  Merges the n_jobs = 4 per-pair parts (tag names `pairs.<name>.csv`) into results/assumption_diagnostic_pairs.csv,
         recomputes the original rows (eta = 0.05 and 0.10 per set), checks them against
         results/assumption_diagnostic.csv (L_min to 1e-9, integer columns exactly), then writes
         results/assumption_diagnostic_bonf.csv.
  onethread  Same correction from the n_jobs = 1 per-pair file; writes the file the article reads (see onethread()).
  all    `pairs` for every dataset and the simulated streams with the current code, then `onethread`.

Run (two processes at most; DASC_DATA_ROOT must point at the folder that holds uci_household_power, open_meteo_dallas, fred_sp500):
  n_jobs = 4 record:  pairs --tag weather --datasets weather --xgb-jobs 4;  pairs --tag rest --datasets electricity finance sims --xgb-jobs 4;  build
  n_jobs = 1 source:  all
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

import datasets as ds
from dasc_lib import grid_term, spectral_feature_matrix
from run_assumption_diagnostic import DELTAS, ETAS, MIN_PER, N_CLUSTERS  # read only: same constants as the original run

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIM_SCENARIOS = ["original_three_phase", "recurring_freq_noise", "recurring_amplitude", "white_noise"]
KEYS = ["dataset", "forecaster", "feature_window"]


# ----------------------------------------------------------------------------
# stage 1: per-pair statistics (same arithmetic as run_assumption_diagnostic.diagnostic)
# ----------------------------------------------------------------------------

def _clusters_and_distances(feats, n_clusters=N_CLUSTERS, min_per=MIN_PER, seed=42):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(feats)
    groups = [np.where(km.labels_ == c)[0] for c in range(n_clusters)]
    groups = [g for g in groups if len(g) >= min_per]
    kept = [c for c in range(n_clusters) if int(np.sum(km.labels_ == c)) >= min_per]
    D = {}
    for i, j in itertools.combinations(range(len(groups)), 2):
        A, B = groups[i], groups[j]
        d = 0.0
        for a0 in range(0, len(A), 500):          # chunked exactly as in the original, to bound memory
            fa = feats[A[a0:a0 + 500]]
            d += np.linalg.norm(fa[:, None, :] - feats[B][None, :, :], axis=2).sum()
        D[(i, j)] = d / (len(A) * len(B))
    return groups, kept, D


def pair_rows(feats, resid, cache):
    """One row per cluster pair: kept-cluster positions i<j, original k-means labels, sizes, KS, D_AB.
    k-means and D_AB depend on the features only, so they are shared across forecasters whose index set is the same;
    k-means is deterministic given the data, so this returns exactly what the original recomputation returned."""
    key = hashlib.sha1(np.ascontiguousarray(feats).tobytes()).hexdigest()
    if key not in cache:
        cache[key] = _clusters_and_distances(feats)
    groups, kept, D = cache[key]
    out = []
    for i, j in itertools.combinations(range(len(groups)), 2):
        A, B = groups[i], groups[j]
        ks = stats.ks_2samp(resid[A], resid[B]).statistic
        out.append({"cluster_i": kept[i], "cluster_j": kept[j], "n_A": len(A), "n_B": len(B),
                    "ks": float(ks), "D_AB": float(D[(i, j)]), "n_clusters_used": len(groups)})
    return out


def run_pairs(which, out_path, xgb_jobs=1):
    t0 = time.time()
    rows = []
    if xgb_jobs != 1:
        # The original diagnostic run (30 Oct 2026, 14:11) produced its gradient-boosting forecasts before
        # forecasters.rolling_xgboost was changed to n_jobs=1 (30 Oct, 16:53).
        # Setting the thread count here reproduces those forecasts, so that the only change is the level.
        import functools
        import forecasters as fc
        fc.rolling_xgboost = functools.partial(fc.rolling_xgboost, n_jobs=xgb_jobs)
    for name, cfg in ds.REAL_DATASETS.items():
        if name not in which:
            continue
        meta = cfg["load"]()
        y = meta["y"]
        preds = cfg["forecasts"](y, meta, with_xgb=True)
        print(f"{name}: forecasts done ({time.time() - t0:.0f} s)", flush=True)
        for fw in cfg["fws"]:
            feats = spectral_feature_matrix(y, fw)
            cache = {}
            for fname, pred in preds.items():
                resid = np.abs(y - pred)
                idx = np.arange(cfg["start"], len(y))
                idx = idx[np.isfinite(resid[idx])]
                for r in pair_rows(feats[idx], resid[idx], cache):
                    rows.append({"dataset": name, "forecaster": fname, "feature_window": fw, "n": len(idx), **r})
                print(f"{name} {fname} fw={fw}: pairs done ({time.time() - t0:.0f} s)", flush=True)
    if "sims" in which:
        for scen in SIM_SCENARIOS:
            sim = ds.make_sim(scen, seed=0)
            y = sim["y"]
            preds = ds.sim_forecasts(y)
            feats = spectral_feature_matrix(y, 64)
            cache = {}
            for fname, pred in preds.items():
                resid = np.abs(y - pred)
                idx = np.arange(ds.SIM_CFG["start"], len(y))
                idx = idx[np.isfinite(resid[idx])]
                for r in pair_rows(feats[idx], resid[idx], cache):
                    rows.append({"dataset": f"sim_{scen}", "forecaster": fname, "feature_window": 64, "n": len(idx), **r})
                print(f"sim_{scen} {fname}: pairs done ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(rows)} pair rows, {time.time() - t0:.0f} s)", flush=True)


# ----------------------------------------------------------------------------
# stage 2: recompute at the per-set level eta and at eta / J
# ----------------------------------------------------------------------------

def _row_stats(p, delta, level):
    """L_min and the number of pairs with a positive numerator at per-set level `level` (as in the original)."""
    ratios = []
    for ks, D, nA, nB in zip(p["ks"], p["D_AB"], p["n_A"], p["n_B"]):
        num = ks - delta - grid_term(nA, level) - grid_term(nB, level)
        ratios.append(max(0.0, num) / D if D > 1e-12 else 0.0)
    ratios = np.asarray(ratios)
    return (float(ratios.max()) if len(ratios) else np.nan), int(np.sum(ratios > 0))


def recompute(pairs):
    rows = []
    for (dname, fname, fw), g in pairs.groupby(KEYS, sort=False):
        J = int(g["n_clusters_used"].iloc[0])
        n = int(g["n"].iloc[0])
        med = float(np.median(g["D_AB"].to_numpy()))
        p = {c: g[c].to_numpy() for c in ["ks", "D_AB", "n_A", "n_B"]}
        for delta in DELTAS:
            for eta in ETAS:
                eta_adj = eta / J
                l0, r0 = _row_stats(p, delta, eta)
                l1, r1 = _row_stats(p, delta, eta_adj)
                rows.append({"dataset": dname, "forecaster": fname, "feature_window": fw, "n": n,
                             "delta": delta, "eta": eta, "J": J, "eta_adj": eta_adj,
                             "L_min_orig": l0, "L_min_bonf": l1,
                             "n_pairs_refuting_L0_orig": r0, "n_pairs_refuting_L0_bonf": r1,
                             "n_pairs": len(g), "n_clusters_used": J, "median_D_AB": med})
    return pd.DataFrame(rows)


def _assemble(new, base):
    """Final column layout: the eleven columns of `base` first, then the correction (J, eta_adj, *_orig, *_bonf, and the
    counts at delta = 0.1, repeated on every row of the (dataset, forecaster, feature_window, eta) group)."""
    new = new.copy()
    d10 = new[np.isclose(new["delta"], 0.1)].set_index(KEYS + ["eta"])
    for tag in ("orig", "bonf"):
        look = d10[f"n_pairs_refuting_L0_{tag}"].to_dict()
        new[f"n_pairs_refuting_delta0p1_{tag}"] = [look[(a, b, c, e)] for a, b, c, e in
                                                    zip(new["dataset"], new["forecaster"], new["feature_window"], new["eta"])]
    out = pd.DataFrame({c: base[c].to_numpy() for c in base.columns})
    for c in ["J", "eta_adj", "L_min_orig", "L_min_bonf", "n_pairs_refuting_L0_orig", "n_pairs_refuting_L0_bonf",
              "n_pairs_refuting_delta0p1_orig", "n_pairs_refuting_delta0p1_bonf"]:
        out[c] = new[c].to_numpy()
    assert (out["J"].to_numpy() == out["n_clusters_used"].to_numpy()).all()
    return out


def build():
    parts = sorted(RESULTS.glob("assumption_diagnostic_pairs.*.csv"))
    if not parts:
        sys.exit("no per-pair part files found")
    pairs = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    orig = pd.read_csv(RESULTS / "assumption_diagnostic.csv")
    # order the pair rows like the original file so the groups come out in the same order
    order = {k: i for i, k in enumerate(dict.fromkeys(zip(orig["dataset"], orig["forecaster"], orig["feature_window"])))}
    pairs["_o"] = [order[k] for k in zip(pairs["dataset"], pairs["forecaster"], pairs["feature_window"])]
    pairs = pairs.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    new = recompute(pairs)

    # reproduction check against the original file
    assert len(new) == len(orig), (len(new), len(orig))
    for c in ["dataset", "forecaster", "feature_window", "n", "delta", "eta", "n_pairs", "n_clusters_used"]:
        assert (new[c].to_numpy() == orig[c].to_numpy()).all(), f"column {c} differs from the original file"
    assert (new["n_pairs_refuting_L0_orig"].to_numpy() == orig["n_pairs_refuting_L0"].to_numpy()).all(), "refuting counts differ"
    max_dl = float(np.max(np.abs(new["L_min_orig"].to_numpy() - orig["L_min"].to_numpy())))
    max_dm = float(np.max(np.abs(new["median_D_AB"].to_numpy() - orig["median_D_AB"].to_numpy())))
    print(f"reproduction of the original rows: {len(orig)} rows, max |L_min diff| = {max_dl:.3e}, max |median_D_AB diff| = {max_dm:.3e}")
    assert max_dl <= 1e-9 and max_dm <= 1e-9, "original L_min values are not reproduced to 1e-9"

    out = _assemble(new, base=orig)
    out.to_csv(RESULTS / "assumption_diagnostic_bonf.csv", index=False)
    pairs.to_csv(RESULTS / "assumption_diagnostic_pairs.csv", index=False)
    for p in parts:
        p.unlink()
    print(f"wrote assumption_diagnostic_pairs.csv ({len(pairs)} rows) and assumption_diagnostic_bonf.csv ({len(out)} rows); removed {len(parts)} part files")


def onethread():
    """The file the article reads: the same correction with the gradient-boosting forecasts of the CURRENT code (n_jobs = 1,
    forecasters.rolling_xgboost default), the setting of the boosting rows reported elsewhere in the paper and of this repository.  Writes results/assumption_diagnostic_bonf_xgb1thread.csv (the file the article reads) and results/assumption_diagnostic_pairs_xgb1thread.csv.
    Input: part files assumption_diagnostic_pairs_n1.*.csv (written by `all`), or, when none exist, the stored
    assumption_diagnostic_pairs_xgb1thread.csv.
    The first eleven columns have the layout of assumption_diagnostic.csv but hold the values recomputed here with n_jobs = 1
    (L_min = L_min_orig).  Naive and ridge rows are identical to the n_jobs = 4 file; only gradient-boosting rows differ.
    The structural columns (n, J, n_pairs) are checked against assumption_diagnostic.csv; the L_min difference to that
    file is printed (it is zero where that file was produced with n_jobs = 1, and it is not zero for the stored file, whose
    boosting rows came from n_jobs = 4)."""
    parts = sorted(RESULTS.glob("assumption_diagnostic_pairs_n1.*.csv"))
    if parts:
        pairs = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    else:
        pairs = pd.read_csv(RESULTS / "assumption_diagnostic_pairs_xgb1thread.csv")
    orig = pd.read_csv(RESULTS / "assumption_diagnostic.csv")
    order = {k: i for i, k in enumerate(dict.fromkeys(zip(orig["dataset"], orig["forecaster"], orig["feature_window"])))}
    pairs["_o"] = [order[k] for k in zip(pairs["dataset"], pairs["forecaster"], pairs["feature_window"])]
    pairs = pairs.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    new = recompute(pairs)
    assert len(new) == len(orig), (len(new), len(orig))
    for c in ["dataset", "forecaster", "feature_window", "n", "delta", "eta", "n_pairs", "n_clusters_used"]:
        assert (new[c].to_numpy() == orig[c].to_numpy()).all(), f"column {c} differs from the original file"
    nx = new["forecaster"].to_numpy() != "xgb"
    d_all = np.abs(new["L_min_orig"].to_numpy() - orig["L_min"].to_numpy())
    print(f"difference to assumption_diagnostic.csv: max |L_min| over non-boosting rows = {d_all[nx].max():.2e}; "
          f"over boosting rows = {d_all[~nx].max():.2e} (non-zero where the stored file used n_jobs = 4)")
    assert d_all[nx].max() <= 1e-9, "non-boosting rows should reproduce the stored file"
    out = _assemble(new, base=new.assign(L_min=new["L_min_orig"], n_pairs_refuting_L0=new["n_pairs_refuting_L0_orig"])[list(orig.columns)])
    out.to_csv(RESULTS / "assumption_diagnostic_bonf_xgb1thread.csv", index=False)
    pairs.to_csv(RESULTS / "assumption_diagnostic_pairs_xgb1thread.csv", index=False)
    for p in parts:
        p.unlink()
    print(f"wrote assumption_diagnostic_bonf_xgb1thread.csv ({len(out)} rows) and assumption_diagnostic_pairs_xgb1thread.csv ({len(pairs)} rows)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("pairs")
    a.add_argument("--tag", required=True)
    a.add_argument("--datasets", nargs="+", required=True, help="electricity weather finance sims")
    a.add_argument("--xgb-jobs", type=int, default=1, help="n_jobs of the gradient-boosting forecaster (default: current code, 1)")
    sub.add_parser("build")
    sub.add_parser("onethread")
    sub.add_parser("all", help="pairs for every dataset and the simulated streams with the current code (n_jobs = 1), then onethread (used by run_all.py)")
    args = ap.parse_args()
    if args.stage == "pairs":
        run_pairs(set(args.datasets), RESULTS / f"assumption_diagnostic_pairs.{args.tag}.csv", args.xgb_jobs)
    elif args.stage == "build":
        build()
    elif args.stage == "all":
        run_pairs({"electricity", "weather", "finance", "sims"}, RESULTS / "assumption_diagnostic_pairs_n1.all.csv")
        onethread()
    else:
        onethread()


if __name__ == "__main__":
    main()
