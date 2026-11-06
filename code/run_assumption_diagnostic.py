"""
run_assumption_diagnostic.py: refutation test of the Lipschitz regime
assumption on the real data and on the simulated streams.

For feature-measurable index sets A and B (k-means clusters of the spectral
features), the assumption with constants (L, delta) and conditional
independence imply, with probability at least 1 - 2 eta,
    KS(Fhat_A, Fhat_B) <= L D_AB + delta + eps_A(eta) + eps_B(eta),
where D_AB is the mean pairwise feature distance between A and B and eps is
the uniform grid term sqrt(log(4K/eta)/(2n)) + 1/K with K = max(2, ceil(n)).
The diagnostic
    L_min(delta, eta) = max over pairs of (KS_AB - delta - eps_A - eps_B)^+ / D_AB
is a lower confidence bound on any admissible L only at the simultaneous level
1 - J eta, J = number of clusters used (each cluster's concentration event fails
with probability at most eta; the maximum over pairs uses all J events).  At the
per-set levels read here (eta = 0.05, 0.10) with J = 11 to 20 that level is nil, so
the values in results/assumption_diagnostic.csv are per-set-level values, not
confidence bounds.  The corrected reading, at the per-set level eta / J, is made by
run_assumption_diagnostic_bonf.py (results/assumption_diagnostic_bonf.csv); the article reports that file.  It is a refutation device:
every pair with a positive numerator refutes (L, delta) for all L below its
ratio; the pair attaining the maximum sets L_min.  It is not an estimate of L.
The conditional-independence condition behind the eps terms does not hold for
serially dependent residuals, so on real data the test is indicative only.

Gradient-boosting setting of the stored file: results/assumption_diagnostic.csv was produced on 30 Oct 2026 with the
boosting forecaster at n_jobs = 4 (the code default then).  forecasters.rolling_xgboost now defaults to n_jobs = 1, so
a rerun of this script gives different boosting rows (naive and ridge rows are unchanged).  The stored file is kept as
the record; the article reports results/assumption_diagnostic_bonf_xgb1thread.csv (n_jobs = 1), see
run_assumption_diagnostic_bonf.py.

Writes ../results/assumption_diagnostic.csv with, per dataset, forecaster,
feature window, delta and eta: L_min, the number of cluster pairs with a
positive numerator (pairs refuting L = 0 at that delta), the number of pairs,
the number of clusters used, and the median D_AB.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

import datasets as ds
from dasc_lib import grid_term, spectral_feature_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"
DELTAS = (0.0, 0.05, 0.10)
ETAS = (0.05, 0.10)
N_CLUSTERS = 20
MIN_PER = 30


def diagnostic(feats, resid, n_clusters=N_CLUSTERS, min_per=MIN_PER, seed=42):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(feats)
    groups = [np.where(km.labels_ == c)[0] for c in range(n_clusters)]
    groups = [g for g in groups if len(g) >= min_per]
    pairs = []
    for i, j in itertools.combinations(range(len(groups)), 2):
        A, B = groups[i], groups[j]
        ks = stats.ks_2samp(resid[A], resid[B]).statistic
        # mean pairwise distance, computed in chunks to bound memory
        D = 0.0
        for a0 in range(0, len(A), 500):
            fa = feats[A[a0:a0 + 500]]
            D += np.linalg.norm(fa[:, None, :] - feats[B][None, :, :], axis=2).sum()
        D /= (len(A) * len(B))
        pairs.append((ks, D, len(A), len(B)))
    rows = []
    for delta in DELTAS:
        for eta in ETAS:
            ratios = []
            for ks, D, nA, nB in pairs:
                num = ks - delta - grid_term(nA, eta) - grid_term(nB, eta)
                ratios.append(max(0.0, num) / D if D > 1e-12 else 0.0)
            ratios = np.asarray(ratios)
            rows.append({"delta": delta, "eta": eta, "L_min": float(ratios.max()) if len(ratios) else np.nan,
                         "n_pairs_refuting_L0": int(np.sum(ratios > 0)), "n_pairs": len(pairs),
                         "n_clusters_used": len(groups),
                         "median_D_AB": float(np.median([p[1] for p in pairs])) if pairs else np.nan})
    return rows


def main():
    t0 = time.time()
    rows = []
    for name, cfg in ds.REAL_DATASETS.items():
        meta = cfg["load"]()
        y = meta["y"]
        preds = cfg["forecasts"](y, meta, with_xgb=True)
        for fw in cfg["fws"]:
            feats = spectral_feature_matrix(y, fw)
            for fname, pred in preds.items():
                resid = np.abs(y - pred)
                idx = np.arange(cfg["start"], len(y))
                idx = idx[np.isfinite(resid[idx])]
                for r in diagnostic(feats[idx], resid[idx]):
                    rows.append({"dataset": name, "forecaster": fname, "feature_window": fw, "n": len(idx), **r})
                print(f"{name} {fname} fw={fw}: done ({time.time() - t0:.0f} s)", flush=True)
    for scen in ["original_three_phase", "recurring_freq_noise", "recurring_amplitude", "white_noise"]:
        sim = ds.make_sim(scen, seed=0)
        y = sim["y"]
        preds = ds.sim_forecasts(y)
        feats = spectral_feature_matrix(y, 64)
        for fname, pred in preds.items():
            resid = np.abs(y - pred)
            idx = np.arange(ds.SIM_CFG["start"], len(y))
            idx = idx[np.isfinite(resid[idx])]
            for r in diagnostic(feats[idx], resid[idx]):
                rows.append({"dataset": f"sim_{scen}", "forecaster": fname, "feature_window": 64, "n": len(idx), **r})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "assumption_diagnostic.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[(df["eta"] == 0.05)].to_string(index=False))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
