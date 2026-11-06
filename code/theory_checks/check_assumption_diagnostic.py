"""
Check of the assumption-checking proposition and of the diagnostic L_min(delta, eta) that
replaces the old estimation proposition.

Proposition: under the Lipschitz regime assumption with constants (L, delta), for any two
feature-measurable index sets A and B,
    sup_r |F_A(r) - F_B(r)| <= L D_AB + delta,
where F_A is the average of the conditional CDFs over A and D_AB the mean pairwise feature
distance between A and B.  With conditional independence, the empirical CDFs satisfy
    KS(Fhat_A, Fhat_B) <= L D_AB + delta + eps_A(eta) + eps_B(eta)
on an event of probability at least 1 - 2 eta, with eps the grid term for uniform weights.
Diagnostic: L_min(delta, eta) = max over pairs of (KS_AB - delta - eps_A - eps_B)^+ / D_AB
is a lower confidence bound on any admissible L at the simultaneous level 1 - J eta only (J = number of
clusters used); eta = 0.05 and 0.2 below are per-set levels, so the values here are not confidence bounds at
those levels.  check_assumption_diagnostic_bonf.py reads the same statistic at eta / J; the article reports that file.  It is NOT an estimate of L.

The script builds model A (assumption exact, L known on the realised path), forms k-means
clusters of the features, and reports L_min(0, eta) against L_path, the old regression
slope (KS on centroid distance, as in the original verification script) and its R^2.
It repeats this in model B1 (conditional independence violated).

Run:  python -X utf8 check_assumption_diagnostic.py [--seeds 10] [--out outputs]
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from check_perstep_bound import driver_stream, ks_half_normal, sigma_of, N, L_WIN
from dasc_causal import all_features, grid_term, spectral_feature


def build(seed, exogenous, phi=0.95, sigma0=0.6, kappa=0.8):
    rng = np.random.default_rng(seed)
    signal, _ = driver_stream(N, rng)
    eps = rng.normal(0, 1, N)
    if exogenous:
        v = signal + rng.normal(0, 0.45, N)
        feats = all_features(v, L_WIN)
        sig = sigma_of(feats, sigma0, kappa)
    else:
        y = np.zeros(N); feats = np.zeros((N, 6)); sig = np.zeros(N)
        for i in range(N):
            feats[i] = spectral_feature(y, i, L_WIN)
            sig[i] = sigma_of(feats[i], sigma0, kappa)
            m = phi * y[i - 1] + signal[i] if i > 0 else signal[i]
            y[i] = m + sig[i] * eps[i]
    resid = sig * np.abs(eps)
    idx = np.arange(64, N)
    return feats[idx], sig[idx], resid[idx]


def l_path(feats, sig):
    best = 0.0
    for j in range(len(feats)):
        d = np.linalg.norm(feats - feats[j], axis=1)
        ks = ks_half_normal(sig, sig[j])
        ok = d > 1e-9
        if ok.any():
            best = max(best, float(np.max(ks[ok] / d[ok])))
    return best


def diagnostic(feats, resid, eta, n_clusters=20, min_per=30, seed=42):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(feats)
    labels = km.labels_
    groups = [np.where(labels == c)[0] for c in range(n_clusters)]
    groups = [g for g in groups if len(g) >= min_per]
    lmin = 0.0
    slopes_x, slopes_y = [], []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            A, B = groups[i], groups[j]
            ks = stats.ks_2samp(resid[A], resid[B]).statistic
            D = np.mean(np.linalg.norm(feats[A][:, None, :] - feats[B][None, :, :], axis=2))
            eA = grid_term(len(A), eta); eB = grid_term(len(B), eta)
            lmin = max(lmin, max(0.0, ks - eA - eB) / D)
            slopes_x.append(np.linalg.norm(km.cluster_centers_[i] - km.cluster_centers_[j])); slopes_y.append(ks)
    sl = stats.linregress(slopes_x, slopes_y)
    return lmin, max(0.0, sl.slope), max(0.0, sl.intercept), sl.rvalue ** 2, len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(__file__).resolve().parent / args.out
    out.mkdir(exist_ok=True)
    rows = []
    for model, exo in [("A_exogenous", True), ("B1_endog_phi0.95", False)]:
        for seed in range(args.seeds):
            feats, sig, resid = build(seed, exo)
            L = l_path(feats, sig)
            for eta in [0.05, 0.2]:
                lmin, slope, intercept, r2, k = diagnostic(feats, resid, eta)
                rows.append({"model": model, "seed": seed, "eta": eta, "L_path": L, "L_min_diag": lmin,
                             "L_min_le_L": lmin <= L, "old_slope": slope, "old_intercept": intercept,
                             "old_R2": r2, "clusters": k})
            print(model, seed, f"L_path={L:.3f}", rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "assumption_diagnostic.csv", index=False)
    summ = df.groupby(["model", "eta"]).agg(L_path=("L_path", "mean"), L_min_diag_mean=("L_min_diag", "mean"),
                                            L_min_diag_max=("L_min_diag", "max"), frac_L_min_le_L=("L_min_le_L", "mean"),
                                            old_slope_mean=("old_slope", "mean"), old_slope_min=("old_slope", "min"),
                                            old_slope_max=("old_slope", "max"), old_R2_mean=("old_R2", "mean")).reset_index()
    text = ("Assumption diagnostic check (model A: assumption exact; B1: conditional independence violated).\n"
            "L_min_diag must be <= L_path with high probability in model A.  The old slope is the regression of\n"
            "cluster-pair KS on centroid distance (the original verification script); it is shown for comparison.\n\n"
            + summ.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")
    (out / "assumption_diagnostic_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
