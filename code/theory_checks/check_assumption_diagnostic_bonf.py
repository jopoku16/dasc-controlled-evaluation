"""
Bonferroni version of the check of the assumption diagnostic (check_assumption_diagnostic.py).

The pair bound of the article: for two index sets A and B the statement holds with probability
at least 1 - 2 eta, because each set's concentration event fails with probability at most eta.
The diagnostic L_min(delta, eta) is a maximum over all pairs of J retained clusters, so it uses all
J set events at once; the union bound gives L_min(delta, eta) <= L with probability at least
1 - J eta.  check_assumption_diagnostic.py reads the bound at eta = 0.05 and 0.2 for each set,
which is a simultaneous level of 1 - J eta (nil for the J = 7 to 17 clusters found).  This script
evaluates the same statistic at the per-set level eta / J, so that the simultaneous statement has
probability at least 1 - eta.

It rebuilds models A and B1 with the same seeds, recomputes L_min at eta (original) and at eta / J,
checks that the original L_min values are reproduced to 1e-9 against outputs/assumption_diagnostic.csv,
and writes outputs/assumption_diagnostic_bonf.csv.  L_path (an O(N^2) maximum that does not depend on eta)
and the old regression-slope columns are taken from the original output file, since the rebuilt features
and residuals are verified against it through L_min_diag and the cluster count.

The original check_assumption_diagnostic.py and its output are left unchanged.

Run:  python -X utf8 check_assumption_diagnostic_bonf.py [--seeds 10]
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

from check_assumption_diagnostic import build
from dasc_causal import grid_term

HERE = Path(__file__).resolve().parent


def lmin_both(feats, resid, eta, n_clusters=20, min_per=30, seed=42):
    """L_min(0, eta) at the per-set level eta and at eta / J, J = number of retained clusters (same arithmetic as
    check_assumption_diagnostic.diagnostic)."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(feats)
    groups = [np.where(km.labels_ == c)[0] for c in range(n_clusters)]
    groups = [g for g in groups if len(g) >= min_per]
    J = len(groups)
    pairs = []
    for i, j in itertools.combinations(range(J), 2):
        A, B = groups[i], groups[j]
        ks = stats.ks_2samp(resid[A], resid[B]).statistic
        D = np.mean(np.linalg.norm(feats[A][:, None, :] - feats[B][None, :, :], axis=2))
        pairs.append((ks, D, len(A), len(B)))
    out = []
    for level in (eta, eta / J):
        lmin = 0.0
        for ks, D, nA, nB in pairs:
            lmin = max(lmin, max(0.0, ks - grid_term(nA, level) - grid_term(nB, level)) / D)
        out.append(lmin)
    return out[0], out[1], J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    orig = pd.read_csv(HERE / "outputs" / "assumption_diagnostic.csv")
    rows = []
    for model, exo in [("A_exogenous", True), ("B1_endog_phi0.95", False)]:
        for seed in range(args.seeds):
            feats, sig, resid = build(seed, exo)
            for eta in [0.05, 0.2]:
                o = orig[(orig.model == model) & (orig.seed == seed) & np.isclose(orig.eta, eta)].iloc[0]
                l0, l1, J = lmin_both(feats, resid, eta)
                assert abs(l0 - o["L_min_diag"]) <= 1e-9, (model, seed, eta, l0, o["L_min_diag"])
                assert J == int(o["clusters"]), (model, seed, eta, J, o["clusters"])
                L = float(o["L_path"])
                rows.append({"model": model, "seed": seed, "eta": eta, "J": J, "eta_adj": eta / J, "L_path": L,
                             "L_min_diag": float(o["L_min_diag"]), "L_min_diag_bonf": l1,
                             "L_min_le_L": bool(o["L_min_le_L"]), "L_min_le_L_bonf": bool(l1 <= L),
                             "old_slope": float(o["old_slope"]), "old_intercept": float(o["old_intercept"]),
                             "old_R2": float(o["old_R2"]), "clusters": J})
            print(model, seed, "ok", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "outputs" / "assumption_diagnostic_bonf.csv", index=False)
    summ = df.groupby(["model", "eta"]).agg(J_min=("J", "min"), J_max=("J", "max"), L_path=("L_path", "mean"),
                                            L_min_orig_max=("L_min_diag", "max"), L_min_bonf_max=("L_min_diag_bonf", "max"),
                                            frac_le_L_orig=("L_min_le_L", "mean"), frac_le_L_bonf=("L_min_le_L_bonf", "mean")).reset_index()
    text = ("Bonferroni version of the assumption diagnostic check (per-set level eta / J, J = retained clusters).\n"
            "Original L_min_diag values reproduced to 1e-9 for every row.\n\n"
            + summ.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")
    (HERE / "outputs" / "assumption_diagnostic_bonf_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
