"""
lead1_repro_check.py: reproduction check of the EXISTING independent-concentration diagnostic.

Calls run_assumption_diagnostic.diagnostic (read only) on simulated data from lead1_sim and compares its L_min(delta=0,
eta=0.05) with the C0 pipeline of lead1_sim, using the SAME k-means labels.  Also prints the old script's own number
for the paper's setting (its default of 20 clusters, min_per 30).
"""
import numpy as np
from sklearn.cluster import KMeans

import lead1_sim as L
import run_assumption_diagnostic as old   # existing code, read only

rows = []
for n, d, rho in ((2160, 1, 0.0), (2160, 2, 0.5), (360, 1, 0.0)):
    s, sigma, r = L.simulate(np.random.SeedSequence([77, n, d]), n, rho, d, 1.0)
    k = 20 if n == 2160 else 6
    res = old.diagnostic(s, r, n_clusters=k, min_per=30, seed=42)
    lm_old = [x["L_min"] for x in res if x["delta"] == 0.0 and x["eta"] == 0.05][0]
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(s)
    keep = [c for c in range(k) if np.sum(km.labels_ == c) >= 30]
    remap = -np.ones(k, dtype=int)
    for new, c in enumerate(keep):
        remap[c] = new
    lab = remap[km.labels_]
    K = len(keep)
    ia, ib = L.pair_index(K)
    D, groups = L.mean_dist(s, lab, K, ia, ib)
    ks, sizes = L.exact_ks(r, lab, K, ia, ib)
    lm_new = L.lower_bound_dkw(ks, D, sizes, ia, ib, 0.05)
    print(f"n={n} d={d} rho={rho}: old script L_min={lm_old:.6f}  lead1 C0={lm_new:.6f}  diff={abs(lm_old - lm_new):.2e}  clusters={K}")
