"""
ext_neff_by_h.py: how close to uniform are the kernel weights for each bandwidth of the extended grid?

For each dataset, feature window, pool size and bandwidth h (original values and c x median pairwise
distance), on every fifth evaluation step: n_eff = (sum w)^2 / sum w^2 of the kernel weights
w_i = exp(-d_i^2 / h^2) over the full pool (no gate), the ratio n_eff / pool size, and the share of steps at
which every weight underflows (the library then falls back to uniform weights).  Features only; no forecaster.

Writes results/ext_neff_by_h.csv.  Single process.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import datasets as ds
import ext_lib as E
from dasc_lib import spectral_feature_matrix

R = E.RESULTS


def one(F, t_list, window, h_list):
    ratios = {h: [] for h in h_list}
    neffs = {h: [] for h in h_list}
    under = {h: 0 for h in h_list}
    for t in t_list:
        lo = max(1, t - window)
        diff = F[lo:t] - F[t]
        d2 = (diff * diff).sum(axis=1)
        p = len(d2)
        for h in h_list:
            wu = np.exp(-d2 / h ** 2)
            sw = wu.sum()
            if sw == 0:
                under[h] += 1
                ratios[h].append(1.0)
                neffs[h].append(float(p))
                continue
            w = wu / sw                     # normalise first, as the library does
            ne = 1.0 / np.sum(w * w)
            ratios[h].append(ne / p)
            neffs[h].append(ne)
    return ratios, neffs, under


def main(argv):
    names = [a for a in argv if a in ds.REAL_DATASETS] or list(ds.REAL_DATASETS)
    rows = []
    for name in names:
        cfg = ds.REAL_DATASETS[name]
        meta = cfg["load"]()
        y = meta["y"]; n = len(y)
        start, dev_end = cfg["start"], int(cfg["dev_frac"] * n)
        t_list = list(range(dev_end, n, 5))
        for fw in cfg["fws"]:
            F = spectral_feature_matrix(y, fw)
            med = E.median_pairwise(F, start, dev_end)
            hl = E.make_hgrid({fw: med})[fw]
            for window in cfg["pools"]:
                ratios, neffs, under = one(F, t_list, window, hl)
                for h in hl:
                    r = np.array(ratios[h])
                    rows.append({"dataset": name, "fw": fw, "pool": window, "h": h, "median_pairwise": med,
                                 "h_over_median": h / med, "scaled": h not in E.H_ORIG,
                                 "n_steps": len(t_list), "neff_median": float(np.median(neffs[h])),
                                 "neff_over_pool_median": float(np.median(r)),
                                 "neff_over_pool_p10": float(np.quantile(r, 0.10)),
                                 "frac_ratio_gt_090": float(np.mean(r > 0.90)),
                                 "frac_all_weights_underflow": under[h] / len(t_list)})
        print(f"[{name}] done", flush=True)
    pd.DataFrame(rows).to_csv(R / "ext_neff_by_h.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
