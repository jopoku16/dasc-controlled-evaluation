"""
lead4_analyze.py: grade the pre-stated criterion (a) from results/lead4_cells.csv.

Criterion (a): formulas agree with the Monte Carlo of the actual recursion within
0.01 in regime-wise coverage in at least 80 per cent of the 36 cells.

Writes results/lead4_summary.csv (one row per formula) and results/lead4_failures.csv
(cells where a formula misses by more than 0.01) and prints a short table.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
TOL = 0.01

df = pd.read_csv(os.path.join(RES, "lead4_cells.csv"))
n = len(df)
rows = []
labels = {
    "F1": "F1 linearised closed form",
    "F2": "F2 averaged ODE (periodic steady state)",
    "F3": "F3 Gaussian moment closure",
    "F4": "F4 exact lattice chain (benchmark, not graded)",
    "T_inf": "T_inf trivial: miss = alpha_0 in both regimes",
    "T_zero": "T_zero trivial: frozen pooled quantile",
}
for tag, lab in labels.items():
    e = df[f"err_{tag}"]
    rows.append({"formula": lab, "cells_within_0.01": int((e <= TOL).sum()), "cells": n,
                 "share_within_0.01": float((e <= TOL).mean()),
                 "median_abs_err": float(e.median()), "max_abs_err": float(e.max())})
# informative cells: where the effect to be explained is larger than the tolerance
inf = df["effect_size"] > TOL
print("cells where |MC miss - alpha_0| > 0.01 in some regime (effect larger than tolerance):",
      int(inf.sum()), "of", n)
for tag in ["F1", "F2", "F3", "F4", "T_inf", "T_zero"]:
    e = df.loc[inf, f"err_{tag}"]
    rows.append({"formula": labels[tag] + " [informative cells only]",
                 "cells_within_0.01": int((e <= TOL).sum()), "cells": int(inf.sum()),
                 "share_within_0.01": float((e <= TOL).mean()), "median_abs_err": float(e.median()),
                 "max_abs_err": float(e.max())})
# share of the regime gap explained (informative cells)
for tag in ["F1", "F2", "F3", "T_inf", "T_zero"]:
    gap_mc = df["mc_miss2"] - df["mc_miss1"]
    gap_f = df[f"{tag}_miss2"] - df[f"{tag}_miss1"]
    df[f"gaprat_{tag}"] = gap_f / gap_mc
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(RES, "lead4_summary.csv"), index=False)
fail = df[(df["err_F2"] > TOL) | (df["err_F3"] > TOL) | (df["err_F1"] > TOL)][
    ["gamma", "m", "rho", "gamma_m", "mc_miss1", "mc_miss2", "F1_miss1", "F1_miss2", "F2_miss1", "F2_miss2",
     "F3_miss1", "F3_miss2", "F4_miss1", "F4_miss2", "err_F1", "err_F2", "err_F3", "err_F4"]]
fail.to_csv(os.path.join(RES, "lead4_failures.csv"), index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
print(summ.to_string(index=False))
print("\nMC steady-state check: max |first half - second half| over cells:", df["mc_halves_diff"].max())
print("max MC standard error:", max(df["mc_se1"].max(), df["mc_se2"].max()))
print("max F4 edge mass:", df["F4_edge_mass"].max(), " max lost mass:", df["F4_lost_mass"].max())
print("F2 vs F4 (ODE vs exact chain), max abs diff of miss:",
      float(np.maximum((df.F2_miss1 - df.F4_miss1).abs(), (df.F2_miss2 - df.F4_miss2).abs()).max()))
print("F4 vs MC max abs diff:", df["err_F4"].max())
print("\ncells where F1 fails by > 0.01:")
print(df.loc[df.err_F1 > TOL, ["gamma", "m", "rho", "gamma_m", "mc_miss1", "mc_miss2", "F1_miss1", "F1_miss2", "err_F1"]].to_string(index=False))
print("\ncells where F2 fails by > 0.01:")
print(df.loc[df.err_F2 > TOL, ["gamma", "m", "rho", "mc_miss1", "mc_miss2", "F2_miss1", "F2_miss2", "err_F2"]].to_string(index=False))
print("\ncells where F3 fails by > 0.01:")
print(df.loc[df.err_F3 > TOL, ["gamma", "m", "rho", "mc_miss1", "mc_miss2", "F3_miss1", "F3_miss2", "err_F3"]].to_string(index=False))
# exact identity check I2: miss_1 = alpha_0 - (E alpha_end - E alpha_start)/(gamma m) from the chain and from MC
df["I2_chain"] = 0.1 - (df.F4_alpha_end - df.F4_alpha_start) / (df.gamma * df.m)
df["I2_mc"] = 0.1 - (df.mc_alpha_end - df.mc_alpha_start) / (df.gamma * df.m)
print("\nIdentity I2 (chain): max |F4_miss1 - (alpha_0 - swing/(gamma m))| =",
      float((df.F4_miss1 - df.I2_chain).abs().max()))
print("Identity I2 (MC): max |mc_miss1 - (alpha_0 - swing/(gamma m))| =",
      float((df.mc_miss1 - df.I2_mc).abs().max()))
print("Symmetry: max |(F4_miss1 + F4_miss2)/2 - alpha_0| =", float(((df.F4_miss1 + df.F4_miss2) / 2 - 0.1).abs().max()))
# fixed-point bound: |miss_r - alpha_0| <= (alpha1* - alpha2*)/(gamma m) for the ODE, check on MC
dev_mc = np.maximum((df.mc_miss1 - 0.1).abs(), (df.mc_miss2 - 0.1).abs())
print("Fixed-point bound violated in MC (dev > bound + 3 SE) in cells:",
      int((dev_mc > df.fixed_point_bound_dev + 3 * np.maximum(df.mc_se1, df.mc_se2)).sum()))
# tightness: dev/bound for gamma_m >= 5
big = df.gamma_m >= 5
print("tightness dev/bound for gamma_m >= 5:", (dev_mc[big] / df.fixed_point_bound_dev[big]).describe()[["min", "50%", "max"]].to_dict())
df.to_csv(os.path.join(RES, "lead4_cells_with_checks.csv"), index=False)
