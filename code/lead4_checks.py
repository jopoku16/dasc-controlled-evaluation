"""
lead4_checks.py: numerical checks of the individual derivation steps (one line per step).

C1  pooled-mixture identity (I1): sum_r pi_r p_r(alpha) = alpha on a grid, three pi.
C2  slope identity (I3): sum_r pi_r s_r(alpha) = 1, and s_r equals a finite difference of p_r.
C3  lattice property: every alpha reached by the simulated recursion is alpha_0 + gamma alpha_0 k.
C4  linearisation error (F1 vs F4) against the size of the swing, per cell, from lead4_cells.csv.
C5  ODE error (F2 vs F4) and closure error (F3 vs F4), per cell.
C6  weighted-quantile first-order formula versus exact root, over w.
Writes results/lead4_derivation_checks.csv (C4, C5 per cell) and prints C1, C2, C3, C6.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(HERE), "results")
import lead4_model as M  # noqa: E402


def main():
    # C1, C2
    worst1 = worst2 = worst_fd = 0.0
    for pi1 in (0.3, 0.5, 0.8):
        for rho in (1.5, 3.0):
            pool = M.Pool((1.0, rho), (pi1, 1 - pi1))
            a = np.linspace(0.002, 0.998, 400)
            p0, p1 = pool.p_exact(0, a), pool.p_exact(1, a)
            worst1 = max(worst1, float(np.abs(pi1 * p0 + (1 - pi1) * p1 - a).max()))
            s0, s1 = pool.slope(0, a), pool.slope(1, a)
            worst2 = max(worst2, float(np.abs(pi1 * s0 + (1 - pi1) * s1 - 1).max()))
            h = 1e-6
            fd = (pool.p_exact(0, a + h) - pool.p_exact(0, a - h)) / (2 * h)
            worst_fd = max(worst_fd, float((np.abs(fd - s0) / np.maximum(s0, 1e-3)).max()))
    print(f"C1 identity (I1) max |sum pi p - alpha| over grid, 3 weights x 2 ratios: {worst1:.2e}")
    print(f"C2 identity (I3) max |sum pi s - 1|: {worst2:.2e}; analytic slope vs central difference, max relative gap: {worst_fd:.2e}")

    # C3 lattice
    rng = np.random.default_rng(3)
    gamma, a0 = 0.02, 0.1
    pool = M.Pool((1.0, 2.0))
    alpha = a0
    ks = []
    for t in range(20000):
        r = (t // 100) % 2
        p = float(pool.p_exact(r, alpha)[0])
        miss = rng.random() < p
        alpha = alpha + gamma * (a0 - miss)
        ks.append((alpha - a0) / (gamma * a0))
    ks = np.array(ks)
    print(f"C3 lattice: max distance of (alpha_t - alpha_0)/(gamma alpha_0) from an integer over 20000 steps: "
          f"{float(np.abs(ks - np.round(ks)).max()):.2e}")

    # C4, C5
    df = pd.read_csv(os.path.join(RES, "lead4_cells.csv"))
    out = pd.DataFrame({
        "gamma": df.gamma, "m": df.m, "rho": df.rho, "gamma_m": df.gamma_m,
        "swing_F4": df.F4_alpha_end - df.F4_alpha_start,
        "F1_minus_F4": np.maximum((df.F1_miss1 - df.F4_miss1).abs(), (df.F1_miss2 - df.F4_miss2).abs()),
        "F2_minus_F4": np.maximum((df.F2_miss1 - df.F4_miss1).abs(), (df.F2_miss2 - df.F4_miss2).abs()),
        "F3_minus_F4": np.maximum((df.F3_miss1 - df.F4_miss1).abs(), (df.F3_miss2 - df.F4_miss2).abs()),
        "F3_var_start": df.F3_var_start})
    out.to_csv(os.path.join(RES, "lead4_derivation_checks.csv"), index=False)
    print("C4 linearisation error F1-F4: max by gamma*m:")
    print(out.groupby("gamma_m")["F1_minus_F4"].max().round(4).to_string())
    print("C5 ODE error F2-F4: max", round(out.F2_minus_F4.max(), 5), " median", round(out.F2_minus_F4.median(), 5),
          "| closure error F3-F4: max", round(out.F3_minus_F4.max(), 6), " median", round(out.F3_minus_F4.median(), 6))
    print("C5b largest ODE error occurs at:", out.loc[out.F2_minus_F4.idxmax(), ["gamma", "m", "rho"]].to_dict(),
          "; stationary sd of alpha there (from closure):", float(np.sqrt(out.loc[out.F2_minus_F4.idxmax(), "F3_var_start"])))

    # C6
    print("C6 weighted quantile: first order versus exact (rho, w, miss_1 exact / linear, miss_2 exact / linear):")
    for rho in (1.5, 2.0, 3.0):
        pool = M.Pool((1.0, rho))
        for w in (0.6, 0.75, 0.9):
            e = M.weighted_miss(pool, w)
            l = M.weighted_miss_linear(pool, w)
            print(f"   rho={rho} w={w}: {e[0]:.5f} / {l[0]:.5f}   {e[1]:.5f} / {l[1]:.5f}")


if __name__ == "__main__":
    main()
