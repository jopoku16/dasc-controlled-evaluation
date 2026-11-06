"""
lead4_run_grid.py: pre-stated grid for the regime-wise coverage lead.

Grid (fixed in advance): gamma in {0.005, 0.01, 0.02, 0.05}, m in {50, 150, 450},
sigma_2/sigma_1 in {1.5, 2, 3}, alpha_0 = 0.1, equal blocks, pool = 50/50 mixture.

For every cell the script computes
  F1  explicit linearised closed form,
  F2  periodic steady state of the averaged ODE,
  F3  Gaussian moment closure,
  F4  exact periodic law of the chain (numerical benchmark),
  MC  Monte Carlo of the actual recursion (>= 200 replications; 400 used),
and the two trivial predictors that need no theory:
  T_inf  miss_r = alpha_0 in both regimes (perfect regime-wise control, gamma m -> infinity)
  T_zero miss_r = p_r(q(alpha_0)) (frozen level, gamma m -> 0).

Monte Carlo horizon: burn-in = max(30/gamma, 20 periods) steps (rounded up to whole
periods), measurement window = max(100 periods, 40/gamma) steps (rounded up to whole
periods), 400 replications; period = 2m.  Steady state is checked by comparing the
regime-wise miss rates of the first and second half of the measurement window.

Output: results/lead4_cells.csv  (one row per cell).  Uses 2 worker processes.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lead4_model as M  # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
GAMMAS = [0.005, 0.01, 0.02, 0.05]
MS = [50, 150, 450]
RHOS = [1.5, 2.0, 3.0]
ALPHA0 = 0.1
REPS = 400


def mc_cell(pool, gamma, m, reps=REPS, seed=0):
    a0 = pool.alpha0
    n = int(round(1 / a0))
    unit = gamma * a0
    kmin = int(np.floor((-gamma - a0) / unit)) - 2
    kmax = int(np.ceil((1 + gamma - a0) / unit)) + 2
    ks = np.arange(kmin, kmax + 1)
    alpha_k = a0 + unit * ks
    P = [pool.p_exact(0, alpha_k), pool.p_exact(1, alpha_k)]
    period = 2 * m
    burn_periods = int(np.ceil(max(30.0 / gamma, 20 * period) / period))
    meas_periods = int(np.ceil(max(100 * period, 40.0 / gamma) / period))
    if meas_periods % 2:
        meas_periods += 1
    rng = np.random.default_rng(seed)
    k = np.zeros(reps, dtype=np.int64)
    off = -kmin
    chunk = 2 * m
    miss_sum = np.zeros((2, 2, reps))      # [half][regime][rep]
    start_sum = 0.0
    end_sum = 0.0
    count_blocks = 0
    total_periods = burn_periods + meas_periods
    for per in range(total_periods):
        measuring = per >= burn_periods
        half = 0 if per < burn_periods + meas_periods // 2 else 1
        U = rng.random((period, reps))
        for r in (0, 1):
            if measuring and r == 0:
                start_sum += float(alpha_k[k + off].mean())
            Pr = P[r]
            for t in range(m):
                idx = k + off
                miss = U[r * m + t] < Pr[idx]
                if measuring:
                    miss_sum[half, r] += miss
                k = k + 1 - n * miss
            if measuring and r == 0:
                end_sum += float(alpha_k[k + off].mean())
        if measuring:
            count_blocks += 1
    steps_half = (meas_periods // 2) * m
    per_rep = miss_sum / steps_half                   # [half][regime][rep]
    full = per_rep.mean(axis=0)                       # [regime][rep]
    out = {
        "mc_miss1": float(full[0].mean()), "mc_miss2": float(full[1].mean()),
        "mc_se1": float(full[0].std(ddof=1) / np.sqrt(reps)),
        "mc_se2": float(full[1].std(ddof=1) / np.sqrt(reps)),
        "mc_h1_miss1": float(per_rep[0, 0].mean()), "mc_h2_miss1": float(per_rep[1, 0].mean()),
        "mc_h1_miss2": float(per_rep[0, 1].mean()), "mc_h2_miss2": float(per_rep[1, 1].mean()),
        "mc_alpha_start": start_sum / count_blocks, "mc_alpha_end": end_sum / count_blocks,
        "mc_burn_steps": burn_periods * period, "mc_meas_steps": meas_periods * period, "mc_reps": reps,
    }
    return out


def one_cell(args):
    gamma, m, rho = args
    t0 = time.time()
    pool = M.Pool((1.0, rho), (0.5, 0.5), ALPHA0)
    q0, p0, s0 = pool.zero_order()
    fp = pool.fixed_points()
    f1 = M.f1_linear(pool, gamma, m)
    f2 = M.f2_ode(pool, gamma, m)
    f3 = M.f3_closure(pool, gamma, m)
    f4 = M.f4_exact_chain(pool, gamma, m)
    mc = mc_cell(pool, gamma, m, seed=int(1e6 * gamma) + 7 * m + int(10 * rho))
    row = {
        "gamma": gamma, "m": m, "rho": rho, "gamma_m": gamma * m, "alpha0": ALPHA0,
        "q0": q0, "p1_frozen": p0[0], "p2_frozen": p0[1], "s1": s0[0], "s2": s0[1],
        "alpha1_star": fp[0], "alpha2_star": fp[1],
        "F1_miss1": f1[0], "F1_miss2": f1[1],
        "F2_miss1": f2[0], "F2_miss2": f2[1], "F2_alpha_start": f2[2], "F2_alpha_end": f2[3],
        "F3_miss1": f3[0], "F3_miss2": f3[1], "F3_alpha_start": f3[2], "F3_var_start": f3[3],
        "F4_miss1": f4[0], "F4_miss2": f4[1], "F4_alpha_start": f4[2], "F4_alpha_end": f4[3],
        "F4_edge_mass": f4[4], "F4_cycles": f4[5], "F4_lost_mass": f4[6],
        "T_inf_miss1": ALPHA0, "T_inf_miss2": ALPHA0,
        "T_zero_miss1": p0[0], "T_zero_miss2": p0[1],
        "fixed_point_bound_dev": (fp[0] - fp[1]) / (gamma * m),
        "seconds": time.time() - t0,
    }
    row.update(mc)
    return row


def main():
    cells = list(itertools.product(GAMMAS, MS, RHOS))
    t0 = time.time()
    with Pool(2) as pool:
        rows = []
        for i, r in enumerate(pool.imap(one_cell, cells)):
            rows.append(r)
            print(f"[{i + 1}/{len(cells)}] gamma={r['gamma']} m={r['m']} rho={r['rho']} "
                  f"MC=({r['mc_miss1']:.4f},{r['mc_miss2']:.4f}) F2=({r['F2_miss1']:.4f},{r['F2_miss2']:.4f}) "
                  f"F4=({r['F4_miss1']:.4f},{r['F4_miss2']:.4f}) {r['seconds']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    # errors in regime-wise coverage (= errors in miss rates)
    for tag in ["F1", "F2", "F3", "F4", "T_inf", "T_zero"]:
        df[f"err_{tag}"] = np.maximum((df[f"{tag}_miss1"] - df["mc_miss1"]).abs(),
                                      (df[f"{tag}_miss2"] - df["mc_miss2"]).abs())
    df["effect_size"] = np.maximum((df["mc_miss1"] - ALPHA0).abs(), (df["mc_miss2"] - ALPHA0).abs())
    df["mc_halves_diff"] = np.maximum((df["mc_h1_miss1"] - df["mc_h2_miss1"]).abs(),
                                      (df["mc_h1_miss2"] - df["mc_h2_miss2"]).abs())
    out = os.path.join(RES, "lead4_cells.csv")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite")
    df.to_csv(out, index=False)
    print("total seconds", time.time() - t0)


if __name__ == "__main__":
    main()
