"""
lead4_link.py: how far the idealised formulas carry to (A) the paper's rolling-window ACI
on a stream with the same block structure, and (B) the paper's actual simulated design
'recurring_freq_noise' (lag-1 forecaster, sinusoid plus noise, blocks of 150).

Nothing in the existing code base is modified; dasc_lib, datasets and forecasters are
imported read-only.

Models used
-----------
Window pool.  The paper's ACI calibrates on the last W = 360 residuals.  At time t the
pool holds a share pi_1(t) of regime-1 residuals that depends on the position of t in the
cycle (for m = 450 the pool is mostly one regime).  The mean-field recursion (MF: variance
set to zero) and the Gaussian moment closure (CL) are run with the table
p_r(alpha; pi) = S_r(q_pi(alpha)), where q_pi solves pi S_1(q) + (1 - pi) S_2(q) = alpha.
The population laws S_r are Gaussian (part A) or the empirical laws of the lag-1 residual
in each regime (part B).

Outputs:
  results/lead4_rolling_cells.csv    36 cells, rolling-window Monte Carlo vs predictions
  results/lead4_scenario_check.csv   ACI in the paper's design: re-run, stored CSV, predictions
  results/lead4_weights.csv          spectral weight mass on the same regime (fixed level)
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import erfc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(HERE), "results")

import lead4_model as M  # noqa: E402

A0 = 0.1
GRID_N = 2001
_u = np.linspace(0.0, 1.0, GRID_N)
AG = 0.5 * (1 - np.cos(np.pi * _u))                 # alpha grid, denser near 0 and 1
PI_GRID = np.linspace(0.0, 1.0, 101)
GHX, GHW = M._GH_X, M._GH_W


# ----------------------------------------------------------------------------
# laws and p-tables
# ----------------------------------------------------------------------------

def gaussian_laws(rho):
    sig = np.array([1.0, rho])
    return [lambda q, s=sig[0]: erfc(np.asarray(q) / (s * np.sqrt(2.0))),
            lambda q, s=sig[1]: erfc(np.asarray(q) / (s * np.sqrt(2.0)))], 14.0 * sig.max()


def empirical_laws(sample1, sample2):
    s1 = np.sort(np.asarray(sample1))
    s2 = np.sort(np.asarray(sample2))

    def mk(s):
        n = len(s)
        return lambda q: 1.0 - np.searchsorted(s, np.asarray(q), side="right") / n
    return [mk(s1), mk(s2)], float(max(s1[-1], s2[-1]))


def build_ptables(laws, qmax):
    """T[r][i_pi, i_alpha] = p_r(alpha; pi) = S_r(q_pi(alpha))."""
    T = np.zeros((2, len(PI_GRID), GRID_N))
    for i, pi in enumerate(PI_GRID):
        lo = np.zeros(GRID_N)
        hi = np.full(GRID_N, qmax)
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            smix = pi * laws[0](mid) + (1 - pi) * laws[1](mid)
            big = smix > AG
            lo = np.where(big, mid, lo)
            hi = np.where(big, hi, mid)
        q = 0.5 * (lo + hi)
        for r in (0, 1):
            p = laws[r](q)
            p = np.where(AG <= 0.0, 0.0, p)
            p = np.where(AG >= 1.0, 1.0, p)
            T[r, i] = p
    return T


def pi_index(pi1):
    return int(round(float(pi1) * (len(PI_GRID) - 1)))


def window_share(t, m, W, first_valid=1):
    """Share of regime-1 residuals among indices [max(first_valid, t-W), t-1] when
    regime(j) = 1 iff (j // m) % 2 == 0."""
    lo = max(first_valid, t - W)
    if t <= lo:
        return 0.5
    j = np.arange(lo, t)
    return float(np.mean(((j // m) % 2) == 0))


# ----------------------------------------------------------------------------
# mean-field and closure recursions on the table
# ----------------------------------------------------------------------------

def step_closure(T, r, ipi, mean, var, gamma, closure=True):
    row = T[r, ipi]
    if closure and var > 0:
        al = mean + np.sqrt(var) * GHX
        pv = np.interp(al, AG, row)
        mp = float(GHW @ pv)
        cov = float(GHW @ ((al - mean) * (pv - mp)))
        mean2 = mean + gamma * (A0 - mp)
        var2 = max(var + gamma * gamma * mp * (1 - mp) - 2 * gamma * cov, 0.0)
    else:
        mp = float(np.interp(mean, AG, row))
        mean2 = mean + gamma * (A0 - mp)
        var2 = (var + gamma * gamma * mp * (1 - mp)) if closure else 0.0
    return mean2, var2, mp


def periodic_prediction(T, gamma, m, W, closure=True, tol=1e-11, max_cycles=3000):
    """Periodic steady state of the recursion with the window-composition p-tables.
    Returns (miss_1, miss_2) block averages."""
    t0 = int(np.ceil(W / (2 * m))) * 2 * m          # first time with a full window, phase aligned
    pis = [pi_index(window_share(t0 + k, m, W)) for k in range(2 * m)]
    mean, var = A0, (gamma * A0 * (1 - A0) / 2 if closure else 0.0)
    prev = None
    for it in range(max_cycles):
        mps = np.zeros(2 * m)
        for k in range(2 * m):
            r = 0 if k < m else 1
            mean, var, mps[k] = step_closure(T, r, pis[k], mean, var, gamma, closure)
        if prev is not None and abs(mean - prev[0]) < tol and abs(var - prev[1]) < tol:
            break
        prev = (mean, var)
    return float(mps[:m].mean()), float(mps[m:].mean())


# ----------------------------------------------------------------------------
# Part A: rolling-window Monte Carlo (the paper's ACI class, vectorised over replications)
# ----------------------------------------------------------------------------

def rolling_aci_mc(rho, gamma, m, W=360, reps=64, seed=0, R_override=None, t0_override=None, burn_override=None, record=False):
    rng = np.random.default_rng(seed)
    period = 2 * m
    burn = int(np.ceil(max(30.0 / gamma, 20 * period) / period)) * period
    meas = int(np.ceil(max(30 * period, 20.0 / gamma) / period)) * period
    t0 = int(np.ceil(W / period)) * period
    T = t0 + burn + meas
    sig = np.where(((np.arange(T) // m) % 2) == 0, 1.0, rho)
    R = np.abs(rng.standard_normal((reps, T))) * sig[None, :]
    if R_override is not None:
        R = R_override
        T = R.shape[1]
        t0 = t0_override
        burn = burn_override
        reps = R.shape[0]
    miss_log = np.zeros((reps, T)) if record else None
    alpha = np.full(reps, A0)
    miss_sum = np.zeros((2, reps))
    cnt = np.zeros(2)
    cdf_cache = {}
    for t in range(t0, T):
        lo = max(1, t - W)
        pool = R[:, lo:t]
        n = pool.shape[1]
        if n not in cdf_cache:
            cdf_cache[n] = np.cumsum(np.ones(n)) / n
        cdf = cdf_cache[n]
        S = np.sort(pool, axis=1)
        level = np.clip(1.0 - alpha, 0.0, 1.0)
        k = np.minimum(np.searchsorted(cdf, level, side="left"), n - 1)
        q = S[np.arange(reps), k]
        miss = np.where(alpha <= 0.0, 0.0, np.where(alpha >= 1.0, 1.0, (R[:, t] > q).astype(float)))
        if t >= t0 + burn:
            r = 0 if ((t // m) % 2) == 0 else 1
            miss_sum[r] += miss
            cnt[r] += 1
        if record:
            miss_log[:, t] = miss
        alpha = alpha + gamma * (A0 - miss)
    if record:
        return miss_log
    per = miss_sum / cnt[:, None]
    return {"roll_miss1": float(per[0].mean()), "roll_miss2": float(per[1].mean()),
            "roll_se1": float(per[0].std(ddof=1) / np.sqrt(reps)),
            "roll_se2": float(per[1].std(ddof=1) / np.sqrt(reps)),
            "roll_steps": int(T - t0), "roll_reps": reps}


_TG = {}


def part_a_cell(args):
    gamma, m, rho = args
    t0 = time.time()
    if rho not in _TG:
        laws, qmax = gaussian_laws(rho)
        _TG[rho] = build_ptables(laws, qmax)
    T = _TG[rho]
    mc = rolling_aci_mc(rho, gamma, m, seed=int(1e6 * gamma) + 11 * m + int(10 * rho))
    mf = periodic_prediction(T, gamma, m, 360, closure=False)
    cl = periodic_prediction(T, gamma, m, 360, closure=True)
    pool = M.Pool((1.0, rho))
    f4 = M.f4_exact_chain(pool, gamma, m)
    row = {"gamma": gamma, "m": m, "rho": rho, "gamma_m": gamma * m,
           "WIN_MF_miss1": mf[0], "WIN_MF_miss2": mf[1], "WIN_CL_miss1": cl[0], "WIN_CL_miss2": cl[1],
           "IDEAL_F4_miss1": f4[0], "IDEAL_F4_miss2": f4[1], "seconds": time.time() - t0}
    row.update(mc)
    for tag in ["WIN_MF", "WIN_CL", "IDEAL_F4"]:
        row[f"err_{tag}"] = max(abs(row[f"{tag}_miss1"] - row["roll_miss1"]),
                                abs(row[f"{tag}_miss2"] - row["roll_miss2"]))
    return row


def part_a():
    cells = list(itertools.product([0.005, 0.01, 0.02, 0.05], [50, 150, 450], [1.5, 2.0, 3.0]))
    with Pool(2) as p:
        rows = []
        for i, r in enumerate(p.imap(part_a_cell, cells)):
            rows.append(r)
            print(f"A [{i + 1}/36] g={r['gamma']} m={r['m']} rho={r['rho']} roll=({r['roll_miss1']:.4f},{r['roll_miss2']:.4f}) "
                  f"WIN_CL=({r['WIN_CL_miss1']:.4f},{r['WIN_CL_miss2']:.4f}) IDEAL=({r['IDEAL_F4_miss1']:.4f},{r['IDEAL_F4_miss2']:.4f}) "
                  f"{r['seconds']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(RES, "lead4_rolling_cells.csv")
    if os.path.exists(out):
        raise SystemExit("exists")
    df.to_csv(out, index=False)
    for tag in ["WIN_MF", "WIN_CL", "IDEAL_F4"]:
        e = df[f"err_{tag}"]
        print(f"{tag}: within 0.01 in {(e <= 0.01).sum()} of {len(e)} cells; median err {e.median():.4f}; max {e.max():.4f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "a"
    if which == "a":
        part_a()
