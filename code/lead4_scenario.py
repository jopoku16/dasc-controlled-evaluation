"""
lead4_scenario.py: does the formula explain the paper's simulated design
'recurring_freq_noise' (lag-1 forecaster, sinusoid plus noise, blocks of 150, pool window 360)?

Read-only use of datasets, forecasters, dasc_lib.

B1  re-run ACI (free and projected) at several step sizes on seeds 0-99 with the paper's
    code path (start = 220, window 360, first_valid 1) and compare with the stored CSVs.
B2  predictions: (a) idealised steady state (rho = 3, population pool, no window),
    (b) window-aware transient closure with the empirical residual laws of the design,
    started at alpha = alpha_0 at t = 220 exactly as in the runs.
B3  weighted quantile (fixed level): kernel weights of the paper's spectral-fixed method
    (h = 0.35, 32-point feature window, pool 360) give the realised weight mass on the
    current regime; the weighted-quantile formula (ii) is evaluated at those masses and
    compared with the paper's spectral_fixed regime coverage.

Outputs: results/lead4_scenario_check.csv, results/lead4_weights.csv
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(os.path.dirname(HERE), "results")

import datasets as ds  # noqa: E402
import forecasters as fc  # noqa: E402
import lead4_link as L  # noqa: E402
import lead4_model as M  # noqa: E402
from dasc_lib import ACI, make_method, run_stream, spectral_feature_matrix  # noqa: E402

BLOCK = 150
START = 220
W = 360
A0 = 0.1
SEEDS = list(range(100))


def regime_cov(out, regime):
    t = out["t"]
    miss = out["miss"].astype(float)
    reg = regime[t]
    return (1 - miss[reg == "R1"].mean(), 1 - miss[reg == "R2"].mean())


def rerun_aci(gamma, clip, seeds=SEEDS):
    c1, c2 = [], []
    for s in seeds:
        sim = ds.make_sim("recurring_freq_noise", s)
        y, regime = sim["y"], sim["regime"]
        pred = fc.seasonal_naive(y, 1)
        out = run_stream(y, pred, ACI(alpha=0.1, window=W, first_valid=1, gamma=gamma, clip=clip), start=START)
        a, b = regime_cov(out, regime)
        c1.append(a)
        c2.append(b)
    return float(np.mean(c1)), float(np.mean(c2)), float(np.std(c1, ddof=1) / np.sqrt(len(c1))), float(np.std(c2, ddof=1) / np.sqrt(len(c2)))


def empirical_laws_of_design(n=1_200_000, seed=12345):
    sim = ds.make_sim  # noqa: F841
    y, regime = ds.sim_recurring_freq_noise(n=n, seed=seed)
    res = np.abs(np.diff(y))                      # residual at t = |y_t - y_{t-1}|, t = 1..n-1
    reg = regime[1:]
    reg_prev = regime[:-1]
    ok = reg == reg_prev                          # drop the one residual at each block boundary
    s1 = res[ok & (reg == "R1")]
    s2 = res[ok & (reg == "R2")]
    return s1, s2


def transient_prediction(T, gamma, closure, n=1400):
    """Mean-field (closure=False) or moment closure (closure=True) from alpha = alpha_0 at t = START.
    Returns predicted coverage in R1 and R2 over t = START..n-1 (with the true window composition)."""
    mean, var = A0, 0.0
    mp1, mp2 = [], []
    for t in range(START, n):
        r = (t // BLOCK) % 2
        ipi = L.pi_index(L.window_share(t, BLOCK, W))
        mean, var, mp = L.step_closure(T, r, ipi, mean, var, gamma, closure)
        (mp1 if r == 0 else mp2).append(mp)
    return 1 - float(np.mean(mp1)), 1 - float(np.mean(mp2))


def transient_prediction_path(T, gamma, closure, pi_path, n=1400):
    """Same as transient_prediction but with the weight mass on regime 1, pi_path[t], given per t
    (mean over seeds of the realised kernel mass; Jensen approximation named in the report)."""
    mean, var = A0, 0.0
    mp1, mp2 = [], []
    for t in range(START, n):
        r = (t // BLOCK) % 2
        ipi = L.pi_index(pi_path[t])
        mean, var, mp = L.step_closure(T, r, ipi, mean, var, gamma, closure)
        (mp1 if r == 0 else mp2).append(mp)
    return 1 - float(np.mean(mp1)), 1 - float(np.mean(mp2))


def kernel_weight_masses(seeds, h=0.35, fw=32):
    """Realised weight mass on regime-1 residuals in the pool at each evaluated t."""
    rows = []
    for s in seeds:
        sim = ds.make_sim("recurring_freq_noise", s)
        y, regime = sim["y"], sim["regime"]
        feats = spectral_feature_matrix(y, fw)
        n = len(y)
        for t in range(START, n):
            lo = max(1, t - W)
            idx = np.arange(lo, t)
            d = np.sqrt(((feats[idx] - feats[t]) ** 2).sum(axis=1))
            wu = np.exp(-(d ** 2) / (h ** 2))
            if wu.sum() == 0:
                wu = np.ones_like(wu)
            w1 = float(wu[regime[idx] == "R1"].sum() / wu.sum())
            rows.append((s, t, (t // BLOCK) % 2, w1, float(np.mean(regime[idx] == "R1"))))
    return pd.DataFrame(rows, columns=["seed", "t", "r", "w_mass_R1", "share_R1"])


def main():
    t0 = time.time()
    stored_sens = pd.read_csv(os.path.join(RES, "sim_gamma_sensitivity.csv"))
    stored_reg = pd.read_csv(os.path.join(RES, "sim_regime_summary.csv"))
    sel = lambda var: stored_sens[(stored_sens.scenario == "recurring_freq_noise") & (stored_sens.forecaster == "naive1")
                                  & (stored_sens.variant == var)].iloc[0]
    reg = lambda meth: stored_reg[(stored_reg.scenario == "recurring_freq_noise") & (stored_reg.forecaster == "naive1")
                                  & (stored_reg.method == meth)].set_index("regime")["coverage"]

    # empirical laws and tables
    s1, s2 = empirical_laws_of_design()
    print("empirical residual sd R1, R2:", s1.std(), s2.std(), " ratio:", s2.std() / s1.std(),
          " 0.9-quantiles:", np.quantile(s1, 0.9), np.quantile(s2, 0.9), flush=True)
    laws, qmax = L.empirical_laws(s1, s2)
    Temp = L.build_ptables(laws, qmax)
    print("tables built", time.time() - t0, flush=True)
    glaws, gq = L.gaussian_laws(3.0)
    Tg = L.build_ptables(glaws, gq)

    pool3 = M.Pool((1.0, 3.0))
    rows = []
    for gamma, clip, label, stored in [
        (0.01, False, "aci_noclip gamma=0.01 (selected)", (float(reg("aci_noclip")["R1"]), float(reg("aci_noclip")["R2"]))),
        (0.015, False, "aci_noclip gamma=0.015", (float(sel("aci_noclip_gamma015").cov_R1), float(sel("aci_noclip_gamma015").cov_R2))),
        (0.015, True, "aci_clip gamma=0.015", (float(sel("aci_clip_gamma015").cov_R1), float(sel("aci_clip_gamma015").cov_R2))),
        (0.05, True, "aci_clip gamma=0.05 (selected)", (float(reg("aci_clip")["R1"]), float(reg("aci_clip")["R2"]))),
        (0.05, False, "aci_noclip gamma=0.05", (np.nan, np.nan)),
        (0.03, False, "aci_noclip gamma=0.03", (np.nan, np.nan)),
    ]:
        c1, c2, se1, se2 = rerun_aci(gamma, clip)
        # idealised steady state at rho = 3, m = 150
        f4 = M.f4_exact_chain(pool3, gamma, BLOCK)
        mf = transient_prediction(Temp, gamma, False)
        cl = transient_prediction(Temp, gamma, True)
        clg = transient_prediction(Tg, gamma, True)
        rows.append({"variant": label, "gamma": gamma, "clip": clip,
                     "stored_cov_R1": stored[0], "stored_cov_R2": stored[1],
                     "rerun_cov_R1": c1, "rerun_cov_R2": c2, "rerun_se1": se1, "rerun_se2": se2,
                     "ideal_F4_cov_R1": 1 - f4[0], "ideal_F4_cov_R2": 1 - f4[1],
                     "window_MF_empirical_cov_R1": mf[0], "window_MF_empirical_cov_R2": mf[1],
                     "window_CL_empirical_cov_R1": cl[0], "window_CL_empirical_cov_R2": cl[1],
                     "window_CL_gaussian_cov_R1": clg[0], "window_CL_gaussian_cov_R2": clg[1]})
        print(label, "rerun", round(c1, 4), round(c2, 4), "stored", stored, "ideal", round(1 - f4[0], 4), round(1 - f4[1], 4),
              "windowCL", round(cl[0], 4), round(cl[1], 4), flush=True)
    sc = pd.DataFrame(rows)
    out = os.path.join(RES, "lead4_scenario_check.csv")
    if os.path.exists(out):
        raise SystemExit("exists")
    sc.to_csv(out, index=False)

    # B3: weights
    wdf = kernel_weight_masses(range(20))
    wdf["w_same"] = np.where(wdf["r"] == 0, wdf["w_mass_R1"], 1 - wdf["w_mass_R1"])
    wdf["share_same"] = np.where(wdf["r"] == 0, wdf["share_R1"], 1 - wdf["share_R1"])
    summ = wdf.groupby("r")[["w_same", "share_same"]].mean()
    print(summ)
    # predicted fixed-level miss with the realised masses
    pred = {0: [], 1: []}
    for r, w1 in zip(wdf["r"], wdf["w_mass_R1"]):
        ipi = L.pi_index(w1)
        pred[int(r)].append(float(np.interp(A0, L.AG, Temp[int(r), ipi])))
    p1 = 1 - np.mean(pred[0])
    p2 = 1 - np.mean(pred[1])
    # unweighted-window prediction (the rolling pool alone)
    predu = {0: [], 1: []}
    for r, s1v in zip(wdf["r"], wdf["share_R1"]):
        ipi = L.pi_index(s1v)
        predu[int(r)].append(float(np.interp(A0, L.AG, Temp[int(r), ipi])))
    # re-run the paper's spectral_fixed
    c1, c2 = [], []
    for s in range(100):
        sim = ds.make_sim("recurring_freq_noise", s)
        y, regime = sim["y"], sim["regime"]
        pr = fc.seasonal_naive(y, 1)
        fm = spectral_feature_matrix(y, 32)
        m = make_method("spectral_fixed", alpha=0.10, first_valid=1, window=360, h=0.35)
        out_s = run_stream(y, pr, m, features=fm, start=START)
        a, b = regime_cov(out_s, regime)
        c1.append(a)
        c2.append(b)
    wrows = [{"quantity": "spectral_fixed regime coverage R1, R2: paper CSV", "R1": float(reg("spectral_fixed")["R1"]), "R2": float(reg("spectral_fixed")["R2"])},
             {"quantity": "spectral_fixed re-run (100 seeds)", "R1": float(np.mean(c1)), "R2": float(np.mean(c2))},
             {"quantity": "formula (ii) at the realised kernel masses (20 seeds)", "R1": p1, "R2": p2},
             {"quantity": "formula at the unweighted window composition (rolling conformal, fixed level)", "R1": 1 - np.mean(predu[0]), "R2": 1 - np.mean(predu[1])},
             {"quantity": "paper rolling conformal coverage CSV", "R1": float(reg("rolling")["R1"]), "R2": float(reg("rolling")["R2"])},
             {"quantity": "mean same-regime kernel weight mass w (regime 1, regime 2)", "R1": float(summ.loc[0, "w_same"]), "R2": float(summ.loc[1, "w_same"])},
             {"quantity": "mean same-regime share in the unweighted window", "R1": float(summ.loc[0, "share_same"]), "R2": float(summ.loc[1, "share_same"])}]
    # spectral ACI (weights plus the level update) with the mean realised kernel mass as pi(t)
    pi_path = wdf.groupby("t")["w_mass_R1"].mean().to_dict()
    for g, stored in [(0.015, (float(sel("spectral_aci_gamma015").cov_R1), float(sel("spectral_aci_gamma015").cov_R2))),
                      (0.03, (float(reg("spectral_aci")["R1"]), float(reg("spectral_aci")["R2"])))]:
        cl = transient_prediction_path(Temp, g, True, pi_path)
        wrows.append({"quantity": f"spectral ACI gamma={g}: stored CSV (R1, R2)", "R1": stored[0], "R2": stored[1]})
        wrows.append({"quantity": f"spectral ACI gamma={g}: closure with mean kernel mass pi(t)", "R1": cl[0], "R2": cl[1]})
    pd.DataFrame(wrows).to_csv(os.path.join(RES, "lead4_weights.csv"), index=False)
    print(pd.DataFrame(wrows).to_string(index=False))
    print("total seconds", time.time() - t0)


if __name__ == "__main__":
    main()
